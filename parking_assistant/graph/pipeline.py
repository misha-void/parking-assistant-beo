"""
LangGraph pipeline - one resumable run per session: chat/RAG -> slot collection -> confirm -> save -> admin -> MCP.
"""

from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from parking_assistant.llm_factory import get_chat_model
from parking_assistant.rag.chain import RAGChain
from parking_assistant.graph.slot_collector import (
    SlotCollector,
    ReservationSlots,
    SLOT_LABELS,
    parse_datetime_with_llm,
    interpret_slot_reply,
    infer_location_from_conversation,
)
from parking_assistant.rag.retriever import get_bookable_locations, price_label
from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import Reservation, ReservationStatus, ParkingLocation
from parking_assistant.admin.admin_agent import escalate_to_admin, get_reservation_status
from parking_assistant.session_registry import set_reservation


def _parse_iso(value: str):
    """Parse an ISO datetime string, or return None."""
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


class State(TypedDict, total=False):
    session_id: str
    user_input: str
    history: list        # prior [{"role","content"}] turns, for conversation memory
    locations: list      # parking options shown earlier, for follow-ups / booking context
    response: str
    intent: str
    slots: dict
    note: str            # hint prepended to the next collect prompt (errors, answered questions, acks)
    confirmed: bool
    reservation_id: int


async def route_intent(state: State) -> dict:
    # Answer real reservation-status questions from the DB (not the LLM's memory)
    rid = state.get("reservation_id")
    if rid and any(w in state["user_input"].lower()
                   for w in ("status", "approved", "confirm", "reject", "my reservation", "my booking")):
        status = get_reservation_status(rid)
        pretty = {
            "pending_approval": "still pending admin approval ⏳",
            "confirmed": "confirmed ✅",
            "cancelled": "refused by the admin ❌",
        }.get(status, status)
        if pretty:
            return {"response": f"Your reservation #{rid} is {pretty}.", "intent": "status"}

    chain = RAGChain(get_chat_model())
    try:
        result = await chain.run(state["user_input"], state.get("history"), state.get("locations"))
    finally:
        chain.close()

    if result["intent"] == "book_spot":
        # Carry the location the user discussed earlier (whole conversation, not just
        # this message), so we don't re-ask for it.
        slots = ReservationSlots()
        note = ""
        bookable = get_bookable_locations()
        matched_id = await infer_location_from_conversation(
            state.get("history"), state["user_input"], bookable, get_chat_model())
        loc = next((l for l in bookable if l["id"] == matched_id), None)
        if loc:
            slots.location_id, slots.location_preference = loc["id"], loc["name"]
            note = f"Great — booking at {loc['name']}."
        updates = {"intent": "book_spot", "slots": slots.model_dump()}
        if note:
            updates["note"] = note
        return updates

    # #2: remember any options we just showed, so later follow-ups can be answered from them
    updates = {"response": result["response"], "intent": result["intent"]}
    if result.get("locations"):
        updates["locations"] = result["locations"]
    return updates


async def collect(state: State) -> dict:
    slots = ReservationSlots(**state["slots"])
    missing = slots.missing_slots()
    if not missing:
        return {"slots": slots.model_dump()}
    slot = missing[0]

    bookable = get_bookable_locations()  # [{id, name, address, price}]

    # Show the full location list (with prices) only on the FIRST ask (no pending note),
    # so we don't spam the whole list on every re-ask.
    prompt = SlotCollector.SLOT_PROMPTS[slot]
    if slot == "location" and not state.get("note"):
        prompt += "\n\n" + "\n".join(
            f"{i+1}. {l['name']} - {l['address']} - {price_label(l['price'])}"
            for i, l in enumerate(bookable))
    prompt = (state.get("note") or "") + ("\n\n" if state.get("note") else "") + prompt

    answer = interrupt(prompt)  # pauses here; resume value is the user's reply
    llm = get_chat_model()

    # One LLM call: classify answer vs question AND normalize the value in context.
    res = await interpret_slot_reply(
        slot, answer, state.get("history"), llm,
        locations=bookable, start_iso=slots.start_date,
    )

    # A question/aside: answer it concisely, then re-ask the same slot (slot stays empty).
    if res["kind"] == "question":
        chain = RAGChain(llm)
        try:
            ans = await chain.run(answer, state.get("history"), state.get("locations"))
        finally:
            chain.close()
        return {"note": ans.get("response") or "Let's continue with your booking."}

    value = res["value"]
    note = ""

    if slot in ("name", "surname"):
        if value:
            setattr(slots, slot, value)
        else:
            note = f"Sorry, what's your {SLOT_LABELS[slot]}?"
    elif slot == "car_number":
        car = value.upper().replace(" ", "")
        if SlotCollector().validate_car_number(car):
            slots.car_number = car
        else:
            note = "That doesn't look like a valid plate (e.g. BG123AB). What's your car number?"
    elif slot == "location":
        loc = next((l for l in bookable if l["name"].lower() == value.lower()), None) if value else None
        if not loc and answer.strip().isdigit():  # allow a raw number pick
            idx = int(answer.strip()) - 1
            loc = bookable[idx] if 0 <= idx < len(bookable) else None
        if loc:
            slots.location_id, slots.location_preference = loc["id"], loc["name"]
        else:
            note = "Which location would you like? Give the name or the number."
    else:  # start_date / end_date
        parsed = _parse_iso(value) or await parse_datetime_with_llm(answer, llm)
        if parsed and slot == "end_date" and slots.start_date and parsed <= _parse_iso(slots.start_date):
            note, parsed = "The end time must be after the start time. When should it end?", None
        if parsed:
            setattr(slots, slot, parsed.isoformat())
        elif not note:
            note = "I couldn't understand that time. Try e.g. 'tomorrow at 5pm'."

    return {"slots": slots.model_dump(), "note": note}


async def confirm(state: State) -> dict:
    slots = ReservationSlots(**state["slots"])
    loc = next((l for l in get_bookable_locations() if l["id"] == slots.location_id), None)
    location_name = loc["name"] if loc else "the selected location"

    summary = SlotCollector().format_confirmation(slots, location_name)

    # Insert price + estimated cost (before the yes/no line) so the user sees everything
    import math
    price = loc["price"] if loc else None
    start, end = _parse_iso(slots.start_date), _parse_iso(slots.end_date)
    if price and start and end and end > start:
        hours = max(1, math.ceil((end - start).total_seconds() / 3600))
        price_line = f"💶 Price: {price} RSD/hour · ~{hours}h → est. {price * hours} RSD"
    else:
        price_line = f"💶 Price: {price_label(price)}"
    summary = summary.replace("\nIs this information correct?",
                              f"\n{price_line}\n\nIs this information correct?")

    ans = interrupt(summary)
    if ans.strip().lower() in {"yes", "confirm", "correct", "ok"}:
        return {"confirmed": True}
    return {"confirmed": False, "response": "Reservation cancelled. Let me know if you'd like to try again!"}


async def save(state: State) -> dict:
    slots = ReservationSlots(**state["slots"])
    session = SessionLocal()
    try:
        reservation = Reservation(
            user_name=slots.name,
            user_surname=slots.surname,
            car_number=slots.car_number,
            location_id=slots.location_id,
            start_datetime=datetime.fromisoformat(slots.start_date),
            end_datetime=datetime.fromisoformat(slots.end_date),
            status=ReservationStatus.PENDING_APPROVAL,
        )
        session.add(reservation)
        session.commit()
        session.refresh(reservation)
        set_reservation(state.get("session_id", ""), reservation.id)  # for UI status polling
        return {"reservation_id": reservation.id}
    finally:
        session.close()


async def admin(state: State) -> dict:
    # Escalate to a human admin (email, or console fallback). The admin confirms/refuses
    # out of band via the admin app; the MCP recording happens on their confirmation.
    rid = state["reservation_id"]
    await escalate_to_admin(rid, get_chat_model())
    return {"response": (
        f"✅ Reservation #{rid} submitted for approval. "
        "An admin will review and confirm it shortly — thank you!"
    )}


def _build() -> StateGraph:
    g = StateGraph(State)
    for node in (route_intent, collect, confirm, save, admin):
        g.add_node(node.__name__, node)
    g.add_edge(START, "route_intent")
    g.add_conditional_edges("route_intent", lambda s: "collect" if s["intent"] == "book_spot" else END)
    g.add_conditional_edges(
        "collect",
        lambda s: "confirm" if ReservationSlots(**s["slots"]).is_complete() else "collect",
    )
    g.add_conditional_edges("confirm", lambda s: "save" if s["confirmed"] else END)
    g.add_edge("save", "admin")
    g.add_edge("admin", END)
    return g


_graph = _build().compile(checkpointer=MemorySaver())

# Cancelling bumps a per-session generation, so the next message gets a fresh thread
_generation: dict[str, int] = {}


async def run_turn(session_id: str, user_input: str, history: list = None) -> str:
    config = {"configurable": {"thread_id": f"{session_id}:{_generation.get(session_id, 0)}"}}
    snapshot = _graph.get_state(config)
    if snapshot.next:  # paused at an interrupt
        if user_input.strip().lower() in {"cancel", "stop", "quit", "nevermind"}:
            _generation[session_id] = _generation.get(session_id, 0) + 1
            return "Reservation cancelled. How else can I help you?"
        out = await _graph.ainvoke(Command(resume=user_input), config)
    else:
        out = await _graph.ainvoke(
            {"session_id": session_id, "user_input": user_input, "history": history or [], "slots": {}},
            config,
        )

    # If the graph paused again, return the interrupt prompt (two API shapes across langgraph versions)
    if out.get("__interrupt__"):
        return out["__interrupt__"][0].value
    for task in _graph.get_state(config).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return out.get("response", "")
