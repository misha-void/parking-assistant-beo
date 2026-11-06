"""LangGraph pipeline: one resumable graph run per chat session.

    route_message ─┬─ answer (info / status) ─────────────────────────────▶ END
                   └─ collect_slot* ─▶ confirm_with_user ─▶ save_reservation
                                                          ─▶ request_approval ─▶ END

Every question put to the user is a LangGraph ``interrupt()``; the next message resumes
the run from its checkpoint, so the UI holds no conversation logic of its own.
"""

import logging
import math
import re
from datetime import datetime
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from parking_assistant.admin.admin_agent import (
    auto_approve_enabled,
    auto_review_reservation,
    escalate_to_admin,
    get_reservation_status,
)
from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import Reservation, ReservationStatus
from parking_assistant.graph.slots import (
    SLOT_LABELS,
    SLOT_PROMPTS,
    ReservationSlots,
    format_confirmation,
    infer_location_from_conversation,
    interpret_slot_reply,
    is_valid_name,
    is_valid_plate,
    parse_iso,
)
from parking_assistant.llm_factory import get_chat_model, get_extraction_model
from parking_assistant.rag.chain import RAGChain
from parking_assistant.rag.retriever import get_bookable_locations, price_label
from parking_assistant.session_registry import get_reservation, set_reservation

logger = logging.getLogger(__name__)

# Replies accepted at the confirmation step. Anything else re-asks rather than cancelling.
AFFIRMATIVE_REPLIES = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay",
    "confirm", "confirmed", "correct", "right", "go ahead",
}
# Just the refusals: run_turn intercepts "cancel", "stop" and friends before this point.
NEGATIVE_REPLIES = {"no", "n", "nope"}

# Typed while a booking is in progress, these abandon it.
CANCEL_REPLIES = {"cancel", "stop", "quit", "nevermind", "never mind", "abort"}

# Phrases that unambiguously ask about an existing reservation. Deliberately narrow:
# a bare "confirm" used to hijack questions like "can you confirm the price?".
STATUS_PHRASES = (
    "my reservation", "my booking", "reservation status", "booking status",
    "status of my", "approved yet", "any update", "is it confirmed",
)
MODIFY_WORDS = ("cancel", "change", "modify", "move", "delete", "reschedule")

STATUS_DESCRIPTIONS = {
    "pending_approval": "still awaiting approval ⏳",
    "confirmed": "confirmed ✅",
    "cancelled": "refused ❌",
}


class ChatState(TypedDict, total=False):
    """State carried by the graph and persisted per session by the checkpointer."""

    session_id: str
    user_input: str
    history: list           # prior [{"role", "content"}] turns
    locations: list         # options shown earlier, for follow-ups and booking context
    slots: dict             # serialised ReservationSlots
    prompt_prefix: str      # text prepended to the next question (errors, asides, acks)
    response: str
    intent: str
    confirm_decision: str   # "yes" | "no" | "unclear"
    reservation_id: int


def _is_status_question(text: str) -> bool:
    """True if the message asks about the user's existing reservation.

    The bare word "status" only counts in a short message: as a plain substring it
    hijacked questions like "what's the status of the Slavija garage, is it full?".
    """
    lowered = text.lower()
    if any(phrase in lowered for phrase in STATUS_PHRASES):
        return True
    return "status" in lowered and len(lowered.split()) <= 4


def _asks_to_modify(text: str) -> bool:
    """True if the message asks to change or cancel an existing reservation."""
    return any(word in text.lower() for word in MODIFY_WORDS)


def _describe_status(reservation_id: int) -> str:
    """Describe a reservation's current status, read from the database."""
    status = get_reservation_status(reservation_id)
    described = STATUS_DESCRIPTIONS.get(status, status or "unknown")
    return f"Your reservation #{reservation_id} is {described}."


async def _answer_with_rag(question: str, state: ChatState, note: str = "") -> dict:
    """Run the RAG chain for one question, with conversation and earlier options.

    The conversational model writes the answer, but intent routing goes through the
    temperature-0 model so the same message always takes the same branch.
    """
    chain = RAGChain(get_chat_model(), classifier_llm=get_extraction_model())
    try:
        return await chain.run(question, state.get("history"), state.get("locations"), note)
    finally:
        chain.close()


async def _start_booking(state: ChatState) -> dict:
    """Begin slot collection, pre-filling the location the conversation settled on."""
    slots = ReservationSlots()
    updates: dict = {"intent": "book_spot"}

    bookable = get_bookable_locations()
    location_id = await infer_location_from_conversation(
        state.get("history"), state["user_input"], bookable, get_extraction_model()
    )
    chosen = next((loc for loc in bookable if loc["id"] == location_id), None)
    if chosen:
        slots.location_id = chosen["id"]
        updates["prompt_prefix"] = f"Great — booking at {chosen['name']}."

    updates["slots"] = slots.model_dump()
    return updates


async def route_message(state: ChatState) -> dict:
    """Answer a status question, answer an info question, or start a booking."""
    message = state["user_input"]
    # Fall back to the registry: cancelling a booking starts a fresh checkpoint thread,
    # so graph state loses the id of a reservation made earlier in the session.
    reservation_id = state.get("reservation_id") or get_reservation(state.get("session_id", ""))

    if reservation_id and _is_status_question(message):
        if _asks_to_modify(message):
            return {
                "intent": "status",
                "response": (
                    f"Reservation #{reservation_id} has already been submitted, so I can't "
                    "change or cancel it here — please contact the parking administrator."
                ),
            }
        return {"intent": "status", "response": _describe_status(reservation_id)}

    note = ""
    if reservation_id:
        note = (
            f"The user has already submitted reservation #{reservation_id}. It cannot be "
            "changed or cancelled in this chat - if they ask to, tell them to contact the "
            "parking administrator."
        )
    result = await _answer_with_rag(message, state, note)
    if result["intent"] == "book_spot":
        return await _start_booking(state)

    updates = {"response": result["response"], "intent": result["intent"]}
    if result.get("locations"):
        updates["locations"] = result["locations"]
    return updates


def _question_for(slot: str, bookable: list, prefix: str) -> str:
    """Build the question for one slot, always listing the options for the location.

    The list is shown on retries too: a prefix here is an error or an aside, and the
    user needs the numbered options in front of them to answer again.
    """
    question = SLOT_PROMPTS[slot]
    if slot == "location_id":
        options = "\n".join(
            f"{number}. {loc['name']} - {loc['address']} - {price_label(loc['price'])}"
            for number, loc in enumerate(bookable, start=1)
        )
        question = f"{question}\n\n{options}"
    return f"{prefix}\n\n{question}" if prefix else question


def _match_location(value: str, reply: str, bookable: list) -> dict | None:
    """Match a location by its exact name, or by the number shown in the list."""
    if value:
        for loc in bookable:
            if loc["name"].lower() == value.lower():
                return loc
    if reply.strip().isdigit():
        index = int(reply.strip()) - 1
        if 0 <= index < len(bookable):
            return bookable[index]
    return None


def _store_moment(slot: str, value: str, slots: ReservationSlots) -> str:
    """Validate and store a start/end date-time. Returns "" or a retry message."""
    moment = parse_iso(value)
    if not moment:
        return "I couldn't read that time. Try something like 'tomorrow at 5pm'."
    if moment < datetime.now():
        return "That time is in the past. When would you like it instead?"
    if slot == "end_iso":
        start = parse_iso(slots.start_iso)
        if start and moment <= start:
            return "The end must be after the start. When should it end?"
    setattr(slots, slot, moment.isoformat())
    return ""


def _store_answer(slot: str, value: str, reply: str, slots: ReservationSlots,
                  bookable: list) -> str:
    """Validate and store one slot value. Returns "" or the text to re-ask with."""
    if slot in ("name", "surname"):
        if is_valid_name(value):
            setattr(slots, slot, value)
            return ""
        return f"Sorry, I didn't catch your {SLOT_LABELS[slot]}."

    if slot == "car_number":
        plate = value.upper().replace(" ", "")
        if is_valid_plate(plate):
            slots.car_number = plate
            return ""
        return "That doesn't look like a licence plate (for example BG123AB)."

    if slot == "location_id":
        chosen = _match_location(value, reply, bookable)
        if chosen:
            slots.location_id = chosen["id"]
            return ""
        return "I didn't recognise that location — give me its name or its number."

    return _store_moment(slot, value, slots)


async def collect_slot(state: ChatState) -> dict:
    """Ask for the next missing slot, then interpret and store the reply."""
    slots = ReservationSlots(**state["slots"])
    missing = slots.missing_slots()
    if not missing:
        return {"slots": slots.model_dump()}

    slot = missing[0]
    bookable = get_bookable_locations()
    reply = interrupt(_question_for(slot, bookable, state.get("prompt_prefix") or ""))

    if slot == "location_id" and reply.strip().isdigit():
        # The question numbers the options, but the extraction model is only shown their
        # names, so it cannot map "2" to one. _store_answer resolves the number instead.
        interpreted = {"kind": "answer", "value": ""}
    else:
        interpreted = await interpret_slot_reply(
            slot, reply, state.get("history"), get_extraction_model(),
            locations=bookable, start_iso=slots.start_iso,
        )

    if interpreted["kind"] == "question":
        answer = await _answer_with_rag(
            reply, state,
            "A booking is already in progress. Do not offer to book and do not end with "
            "an offer of further help - the next question follows your answer.",
        )
        return {"prompt_prefix": answer.get("response") or "Let's carry on with your booking."}
    if interpreted["kind"] == "unclear":
        return {"prompt_prefix": f"Sorry, I still need your {SLOT_LABELS[slot]}."}

    prefix = _store_answer(slot, interpreted["value"], reply, slots, bookable)
    return {"slots": slots.model_dump(), "prompt_prefix": prefix}


def _price_line(slots: ReservationSlots, location: dict | None) -> str:
    """Hourly price plus an estimate for the requested period."""
    price = location["price"] if location else None
    start, end = parse_iso(slots.start_iso), parse_iso(slots.end_iso)
    if price and start and end and end > start:
        hours = max(1, math.ceil((end - start).total_seconds() / 3600))
        return f"💶 Price: {price} RSD/hour · ~{hours}h → est. {price * hours} RSD"
    return f"💶 Price: {price_label(price)}"


def _normalise_reply(reply: str) -> tuple[str, list[str]]:
    """Return a reply lowercased and stripped of end punctuation, plus its words."""
    cleaned = reply.strip().lower().strip(".!,?")
    return cleaned, re.findall(r"[a-z']+", cleaned)


def _classify_reply(reply: str) -> str:
    """Read a confirmation reply as "yes", "no" or "unclear".

    A leading yes or no decides only a short reply, so "yes please" and "no thanks" are
    understood while "no, change the end time to 6pm" is a correction rather than a
    cancellation, and "not sure" is asked again instead of silently dropping the booking.
    """
    cleaned, words = _normalise_reply(reply)
    if cleaned in AFFIRMATIVE_REPLIES:
        return "yes"
    if cleaned in NEGATIVE_REPLIES:
        return "no"
    if len(words) <= 3:
        first_word = words[0] if words else ""
        if first_word in AFFIRMATIVE_REPLIES:
            return "yes"
        if first_word in NEGATIVE_REPLIES:
            return "no"
    return "unclear"


def _wants_to_cancel(reply: str) -> bool:
    """True if the user asked to abandon the booking, allowing "cancel please" etc."""
    cleaned, words = _normalise_reply(reply)
    return cleaned in CANCEL_REPLIES or (bool(words) and words[0] in CANCEL_REPLIES)


async def confirm_with_user(state: ChatState) -> dict:
    """Show the summary with an estimated price and wait for a yes or no."""
    slots = ReservationSlots(**state["slots"])
    chosen = next(
        (loc for loc in get_bookable_locations() if loc["id"] == slots.location_id), None
    )
    summary = format_confirmation(
        slots,
        chosen["name"] if chosen else "the selected location",
        _price_line(slots, chosen),
    )

    prefix = state.get("prompt_prefix") or ""
    reply = interrupt(f"{prefix}\n\n{summary}" if prefix else summary)

    decision = _classify_reply(reply)
    if decision == "yes":
        return {"confirm_decision": "yes", "prompt_prefix": ""}
    if decision == "no":
        return {
            "confirm_decision": "no",
            "response": "Reservation cancelled. Tell me if you'd like to start again.",
        }
    return {
        "confirm_decision": "unclear",
        "prompt_prefix": "Please reply 'yes' to confirm or 'no' to cancel.",
    }


async def save_reservation(state: ChatState) -> dict:
    """Persist the reservation as pending approval."""
    slots = ReservationSlots(**state["slots"])
    session = SessionLocal()
    try:
        reservation = Reservation(
            user_name=slots.name,
            user_surname=slots.surname,
            car_number=slots.car_number,
            location_id=slots.location_id,
            start_datetime=parse_iso(slots.start_iso),
            end_datetime=parse_iso(slots.end_iso),
            status=ReservationStatus.PENDING_APPROVAL,
        )
        session.add(reservation)
        session.commit()
        session.refresh(reservation)
        set_reservation(state.get("session_id", ""), reservation.id)
        return {"reservation_id": reservation.id}
    finally:
        session.close()


async def request_approval(state: ChatState) -> dict:
    """Notify the admin, then let the approval agent decide unless a human must."""
    reservation_id = state["reservation_id"]
    try:
        await escalate_to_admin(reservation_id, get_chat_model())
    except Exception:  # a failed notification must not lose a saved reservation
        logger.exception("Could not notify the admin about reservation %s", reservation_id)

    pending_message = (
        f"✅ Reservation #{reservation_id} submitted. An administrator will review it "
        "shortly — I'll let you know as soon as they decide."
    )
    if not auto_approve_enabled():
        return {"response": pending_message}

    decision, reason = await auto_review_reservation(reservation_id, get_extraction_model())
    if decision == "confirm":
        return {"response": f"✅ Reservation #{reservation_id} approved — {reason}"}
    if decision == "refuse":
        return {"response": f"❌ Reservation #{reservation_id} was refused — {reason}"}
    return {"response": pending_message}


def _after_collect(state: ChatState) -> str:
    """Keep collecting until every slot is filled."""
    if ReservationSlots(**state["slots"]).is_complete():
        return "confirm_with_user"
    return "collect_slot"


def _after_confirmation(state: ChatState) -> str:
    """Save on yes, stop on no, ask again if the reply was neither."""
    decision = state.get("confirm_decision")
    if decision == "yes":
        return "save_reservation"
    if decision == "no":
        return END
    return "confirm_with_user"


def _build_graph() -> StateGraph:
    """Wire the nodes and edges of the conversation graph."""
    graph = StateGraph(ChatState)
    for node in (route_message, collect_slot, confirm_with_user,
                 save_reservation, request_approval):
        graph.add_node(node.__name__, node)

    graph.add_edge(START, "route_message")
    graph.add_conditional_edges(
        "route_message",
        lambda state: "collect_slot" if state.get("intent") == "book_spot" else END,
    )
    graph.add_conditional_edges("collect_slot", _after_collect)
    graph.add_conditional_edges("confirm_with_user", _after_confirmation)
    graph.add_edge("save_reservation", "request_approval")
    graph.add_edge("request_approval", END)
    return graph


_compiled_graph = _build_graph().compile(checkpointer=MemorySaver())

# Cancelling starts a fresh thread for the session instead of mutating graph state.
_thread_generation: dict[str, int] = {}


def _thread_config(session_id: str) -> dict:
    """Checkpointer thread for this session's current conversation."""
    generation = _thread_generation.get(session_id, 0)
    return {"configurable": {"thread_id": f"{session_id}:{generation}"}}


def _pending_question(config: dict, result: dict) -> str | None:
    """The interrupt text the graph paused on, or None if the run finished."""
    interrupts = result.get("__interrupt__")
    if interrupts:
        return interrupts[0].value
    for task in _compiled_graph.get_state(config).tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


async def run_turn(session_id: str, user_input: str, history: list | None = None) -> str:
    """Advance the conversation by one user message and return the reply to show."""
    config = _thread_config(session_id)
    is_paused = bool(_compiled_graph.get_state(config).next)

    if is_paused:
        if _wants_to_cancel(user_input):
            _thread_generation[session_id] = _thread_generation.get(session_id, 0) + 1
            return "Reservation cancelled. How else can I help?"
        # Refresh the history too, so mid-booking asides see the latest turns.
        result = await _compiled_graph.ainvoke(
            Command(resume=user_input, update={"history": history or []}), config
        )
    else:
        result = await _compiled_graph.ainvoke(
            {
                "session_id": session_id,
                "user_input": user_input,
                "history": history or [],
                "slots": {},
                "prompt_prefix": "",
            },
            config,
        )

    return _pending_question(config, result) or result.get("response", "")
