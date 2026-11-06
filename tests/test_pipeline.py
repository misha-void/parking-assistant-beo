"""Stage 4 - integration tests for the LangGraph orchestration.

The LLM calls are faked, so these exercise the graph's wiring: slot collection over
interrupts, the forgiving confirmation step, cancelling, mid-booking questions, and the
hand-off to the approval agent.
"""

import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from parking_assistant.db.models import (
    ParkingLocation,
    ParkingType,
    PriceRule,
    Reservation,
    ReservationStatus,
    max_stay_minutes,
)
from parking_assistant.graph.pipeline import _store_moment, run_turn
from parking_assistant.graph.slots import ReservationSlots

PIPELINE = "parking_assistant.graph.pipeline"


class FakeRAGChain:
    """Routes anything mentioning "book" into the reservation flow, else answers."""

    def __init__(self, llm, classifier_llm=None, enable_guardrails=True):
        pass

    async def run(self, query, history=None, remembered_locations=None, note=""):
        if "book" in query.lower():
            return {"intent": "book_spot", "response": None, "locations": None}
        return {
            "intent": "general_info",
            "response": "Zone A allows a maximum of 30 minutes.",
            "locations": None,
        }

    def close(self):
        pass


async def fake_interpret(slot, reply, history, llm, locations=None, start_iso=None):
    """Anything ending in "?" is a question; otherwise the reply is the answer."""
    if reply.strip().endswith("?"):
        return {"kind": "question", "value": ""}
    if slot == "start_iso":
        return {"kind": "answer", "value": "2030-01-01T10:00:00"}
    if slot == "end_iso":
        return {"kind": "answer", "value": "2030-01-01T18:00:00"}
    if slot == "location_id":
        return {"kind": "answer", "value": ""}  # falls through to the numbered pick
    return {"kind": "answer", "value": reply.strip()}


@contextmanager
def patched_pipeline(review=("confirm", "Looks fine.")):
    """Patch every outbound call the graph makes, yielding the mocks worth asserting on."""
    escalate = AsyncMock()
    reviewer = AsyncMock(return_value=review)
    with patch(f"{PIPELINE}.RAGChain", FakeRAGChain), \
         patch(f"{PIPELINE}.get_chat_model", lambda *a, **k: None), \
         patch(f"{PIPELINE}.get_extraction_model", lambda *a, **k: None), \
         patch(f"{PIPELINE}.interpret_slot_reply", new=fake_interpret), \
         patch(f"{PIPELINE}.infer_location_from_conversation", new=AsyncMock(return_value=None)), \
         patch(f"{PIPELINE}.escalate_to_admin", new=escalate), \
         patch(f"{PIPELINE}.auto_approve_enabled", return_value=True), \
         patch(f"{PIPELINE}.auto_review_reservation", new=reviewer):
        yield {"escalate": escalate, "review": reviewer}


async def drive_to_confirmation(session_id: str) -> str:
    """Fill every slot; returns the confirmation summary shown to the user."""
    await run_turn(session_id, "I want to book")
    for reply in ("John", "Doe", "BG123AB", "1", "tomorrow at 10am"):
        await run_turn(session_id, reply)
    return await run_turn(session_id, "tomorrow at 6pm")


def delete_test_reservation(db_session):
    """Remove the reservation the booking flow created, if any."""
    reservation = (
        db_session.query(Reservation)
        .filter_by(car_number="BG123AB", user_surname="Doe")
        .order_by(Reservation.id.desc())
        .first()
    )
    if reservation:
        db_session.delete(reservation)
        db_session.commit()


@pytest.mark.asyncio
async def test_booking_flow_saves_then_the_approval_agent_decides(db_session):
    assert db_session.query(ParkingLocation).first() is not None
    session_id = f"test-{uuid.uuid4()}"

    with patched_pipeline() as mocks:
        assert "first name" in (await run_turn(session_id, "I want to book a spot")).lower()
        for reply in ("John", "Doe", "BG123AB", "1", "tomorrow at 10am"):
            await run_turn(session_id, reply)
        summary = await run_turn(session_id, "tomorrow at 6pm")
        assert "BG123AB" in summary
        # "yes please" used to cancel the whole booking; it must be accepted.
        final = await run_turn(session_id, "yes please")

    assert "approved" in final.lower()
    assert "looks fine" in final.lower()
    mocks["escalate"].assert_awaited_once()
    mocks["review"].assert_awaited_once()

    reservation = (
        db_session.query(Reservation)
        .filter_by(car_number="BG123AB", user_surname="Doe")
        .order_by(Reservation.id.desc())
        .first()
    )
    assert reservation is not None
    assert reservation.status == ReservationStatus.PENDING_APPROVAL  # the reviewer is mocked
    delete_test_reservation(db_session)


@pytest.mark.asyncio
async def test_refusal_by_the_approval_agent_is_reported(db_session):
    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline(review=("refuse", "The period looks implausible.")):
        await drive_to_confirmation(session_id)
        final = await run_turn(session_id, "yes")

    assert "refused" in final.lower()
    assert "implausible" in final.lower()
    delete_test_reservation(db_session)


@pytest.mark.asyncio
async def test_unclear_confirmation_asks_again_instead_of_cancelling(db_session):
    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline():
        await drive_to_confirmation(session_id)
        reply = await run_turn(session_id, "hmm I am not sure")

        assert "yes" in reply.lower()
        assert "cancelled" not in reply.lower()
        # The booking survives, so confirming afterwards still works.
        final = await run_turn(session_id, "yes")
        assert "approved" in final.lower()

    delete_test_reservation(db_session)


@pytest.mark.asyncio
async def test_saying_no_at_confirmation_cancels():
    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline():
        await drive_to_confirmation(session_id)
        assert "cancelled" in (await run_turn(session_id, "no")).lower()


@pytest.mark.asyncio
async def test_cancel_mid_booking_abandons_the_reservation():
    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline():
        await run_turn(session_id, "I want to book")
        assert "cancelled" in (await run_turn(session_id, "cancel")).lower()
        # A fresh thread: the next booking starts from the first slot again.
        assert "first name" in (await run_turn(session_id, "I want to book")).lower()


@pytest.mark.asyncio
async def test_question_mid_booking_is_answered_then_the_slot_is_asked_again():
    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline():
        await run_turn(session_id, "I want to book")
        reply = await run_turn(session_id, "wait, what are the zone A rules?")

    assert "30 minutes" in reply  # the aside was answered
    assert "first name" in reply.lower()  # and the slot re-asked in the same turn


@pytest.mark.asyncio
async def test_a_numbered_location_choice_skips_the_extraction_model():
    """Regression: the question numbers the options, the model only sees their names.

    It therefore cannot map "1" to a location, and used to answer "unclear", leaving the
    user stuck on a list they were invited to pick from by number. A digit now bypasses
    interpretation entirely. Forcing "unclear" for that slot proves the bypass.
    """
    async def unclear_for_location(slot, reply, history, llm, locations=None, start_iso=None):
        if slot == "location_id":
            return {"kind": "unclear", "value": ""}
        return await fake_interpret(slot, reply, history, llm, locations, start_iso)

    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline(), patch(f"{PIPELINE}.interpret_slot_reply", new=unclear_for_location):
        await run_turn(session_id, "I want to book")
        for reply in ("John", "Doe", "BG123AB"):
            await run_turn(session_id, reply)
        reply = await run_turn(session_id, "1")

    assert "when would you like to start" in reply.lower()
    assert "still need" not in reply.lower()


@pytest.mark.asyncio
async def test_the_location_list_is_shown_again_when_the_choice_fails(db_session):
    """The list used to appear only on the first ask, so a retry had nothing to pick from."""
    location_name = db_session.query(ParkingLocation).first().name

    async def unclear_for_location(slot, reply, history, llm, locations=None, start_iso=None):
        if slot == "location_id":
            return {"kind": "unclear", "value": ""}
        return await fake_interpret(slot, reply, history, llm, locations, start_iso)

    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline(), patch(f"{PIPELINE}.interpret_slot_reply", new=unclear_for_location):
        await run_turn(session_id, "I want to book")
        for reply in ("John", "Doe", "BG123AB"):
            await run_turn(session_id, reply)
        retry = await run_turn(session_id, "somewhere nice")

    assert "still need" in retry.lower()
    assert location_name in retry


@pytest.mark.asyncio
async def test_a_correction_at_confirmation_does_not_cancel_the_booking(db_session):
    """"no" alone cancels, but "no, change ..." is a correction and must keep the slots."""
    session_id = f"test-{uuid.uuid4()}"
    with patched_pipeline():
        await drive_to_confirmation(session_id)
        reply = await run_turn(session_id, "no, change the end time to 6pm")

        assert "cancelled" not in reply.lower()
        assert "yes" in reply.lower()  # asked again instead
        final = await run_turn(session_id, "yes")
        assert "approved" in final.lower()

    delete_test_reservation(db_session)


def test_an_over_long_stay_is_refused_while_booking():
    """The zone max stay is enforced as the end time is given, not only at approval."""
    slots = ReservationSlots(start_iso="2030-01-01T10:00:00")

    message = _store_moment("end_iso", "2030-01-01T18:00:00", slots, max_stay=180)
    assert "maximum stay of 3 hours" in message
    assert slots.end_iso is None  # nothing stored, so the slot is asked again

    assert _store_moment("end_iso", "2030-01-01T13:00:00", slots, max_stay=180) == ""
    assert slots.end_iso == "2030-01-01T13:00:00"


def test_an_uncapped_location_takes_a_long_booking():
    """Garages carry a zone for pricing only, so the street cap must not apply to them."""
    slots = ReservationSlots(start_iso="2030-01-01T10:00:00")
    assert _store_moment("end_iso", "2030-01-01T18:00:00", slots, max_stay=None) == ""


def test_the_max_stay_rule_caps_street_parking_but_not_garages(db_session):
    limits = {rule.zone: rule.max_duration_minutes for rule in db_session.query(PriceRule).all()}
    by_type = {}
    for location in db_session.query(ParkingLocation).all():
        by_type.setdefault(location.type, location)

    street = by_type.get(ParkingType.CAR_PARK)
    garage = by_type.get(ParkingType.GARAGE)
    assert max_stay_minutes(street, limits) == limits[street.zone]
    assert max_stay_minutes(garage, limits) is None
