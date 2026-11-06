"""Tests for Stage 2 - the admin approval agent and the human override path."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from parking_assistant.admin.admin_agent import (
    auto_review_reservation,
    decide_reservation,
    list_pending_reservations,
)
from parking_assistant.db.models import ParkingLocation, Reservation, ReservationStatus

MCP_WRITE = "parking_assistant.admin.admin_agent.send_confirmed_reservation"


@pytest.fixture
def pending_reservation(db_session):
    """A pending reservation for tomorrow, removed again afterwards."""
    location = db_session.query(ParkingLocation).first()
    start = datetime.now() + timedelta(days=1)
    reservation = Reservation(
        user_name="John",
        user_surname="Doe",
        car_number="BG123AB",
        location_id=location.id,
        start_datetime=start,
        end_datetime=start + timedelta(hours=2),
        status=ReservationStatus.PENDING_APPROVAL,
    )
    db_session.add(reservation)
    db_session.commit()
    yield reservation
    db_session.delete(reservation)
    db_session.commit()


def test_confirm_sets_status_and_records_to_mcp(pending_reservation, db_session):
    with patch(MCP_WRITE, return_value=True) as record:
        message = decide_reservation(pending_reservation.id, "confirm")

    db_session.refresh(pending_reservation)
    assert pending_reservation.status == ReservationStatus.CONFIRMED
    assert "confirmed" in message.lower()
    record.assert_called_once()


def test_refuse_sets_status_cancelled(pending_reservation, db_session):
    message = decide_reservation(pending_reservation.id, "refuse")

    db_session.refresh(pending_reservation)
    assert pending_reservation.status == ReservationStatus.CANCELLED
    assert "cancelled" in message.lower()


def test_failed_mcp_write_warns_the_admin(pending_reservation):
    # A confirmed booking that never reached the file must not look like a success.
    with patch(MCP_WRITE, return_value=False):
        message = decide_reservation(pending_reservation.id, "confirm")
    assert "WARNING" in message


def test_invalid_decision_is_rejected(pending_reservation):
    assert "invalid" in decide_reservation(pending_reservation.id, "maybe").lower()


def test_unknown_reservation_does_not_raise():
    assert "not found" in decide_reservation(999999, "confirm").lower()


def test_a_decided_reservation_is_not_decided_twice(pending_reservation):
    with patch(MCP_WRITE, return_value=True):
        decide_reservation(pending_reservation.id, "confirm")
    assert "already" in decide_reservation(pending_reservation.id, "refuse").lower()


def test_pending_list_includes_the_reservation(pending_reservation):
    assert pending_reservation.id in [row["id"] for row in list_pending_reservations()]


@pytest.mark.asyncio
async def test_approval_agent_approves_and_records_its_reason(pending_reservation, db_session):
    llm = FakeListChatModel(responses=["APPROVE\nValid plate and a sensible two-hour slot."])
    with patch(MCP_WRITE, return_value=True):
        decision, reason = await auto_review_reservation(pending_reservation.id, llm)

    db_session.refresh(pending_reservation)
    assert decision == "confirm"
    assert "sensible" in reason.lower()
    assert pending_reservation.status == ReservationStatus.CONFIRMED
    assert pending_reservation.decision_reason


@pytest.mark.asyncio
async def test_approval_agent_can_refuse(pending_reservation, db_session):
    llm = FakeListChatModel(responses=["REFUSE\nThe requested period is implausible."])
    decision, _ = await auto_review_reservation(pending_reservation.id, llm)

    db_session.refresh(pending_reservation)
    assert decision == "refuse"
    assert pending_reservation.status == ReservationStatus.CANCELLED


@pytest.mark.asyncio
async def test_unparseable_review_leaves_it_for_a_human(pending_reservation, db_session):
    llm = FakeListChatModel(responses=["I am really not sure about this one."])
    decision, _ = await auto_review_reservation(pending_reservation.id, llm)

    db_session.refresh(pending_reservation)
    assert decision == "pending"
    assert pending_reservation.status == ReservationStatus.PENDING_APPROVAL
