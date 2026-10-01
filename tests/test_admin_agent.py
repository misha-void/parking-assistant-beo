"""Essential tests for Stage 2 - admin approval agent."""

import pytest
from datetime import datetime

from parking_assistant.db.models import Reservation, ReservationStatus, ParkingLocation
from parking_assistant.admin.admin_agent import record_decision


@pytest.fixture
def pending_reservation(db_session):
    """Create a pending reservation with an approval token, clean up after test."""
    location = db_session.query(ParkingLocation).first()
    reservation = Reservation(
        user_name="John",
        user_surname="Doe",
        car_number="BG123AB",
        location_id=location.id,
        start_datetime=datetime(2024, 1, 1, 10, 0),
        end_datetime=datetime(2024, 1, 1, 18, 0),
        status=ReservationStatus.PENDING_APPROVAL,
        approval_token="test-token-123",
    )
    db_session.add(reservation)
    db_session.commit()

    yield reservation

    db_session.delete(reservation)
    db_session.commit()


def test_confirm_sets_status_and_clears_token(pending_reservation, db_session):
    """Admin confirming the reservation marks it CONFIRMED and invalidates the token."""
    record_decision(pending_reservation.approval_token, "confirm")

    db_session.refresh(pending_reservation)
    assert pending_reservation.status == ReservationStatus.CONFIRMED
    assert pending_reservation.approval_token is None


def test_refuse_sets_status_cancelled(pending_reservation, db_session):
    """Admin refusing the reservation marks it CANCELLED."""
    record_decision(pending_reservation.approval_token, "refuse")

    db_session.refresh(pending_reservation)
    assert pending_reservation.status == ReservationStatus.CANCELLED


def test_invalid_token_returns_error_without_crash():
    """Unknown token should return a safe message, not raise."""
    message = record_decision("non-existent-token", "confirm")
    assert "not found" in message.lower()


def test_invalid_decision_value_rejected(pending_reservation):
    """Decision values other than confirm/refuse are rejected."""
    message = record_decision(pending_reservation.approval_token, "maybe")
    assert "invalid" in message.lower()


def test_token_cannot_be_reused(pending_reservation, db_session):
    """Once used, the same token must not apply a second decision."""
    record_decision(pending_reservation.approval_token, "confirm")

    # Token was cleared, so re-using the original token string should fail now
    second_result = record_decision("test-token-123", "refuse")
    assert "not found" in second_result.lower()
