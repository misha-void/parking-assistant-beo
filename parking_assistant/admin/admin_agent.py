"""
Second agent: Admin Approval Agent.

Reviews pending reservation requests and applies a decision (confirm/refuse)
back to the DB. By default an LLM plays the administrator (auto_review_reservation);
a human can still decide in the Streamlit admin app (decide_reservation).

Uses LangChain for summary and review generation; the notification is a
pluggable tool (currently email, see notifier.py).
"""

import logging
import os
from datetime import UTC, datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from parking_assistant.admin.mcp_client import send_confirmed_reservation
from parking_assistant.admin.notifier import send_admin_notification
from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import PriceRule, Reservation, ReservationStatus

logger = logging.getLogger(__name__)

def auto_approve_enabled() -> bool:
    """True when the approval agent should decide instead of a human.

    Read at call time on purpose: a module-level constant would capture the value
    before .env is loaded, which silently ignored AUTO_APPROVE=false.
    """
    return os.getenv("AUTO_APPROVE", "true").strip().lower() == "true"

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarize this parking reservation request for an administrator deciding whether to "
    "approve it. Use 2-3 short plain-text lines. State only the facts given below — do not "
    "invent details. Lead with the guest name and the time window.\n\n"
    "Name: {name} {surname}\n"
    "Car number: {car_number}\n"
    "Location: {location}\n"
    "From: {start} To: {end}\n"
)

REVIEW_PROMPT = ChatPromptTemplate.from_template(
    "You are a parking administrator reviewing a reservation request. Approve it unless "
    "something is clearly wrong: the period is in the past, the end is not after the start, "
    "the booking is implausibly long (more than 24 hours), or the name or car plate is "
    "obviously fake.\n\n"
    "Current time: {now}\n"
    "Guest name: {name} {surname}\n"
    "Car plate: {car_number}\n"
    "Location: {location}\n"
    "Zone / price: {price}\n"
    "Start: {start}\n"
    "End: {end}\n"
    "Duration: {hours:.1f} hours\n\n"
    "Answer in exactly two lines. Line 1: APPROVE or REFUSE. "
    "Line 2: the reason, at most 15 words."
)


def _price_text(session, reservation: Reservation) -> str:
    """Describe the location's zone and hourly rate, e.g. 'green_3, 80 RSD/h'."""
    zone = reservation.location.zone if reservation.location else None
    if zone is None:
        return "no street zone / not specified"
    rule = session.query(PriceRule).filter(PriceRule.zone == zone).first()
    return f"{zone.value}, {rule.hourly_rate_rsd} RSD/h" if rule else zone.value


async def _generate_summary(reservation: Reservation, llm) -> str:
    """Use the LLM to render a human-friendly request summary."""
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return await chain.ainvoke({
        "name": reservation.user_name,
        "surname": reservation.user_surname,
        "car_number": reservation.car_number,
        "location": reservation.location.name,
        "start": reservation.start_datetime,
        "end": reservation.end_datetime,
    })


async def escalate_to_admin(reservation_id: int, llm) -> None:
    """Send the administrator a summary of a pending reservation (email or console)."""
    session = SessionLocal()
    try:
        reservation = session.query(Reservation).filter(Reservation.id == reservation_id).first()
        if reservation is None:
            return

        summary = await _generate_summary(reservation, llm)
        send_admin_notification(summary, reservation.id)
    finally:
        session.close()


def _apply_decision(session, reservation: Reservation, decision: str) -> str:
    """Set status and (on confirm) hand off to the MCP server; return a status message."""
    reservation.status = (
        ReservationStatus.CONFIRMED if decision == "confirm" else ReservationStatus.CANCELLED
    )
    session.commit()

    message = f"Reservation #{reservation.id} has been {reservation.status.value}."
    if reservation.status == ReservationStatus.CONFIRMED:
        # Stage 3: hand off to MCP server for durable storage
        recorded = send_confirmed_reservation(
            name=f"{reservation.user_name} {reservation.user_surname}",
            car_number=reservation.car_number,
            reservation_period=f"{reservation.start_datetime} to {reservation.end_datetime}",
            approval_time=datetime.now(UTC).isoformat(),
        )
        if not recorded:
            message += " WARNING: could not record it to the MCP server (see server log)."
    return message


def _parse_review(text: str):
    """Parse 'APPROVE|REFUSE' + reason; return (decision, reason) or None if malformed."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return None
    verdict = lines[0].strip(" .:*`").upper()
    if verdict not in ("APPROVE", "REFUSE"):
        return None
    reason = lines[1][:300] if len(lines) > 1 else "No reason given."
    return ("confirm" if verdict == "APPROVE" else "refuse"), reason


async def auto_review_reservation(reservation_id: int, llm) -> tuple[str, str]:
    """
    Let the LLM act as the administrator: decide a pending reservation.

    The caller must pass a deterministic model (temperature 0). Returns
    (decision, reason) with decision "confirm", "refuse" or "pending"
    (left pending if the review could not be obtained or parsed). Never raises.
    """
    session = SessionLocal()
    try:
        reservation = session.query(Reservation).filter(Reservation.id == reservation_id).first()
        if reservation is None or reservation.status != ReservationStatus.PENDING_APPROVAL:
            return "pending", "Reservation not found or not pending."

        duration = reservation.end_datetime - reservation.start_datetime
        chain = REVIEW_PROMPT | llm | StrOutputParser()
        try:
            text = await chain.ainvoke({
                "now": datetime.now().isoformat(timespec="minutes"),
                "name": reservation.user_name,
                "surname": reservation.user_surname,
                "car_number": reservation.car_number,
                "location": reservation.location.name,
                "price": _price_text(session, reservation),
                "start": reservation.start_datetime,
                "end": reservation.end_datetime,
                "hours": duration.total_seconds() / 3600,
            })
            parsed = _parse_review(text)
        except Exception:
            parsed = None
        if parsed is None:
            return "pending", "Could not parse review."

        decision, reason = parsed
        reservation.decision_reason = reason
        outcome = _apply_decision(session, reservation, decision)

        # A confirmed booking that never reached the MCP server must not look like a
        # clean success, so the problem is carried back in the reason the user sees.
        if "WARNING" in outcome:
            logger.warning("Reservation %s confirmed but not recorded: %s",
                           reservation_id, outcome)
            reason = f"{reason} (note: it could not be recorded to the MCP server)"
        return decision, reason
    except Exception:
        session.rollback()
        return "pending", "Could not parse review."
    finally:
        session.close()


def decide_reservation(reservation_id: int, decision: str, reason: str = None) -> str:
    """Apply the admin's decision by reservation id, optionally storing a reason."""
    if decision not in ("confirm", "refuse"):
        return "Invalid decision."

    session = SessionLocal()
    try:
        reservation = session.query(Reservation).filter(Reservation.id == reservation_id).first()
        if reservation is None:
            return "Reservation not found."
        if reservation.status != ReservationStatus.PENDING_APPROVAL:
            return f"Reservation #{reservation.id} is already {reservation.status.value}."
        if reason:
            reservation.decision_reason = reason[:300]
        return _apply_decision(session, reservation, decision)
    finally:
        session.close()


def get_reservation_status(reservation_id: int):
    """Return the current status value (e.g. 'confirmed') for a reservation, or None."""
    session = SessionLocal()
    try:
        r = session.query(Reservation).filter(Reservation.id == reservation_id).first()
        return r.status.value if r else None
    finally:
        session.close()


def list_pending_reservations() -> list:
    """Return pending reservations as plain dicts (slot overview) for the admin app."""
    session = SessionLocal()
    try:
        rows = (session.query(Reservation)
                .filter(Reservation.status == ReservationStatus.PENDING_APPROVAL)
                .order_by(Reservation.id).all())
        return [{
            "id": r.id,
            "name": f"{r.user_name} {r.user_surname}",
            "car_number": r.car_number,
            "location": r.location.name if r.location else "-",
            "price": _price_text(session, r),
            "start": str(r.start_datetime),
            "end": str(r.end_datetime),
        } for r in rows]
    finally:
        session.close()
