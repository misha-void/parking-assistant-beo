"""
Second agent: Admin Approval Agent.

Responsible for escalating reservation requests to a human administrator
and applying the administrator's decision (confirm/refuse) back to the DB.

Uses LangChain for summary generation; the actual notification is a
pluggable tool (currently email, see notifier.py).
"""

import secrets
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from parking_assistant.db.database import SessionLocal
from parking_assistant.db.models import Reservation, ReservationStatus
from parking_assistant.admin.notifier import send_admin_notification
from mcp_server.client import send_confirmed_reservation


SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarize this parking reservation request for an administrator deciding whether to "
    "approve it. Use 2-3 short plain-text lines. State only the facts given below — do not "
    "invent details. Lead with the guest name and the time window.\n\n"
    "Name: {name} {surname}\n"
    "Car number: {car_number}\n"
    "Location: {location}\n"
    "From: {start} To: {end}\n"
)


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
    """
    Escalate a pending reservation to the administrator for approval.
    Generates an approval token and sends a notification with
    confirm/refuse links.
    """
    session = SessionLocal()
    try:
        reservation = session.query(Reservation).filter(Reservation.id == reservation_id).first()
        if reservation is None:
            return

        reservation.approval_token = secrets.token_urlsafe(32)
        session.commit()

        summary = await _generate_summary(reservation, llm)
        send_admin_notification(summary, reservation.approval_token)
    finally:
        session.close()


def _apply_decision(session, reservation: Reservation, decision: str) -> str:
    """Set status, invalidate the token, and (on confirm) hand off to the MCP server."""
    reservation.status = (
        ReservationStatus.CONFIRMED if decision == "confirm" else ReservationStatus.CANCELLED
    )
    reservation.approval_token = None  # invalidate any pending link
    session.commit()

    if reservation.status == ReservationStatus.CONFIRMED:
        # Stage 3: hand off to MCP server for durable storage
        period = f"{reservation.start_datetime} to {reservation.end_datetime}"
        send_confirmed_reservation(
            name=f"{reservation.user_name} {reservation.user_surname}",
            car_number=reservation.car_number,
            reservation_period=period,
            approval_time=datetime.utcnow().isoformat(),
        )

    return f"Reservation #{reservation.id} has been {reservation.status.value}."


def record_decision(token: str, decision: str) -> str:
    """Apply the admin's decision for a reservation identified by approval token (REST/email path)."""
    if decision not in ("confirm", "refuse"):
        return "Invalid decision."

    session = SessionLocal()
    try:
        reservation = session.query(Reservation).filter(Reservation.approval_token == token).first()
        if reservation is None:
            return "Reservation not found or token already used."
        return _apply_decision(session, reservation, decision)
    finally:
        session.close()


def decide_reservation(reservation_id: int, decision: str) -> str:
    """Apply the admin's decision by reservation id (used by the admin app)."""
    if decision not in ("confirm", "refuse"):
        return "Invalid decision."

    session = SessionLocal()
    try:
        reservation = session.query(Reservation).filter(Reservation.id == reservation_id).first()
        if reservation is None:
            return "Reservation not found."
        if reservation.status != ReservationStatus.PENDING_APPROVAL:
            return f"Reservation #{reservation.id} is already {reservation.status.value}."
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
            "start": str(r.start_datetime),
            "end": str(r.end_datetime),
        } for r in rows]
    finally:
        session.close()
