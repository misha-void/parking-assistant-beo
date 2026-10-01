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
    "Write a short, clear summary (3-4 lines, plain text) of this parking "
    "reservation request for an administrator to review:\n\n"
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


def record_decision(token: str, decision: str) -> str:
    """
    Apply the administrator's decision for a reservation identified by token.

    Args:
        token: approval_token from the email link
        decision: "confirm" or "refuse"

    Returns:
        Human-readable result message.
    """
    if decision not in ("confirm", "refuse"):
        return "Invalid decision."

    session = SessionLocal()
    try:
        reservation = session.query(Reservation).filter(Reservation.approval_token == token).first()
        if reservation is None:
            return "Reservation not found or token already used."

        reservation.status = (
            ReservationStatus.CONFIRMED if decision == "confirm" else ReservationStatus.CANCELLED
        )
        reservation.approval_token = None  # invalidate link after use
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
    finally:
        session.close()
