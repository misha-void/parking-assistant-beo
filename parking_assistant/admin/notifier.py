"""Email notification channel for the admin approval agent.

Sends a reservation request to the administrator with clickable
confirm/refuse links. Falls back to printing to console if SMTP
is not configured (useful for local dev/testing).
"""

import os
import smtplib
from email.mime.text import MIMEText

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)


def _build_email_body(summary: str, token: str) -> str:
    """Build plain-text email body with confirm/refuse links."""
    confirm_url = f"{APP_BASE_URL}/admin/respond?token={token}&decision=confirm"
    refuse_url = f"{APP_BASE_URL}/admin/respond?token={token}&decision=refuse"
    return (
        f"New parking reservation request:\n\n{summary}\n\n"
        f"Confirm: {confirm_url}\n"
        f"Refuse:  {refuse_url}\n"
    )


def send_admin_notification(summary: str, token: str) -> None:
    """
    Send reservation details to the administrator via email.
    If SMTP is not configured, print to console instead (dev fallback).
    """
    body = _build_email_body(summary, token)

    if not (SMTP_HOST and ADMIN_EMAIL and SMTP_USER and SMTP_PASSWORD):
        print("\n[ADMIN NOTIFICATION - console fallback, SMTP not configured]")
        print(body)
        return

    msg = MIMEText(body)
    msg["Subject"] = "Parking Reservation Approval Needed"
    msg["From"] = SMTP_FROM
    msg["To"] = ADMIN_EMAIL

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(SMTP_FROM, [ADMIN_EMAIL], msg.as_string())
