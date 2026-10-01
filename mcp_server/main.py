"""MCP server (FastAPI) - Stage 3.

Processes confirmed reservations and persists them to a text file.
Exposes a single endpoint, protected by an API key header, to prevent
unauthorized writes. Run separately from the main app:

    uvicorn mcp_server.main:app --port 8001
"""

import os
import secrets
import threading
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

MCP_API_KEY = os.getenv("MCP_API_KEY", "")
RESERVATIONS_FILE = os.getenv("MCP_RESERVATIONS_FILE", "confirmed_reservations.txt")

app = FastAPI(title="Reservation MCP Server")
_write_lock = threading.Lock()  # serialize file writes for reliability under concurrency


class ReservationRecord(BaseModel):
    """Payload for a confirmed reservation to persist."""
    name: str
    car_number: str
    reservation_period: str
    approval_time: str


def _verify_api_key(x_api_key: Optional[str]) -> None:
    """Reject requests without a valid, constant-time-compared API key."""
    if not MCP_API_KEY or not x_api_key or not secrets.compare_digest(x_api_key, MCP_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/reservations/confirmed")
def write_confirmed_reservation(
    record: ReservationRecord,
    x_api_key: Optional[str] = Header(default=None),
):
    """Append a confirmed reservation to storage. Format: Name | Car | Period | Approval Time."""
    _verify_api_key(x_api_key)

    line = f"{record.name} | {record.car_number} | {record.reservation_period} | {record.approval_time}\n"
    with _write_lock:
        with open(RESERVATIONS_FILE, "a", encoding="utf-8") as f:
            f.write(line)

    return {"status": "ok"}
