"""Client for calling the Reservation MCP server (used by the admin agent)."""

import os
import httpx

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
MCP_API_KEY = os.getenv("MCP_API_KEY", "")


def send_confirmed_reservation(name: str, car_number: str, reservation_period: str, approval_time: str) -> None:
    """
    Notify the MCP server that a reservation was confirmed, so it can
    persist it to file storage. Fails silently with a log if the MCP
    server is unreachable, so admin approval itself is never blocked.
    """
    payload = {
        "name": name,
        "car_number": car_number,
        "reservation_period": reservation_period,
        "approval_time": approval_time,
    }
    headers = {"x-api-key": MCP_API_KEY}

    try:
        httpx.post(f"{MCP_SERVER_URL}/reservations/confirmed", json=payload, headers=headers, timeout=5)
    except httpx.RequestError as e:
        print(f"[MCP client] Failed to reach MCP server: {e}")
