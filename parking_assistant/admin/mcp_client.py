"""The admin agent's side of the MCP call: posts a confirmed reservation to mcp_server.py."""

import os

import httpx

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8001")
MCP_API_KEY = os.getenv("MCP_API_KEY", "")


def send_confirmed_reservation(
    name: str, car_number: str, reservation_period: str, approval_time: str
) -> bool:
    """
    Notify the MCP server that a reservation was confirmed, so it can
    persist it to file storage. Never raises: a failure is logged but must
    not block admin approval. Returns True on success, False on a rejected
    write (e.g. 401 from a missing API key) or a transport error.
    """
    payload = {
        "name": name,
        "car_number": car_number,
        "reservation_period": reservation_period,
        "approval_time": approval_time,
    }
    headers = {"x-api-key": MCP_API_KEY}

    try:
        response = httpx.post(
            f"{MCP_SERVER_URL}/reservations/confirmed", json=payload, headers=headers, timeout=5
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print("[MCP client] MCP server rejected the write: "
              f"{e.response.status_code} {e.response.text}")
        return False
    except httpx.RequestError as e:
        print(f"[MCP client] Failed to reach MCP server: {e}")
        return False
    return True
