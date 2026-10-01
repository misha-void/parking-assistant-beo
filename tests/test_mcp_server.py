"""Essential tests for Stage 3 - MCP server (confirmed reservation storage)."""

import os
import pytest
from fastapi.testclient import TestClient

os.environ["MCP_API_KEY"] = "test-key"
os.environ["MCP_RESERVATIONS_FILE"] = "test_confirmed_reservations.txt"

from mcp_server.main import app  # noqa: E402 (env vars must be set first)

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_file():
    """Remove the test output file before/after each test."""
    path = os.environ["MCP_RESERVATIONS_FILE"]
    if os.path.exists(path):
        os.remove(path)
    yield
    if os.path.exists(path):
        os.remove(path)


def _payload():
    return {
        "name": "John Doe",
        "car_number": "BG123AB",
        "reservation_period": "2024-01-01 10:00 to 2024-01-01 18:00",
        "approval_time": "2024-01-01T09:00:00",
    }


def test_rejects_request_without_api_key():
    response = client.post("/reservations/confirmed", json=_payload())
    assert response.status_code == 401


def test_rejects_request_with_wrong_api_key():
    response = client.post(
        "/reservations/confirmed", json=_payload(), headers={"x-api-key": "wrong"}
    )
    assert response.status_code == 401


def test_writes_correctly_formatted_line_with_valid_key():
    response = client.post(
        "/reservations/confirmed", json=_payload(), headers={"x-api-key": "test-key"}
    )
    assert response.status_code == 200

    with open(os.environ["MCP_RESERVATIONS_FILE"], encoding="utf-8") as f:
        line = f.readline()

    assert line.strip() == "John Doe | BG123AB | 2024-01-01 10:00 to 2024-01-01 18:00 | 2024-01-01T09:00:00"


def test_appends_multiple_entries():
    headers = {"x-api-key": "test-key"}
    client.post("/reservations/confirmed", json=_payload(), headers=headers)
    client.post("/reservations/confirmed", json=_payload(), headers=headers)

    with open(os.environ["MCP_RESERVATIONS_FILE"], encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 2
