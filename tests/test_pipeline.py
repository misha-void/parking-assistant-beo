"""Stage 4 - end-to-end integration test for the LangGraph orchestration.

Drives a full reservation through the graph (RAG route -> slot collection -> confirm ->
save -> escalate to admin), then has the admin approve it out of band (decide_reservation
-> MCP record). LLM and MCP calls are mocked and a numbered location pick avoids LLM
matching. Uses the seeded SQLite DB.
"""

import uuid
from datetime import datetime
from unittest.mock import patch, AsyncMock

import pytest

from parking_assistant.db.models import ParkingLocation, Reservation, ReservationStatus


class _FakeRAG:
    """Stand-in for RAGChain that always routes to the reservation flow."""
    def __init__(self, llm):
        pass

    async def run(self, query, history=None, remembered_locations=None):
        return {"intent": "book_spot"}

    def close(self):
        pass


P = "parking_assistant.graph.pipeline"


async def _fake_interpret(slot, answer, history, llm, locations=None, start_iso=None, today=None):
    """Treat every reply as an answer; return ISO for dates, "" for location (numeric fallback)."""
    if slot == "start_date":
        return {"kind": "answer", "value": "2024-01-01T10:00:00"}
    if slot == "end_date":
        return {"kind": "answer", "value": "2024-01-01T18:00:00"}
    if slot == "location":
        return {"kind": "answer", "value": ""}
    return {"kind": "answer", "value": answer.strip()}


@pytest.mark.asyncio
async def test_full_reservation_flow_submits_then_admin_confirms(db_session):
    loc = db_session.query(ParkingLocation).first()
    assert loc is not None, "seed the DB first: python data_ingestions/seed_dynamic_db.py"
    sid = f"it-{uuid.uuid4()}"

    with patch(f"{P}.RAGChain", _FakeRAG), \
         patch(f"{P}.get_chat_model", lambda *a, **k: None), \
         patch(f"{P}.interpret_slot_reply", new=_fake_interpret), \
         patch(f"{P}.infer_location_from_conversation", new=AsyncMock(return_value=None)), \
         patch(f"{P}.escalate_to_admin", new=AsyncMock()) as escalate:
        from parking_assistant.graph.pipeline import run_turn

        assert "first name" in (await run_turn(sid, "I want to book")).lower()
        await run_turn(sid, "John")            # -> ask surname
        await run_turn(sid, "Doe")             # -> ask car number
        await run_turn(sid, "BG123AB")         # -> ask location (shows list)
        await run_turn(sid, "1")               # pick first listed location -> ask start date
        await run_turn(sid, "tomorrow 10am")   # -> ask end date
        summary = await run_turn(sid, "tomorrow 6pm")
        assert "BG123AB" in summary            # confirmation summary shown
        final = await run_turn(sid, "yes")     # confirm -> save -> escalate

    assert "submitted for approval" in final.lower()
    escalate.assert_awaited_once()             # admin was notified

    r = (db_session.query(Reservation)
         .filter_by(car_number="BG123AB", user_surname="Doe")
         .order_by(Reservation.id.desc()).first())
    assert r is not None and r.status == ReservationStatus.PENDING_APPROVAL

    # Admin approves out of band -> status CONFIRMED + MCP record
    with patch("parking_assistant.admin.admin_agent.send_confirmed_reservation") as mcp_record:
        from parking_assistant.admin.admin_agent import decide_reservation
        msg = decide_reservation(r.id, "confirm")
    assert "confirmed" in msg.lower()
    mcp_record.assert_called_once()

    db_session.refresh(r)
    assert r.status == ReservationStatus.CONFIRMED

    db_session.delete(r)
    db_session.commit()
