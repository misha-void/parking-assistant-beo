"""Maps a chat session to the reservation it created, so the UI can poll its status.

This lives in the chat app's process. The admin app runs separately, so the database is
the only channel between them; the chat polls by the id recorded here. Restarting the
app clears it, which is acceptable for a demo.
"""


_reservation_by_session: dict[str, int] = {}


def set_reservation(session_id: str, reservation_id: int) -> None:
    """Remember the reservation most recently created in this session."""
    _reservation_by_session[session_id] = reservation_id


def get_reservation(session_id: str) -> int | None:
    """Return this session's reservation id, or None if it hasn't booked."""
    return _reservation_by_session.get(session_id)
