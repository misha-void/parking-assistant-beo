"""In-process map of chat session -> its reservation id, so the UI can poll real status.

Lives in the same process as the Streamlit chat app and the graph. The admin app (a
separate process) changes status in the shared DB; the chat polls the DB by this id.
"""

_session_reservation: dict[str, int] = {}


def set_reservation(session_id: str, reservation_id: int) -> None:
    _session_reservation[session_id] = reservation_id


def get_reservation(session_id: str):
    return _session_reservation.get(session_id)
