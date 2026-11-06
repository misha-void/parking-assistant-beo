"""Chat service: a thin, fail-safe wrapper over the LangGraph pipeline."""

import logging
from typing import Optional

from parking_assistant.chat_logger import log_turn
from parking_assistant.graph.pipeline import run_turn

logger = logging.getLogger(__name__)

FRIENDLY_ERROR = (
    "Sorry — something went wrong on my side. Please try that again, "
    "or type 'cancel' to start over."
)


async def process_chat_message(
    user_message: str,
    session_id: str = "default",
    chat_history: Optional[list[dict]] = None,
) -> str:
    """Handle one user message.

    Never raises: an unexpected failure is logged and turned into a readable reply, so
    the UI shows a message instead of a traceback.
    """
    try:
        response = await run_turn(session_id, user_message, chat_history)
    except Exception:
        logger.exception("Chat turn failed for session %s", session_id)
        response = FRIENDLY_ERROR

    log_turn(session_id, user_message, response)
    return response
