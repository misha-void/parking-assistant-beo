"""
Core chat service - thin wrapper over the LangGraph pipeline.
"""

from typing import List, Dict

from parking_assistant.graph.pipeline import run_turn
from parking_assistant.chat_logger import log_turn


async def process_chat_message(
    user_message: str,
    session_id: str = "default",
    chat_history: List[Dict[str, str]] = None
) -> str:
    """Process a user message; conversation state lives in the graph checkpointer."""
    response = await run_turn(session_id, user_message, chat_history)
    log_turn(session_id, user_message, response)
    return response
