"""
Core chat service - shared business logic for both Streamlit and FastAPI interfaces.

This module will eventually contain the full RAG pipeline, guardrails, and LangGraph logic.
For now, it's a simple pass-through to the LLM to prove the wiring works.
"""

from typing import List, Dict
from parking_assistant.llm_factory import get_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# System prompt for the parking assistant
SYSTEM_PROMPT = """You are a helpful parking lot assistant for a parking facility in Serbia.
You help users with:
- General information about the parking lot
- Availability of parking spaces
- Pricing and working hours
- Reservation process

Be friendly, concise, and helpful. If you don't know something, admit it honestly."""


async def process_chat_message(
    user_message: str,
    chat_history: List[Dict[str, str]] = None
) -> str:
    """
    Process a user message and return the assistant's response.
    
    This is a temporary trivial implementation. Will be replaced with:
    - RAG retrieval from vector DB
    - Guardrails (PII filtering)
    - LangGraph state machine for reservation flow
    
    Args:
        user_message: The user's input message
        chat_history: Optional list of previous messages [{"role": "user"|"assistant", "content": "..."}]
    
    Returns:
        The assistant's response as a string
    """
    llm = get_chat_model(temperature=0.7)
    
    # Build message history for context
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    
    # Add the current user message
    messages.append(HumanMessage(content=user_message))
    
    # Invoke the LLM (using async ainvoke for FastAPI compatibility)
    response = await llm.ainvoke(messages)
    
    return response.content
