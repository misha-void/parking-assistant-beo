"""Intent classification for parking assistant queries."""

from typing import Literal, get_args

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from parking_assistant.rag.history import format_history

IntentType = Literal["general_info", "pricing_hours", "availability_search", "book_spot", "mixed"]

INTENT_CLASSIFICATION_PROMPT = """\
You are an intent classifier for a parking assistant in Belgrade, Serbia.

Classify the user's LATEST message into exactly ONE intent:

- general_info: zones, rules, parking types, how the system works, FAQs.
  "What are zone A rules?", "How do garages work?", "Can tourists park here?"
- pricing_hours: generic price rules per zone, operating hours, payment methods, free-parking times.
  "How much does the red zone cost?", "When is parking free?", "What are the working hours?"
- availability_search: listing or finding specific locations, free spots, and ANY ranking or
  comparison of locations or their prices (cheapest, most expensive, compare prices).
  "Show me available garages", "Which is the cheapest parking?", "Where can I park now?"
- book_spot: the user explicitly wants to reserve, or accepts the assistant's offer to book.
  "I want to book a spot", "Reserve parking for tomorrow", "Yes, book it" (after such an offer)
- mixed: clearly combines two or more of the above, or is too unclear to place.
  "What's available in the green zone and what are its rules?"

Rules:
- Use the conversation to resolve follow-ups ("the first one", "that garage", "yes").
- Asking about availability or prices is NOT book_spot; only an explicit wish to reserve is.
{history_section}
Latest user message: {query}

Answer with ONLY the intent name."""


def _parse_intent(reply: str) -> IntentType:
    """Map the LLM reply to a valid intent: exact match first, then substring, else general_info."""
    valid_intents = get_args(IntentType)
    cleaned = reply.strip().strip("\"'`.*").lower()
    if cleaned in valid_intents:
        return cleaned
    for intent in valid_intents:
        if intent in cleaned:
            return intent
    return "general_info"


async def classify_intent(query: str, llm, history: list = None) -> IntentType:
    """
    Classify user query intent using the LLM (history-aware for follow-ups).

    Args:
        query: User's question
        llm: Language model instance
        history: Prior messages [{"role", "content"}], oldest first, excluding query

    Returns:
        One of the IntentType strings
    """
    prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    history_text = format_history(history)
    history_section = f"\nConversation so far:\n{history_text}\n" if history_text else ""
    reply = await chain.ainvoke({"query": query, "history_section": history_section})
    return _parse_intent(reply)
