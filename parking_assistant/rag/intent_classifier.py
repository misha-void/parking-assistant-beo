"""Intent classification for parking assistant queries."""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

IntentType = Literal["general_info", "pricing_hours", "availability_search", "book_spot", "mixed"]


class QueryIntent(BaseModel):
    """Structured intent classification result."""
    intent: IntentType = Field(description="The classified intent type")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    reasoning: str = Field(description="Brief explanation of why this intent was chosen")


def _format_history(history: list = None, limit: int = 6) -> str:
    """Format the last `limit` chat messages as 'User: ...' / 'Assistant: ...' lines."""
    return "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
        for m in (history or [])[-limit:]
    )


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a parking assistant chatbot in Belgrade, Serbia.

Classify the user's LATEST message into exactly ONE of these intents:

1. **general_info**: Parking zones, rules, parking types (garage/car park), how the system works, FAQs
   Examples: "What are zone A rules?", "How do garages work?", "Can tourists park here?"

2. **pricing_hours**: Costs, prices, operating hours, payment methods, when parking is free
   Examples: "How much does red zone cost?", "When is parking free?", "What are the working hours?"

3. **availability_search**: Finding parking, listing locations, checking free spots/capacity
   Examples: "Show me available garages", "Any parking near Clinical Centre?", "Where can I park now?"

4. **book_spot**: User wants to reserve/book a spot, or agrees to an offer to book
   Examples: "I want to book a spot", "Reserve parking for tomorrow", "Yes, book it" (after the assistant offered to book)

5. **mixed**: Clearly combines two or more of the above, or is too unclear to place
   Examples: "What's available in green zone and what are the rules?", "Show me cheap parking with spots available"

Rules:
- Use the conversation so far to resolve follow-ups ("the first one", "that garage", "yes").
  e.g. after the assistant lists garages, "how much is the first one?" -> pricing_hours.
- A bare "yes"/"ok" right after the assistant offered to book -> book_spot.
- Merely asking about availability or prices is NOT book_spot; only explicit intent to reserve is.
{history_section}
Latest user message: {query}

Respond with ONLY the intent name: general_info, pricing_hours, availability_search, book_spot, or mixed"""


async def classify_intent(query: str, llm, history: list = None) -> IntentType:
    """
    Classify user query intent using LLM (history-aware for follow-ups).

    Args:
        query: User's question
        llm: Language model instance
        history: Prior messages [{"role", "content"}], oldest first, excluding query

    Returns:
        Intent type as string
    """
    prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    hist = _format_history(history)
    history_section = f"\nConversation so far:\n{hist}\n" if hist else ""
    result = await chain.ainvoke({"query": query, "history_section": history_section})
    
    # Parse and validate result
    intent = result.strip().lower()
    
    # Map to valid intent (defensive)
    valid_intents = ["general_info", "pricing_hours", "availability_search", "book_spot", "mixed"]
    
    if intent in valid_intents:
        return intent
    
    # Fallback: check if intent is mentioned in response
    for valid in valid_intents:
        if valid in intent:
            return valid
    
    # Default to mixed if unclear
    return "mixed"
