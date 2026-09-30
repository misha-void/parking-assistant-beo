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


INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a parking assistant chatbot in Belgrade, Serbia.

Classify the user's query into ONE of these intents:

1. **general_info**: Questions about parking zones, rules, parking types (garage/car park), how the system works, FAQs
   Examples: "What are zone A rules?", "How do garages work?", "Can tourists park here?"

2. **pricing_hours**: Questions about costs, prices, operating hours, payment methods, when parking is free
   Examples: "How much does red zone cost?", "When is parking free?", "What are the working hours?"

3. **availability_search**: Finding available parking spots, searching for locations, checking capacity
   Examples: "Show me available garages", "Any parking near Clinical Centre?", "Where can I park now?"

4. **book_spot**: User wants to make a reservation or book a parking spot
   Examples: "I want to book a spot", "Reserve parking for tomorrow", "Can I make a reservation?"

5. **mixed**: Query combines multiple intents OR is unclear and needs both knowledge base and database info
   Examples: "What's available in green zone and what are the rules?", "Show me cheap parking with spots available"

User query: {query}

Respond with ONLY the intent name: general_info, pricing_hours, availability_search, book_spot, or mixed"""


async def classify_intent(query: str, llm) -> IntentType:
    """
    Classify user query intent using LLM.
    
    Args:
        query: User's question
        llm: Language model instance
    
    Returns:
        Intent type as string
    """
    prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    result = await chain.ainvoke({"query": query})
    
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
