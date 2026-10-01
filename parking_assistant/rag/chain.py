"""RAG chain orchestration - ties together intent classification, retrieval, and response generation."""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from parking_assistant.rag.intent_classifier import classify_intent, _format_history  # helper defined once there (avoids circular import)
from parking_assistant.rag.retriever import Retriever
from parking_assistant.guardrails.pii_filter import get_guardrail


def _format_locations(locations: list) -> str:
    """Render remembered parking options (name, address, price, availability) as plain text."""
    lines = []
    for loc in locations:
        price = loc.get("price")
        price_str = f" — {price} RSD/hour" if price else " — price varies (private garage)"
        avail = loc.get("available")
        avail_str = f" — {avail} spots" if avail is not None else ""
        lines.append(f"- {loc.get('name')} — {loc.get('address', 'address n/a')}{price_str}{avail_str}")
    return "\n".join(lines)


RAG_RESPONSE_PROMPT = """You are a helpful parking assistant for Belgrade, Serbia.

Answer the user's latest question using ONLY the Context below and the Conversation so far.

Rules:
- Never invent facts, counts, prices, hours or locations that are not in the Context.
- If the Context lists options/locations, list ALL of them; do not drop any. Always include each location's price (RSD/hour) when shown; if it says "price varies", say the price varies for that garage.
- For "cheapest"/"most expensive" questions, rank by the listed RSD/hour price and name the specific location (ignore "price varies" garages when ranking). Never claim a price that isn't in the Context.
- Do NOT say "several", "multiple", "various" or "a few" unless TWO OR MORE distinct items are actually in the Context. If exactly one item is present, say "one option" and name it. If none, say you don't have that info and offer related help.
- Use the conversation to resolve "it", "that one", follow-ups; stay consistent with earlier answers and do not re-ask for details already given.
- Be concise and friendly. Never mention the words "context" or "retrieved" to the user.
- When you have shown or discussed specific parking locations the user could use, END with a short, direct offer to book — e.g. "Would you like to book a parking spot?" (or "...book Pasterova?" when one location is in focus). Do NOT end with passive lines like "let me know if you need details".

Conversation so far:
{history}

Context:
{context}

User question: {query}

Your response:"""


class RAGChain:
    """Orchestrates the full RAG pipeline."""
    
    def __init__(self, llm, enable_guardrails: bool = True):
        self.llm = llm
        self.retriever = Retriever()
        self.response_prompt = ChatPromptTemplate.from_template(RAG_RESPONSE_PROMPT)
        self.enable_guardrails = enable_guardrails

        if enable_guardrails:
            self.guardrail = get_guardrail()
    
    async def run(self, query: str, history: list = None, remembered_locations: list = None) -> Dict[str, Any]:
        """
        Execute full RAG pipeline.

        Args:
            query: User's question
            history: Prior messages [{"role", "content"}], oldest first, excluding query
            remembered_locations: parking options shown earlier, so follow-ups
                ("address?", "are those different?") can be answered from them

        Returns:
            Dict with 'intent', 'context', 'response', 'metadata', 'locations'
        """
        # Step 1: Classify intent (history-aware)
        intent = await classify_intent(query, self.llm, history)
        
        # Step 2: Retrieve based on intent
        if intent == "general_info":
            retrieval_result = await self.retriever.retrieve_general_info(query)
        
        elif intent == "pricing_hours":
            retrieval_result = await self.retriever.retrieve_pricing_hours(query)
        
        elif intent == "availability_search":
            # TODO: Extract filters from query (parking type, zone, etc.)
            # For now, retrieve all available
            retrieval_result = await self.retriever.retrieve_availability()
        
        elif intent == "book_spot":
            # Signal that reservation flow should start
            return {
                "intent": "book_spot",
                "context": None,
                "response": None,  # Service layer will handle this
                "metadata": {"action": "start_reservation"}
            }
        
        elif intent == "mixed":
            retrieval_result = await self.retriever.retrieve_hybrid(query)
        
        else:
            # Fallback
            retrieval_result = await self.retriever.retrieve_general_info(query)
        
        # Step 3: Generate response using retrieved context
        context = retrieval_result.get("context", "")

        # Step 4: Apply guardrails to filter PII from retrieved context
        if self.enable_guardrails and context:
            context = self.guardrail.filter_retrieved_context(context)

        # Step 5: Add earlier-shown options so follow-up questions have the data
        if remembered_locations:
            context = (context + "\n\n[Parking options shown earlier]\n" +
                       _format_locations(remembered_locations))

        chain = self.response_prompt | self.llm | StrOutputParser()

        response = await chain.ainvoke({
            "history": _format_history(history) or "(none)",
            "context": context or "(no information found)",
            "query": query
        })

        return {
            "intent": intent,
            "context": context,
            "response": response,
            "locations": retrieval_result.get("locations"),  # set for availability searches
            "metadata": {
                "source": retrieval_result.get("source", "unknown"),
                "num_results": retrieval_result.get("num_results", 0)
            }
        }
    
    def close(self):
        """Clean up resources."""
        self.retriever.close()
