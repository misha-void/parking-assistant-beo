"""RAG chain orchestration: intent classification, retrieval and response generation."""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from parking_assistant.guardrails.pii_filter import get_guardrail
from parking_assistant.rag.history import format_history
from parking_assistant.rag.intent_classifier import classify_intent
from parking_assistant.rag.retriever import Retriever, format_location_line

RAG_RESPONSE_PROMPT = """You are a friendly parking assistant for Belgrade, Serbia.

Answer the user's latest question using ONLY the Context and the Conversation below.

Rules:
- Never invent prices, hours, counts or locations that are not in the Context.
- If the Context lists locations, show every one on its own line with its price (RSD/hour).
- For "cheapest" or "most expensive", compare the listed prices and name the winner.
- State the exact number of options; do not say "several" or "a few".
- If the Context lacks the answer, say so briefly and offer related help.
- Use the Conversation to resolve follow-ups; do not re-ask for details already given.
- Be concise. Never mention the words "context" or "retrieved".
- Only if you listed bookable locations, end with one short offer to book
  (e.g. "Would you like to book a spot?"); otherwise do not offer to book.

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

    async def _retrieve(self, intent: str, query: str) -> dict[str, Any]:
        """Run the retrieval strategy that matches the intent."""
        if intent == "pricing_hours":
            return await self.retriever.retrieve_pricing_hours(query)
        if intent == "availability_search":
            return await self.retriever.retrieve_availability()
        if intent == "mixed":
            return await self.retriever.retrieve_hybrid(query)
        return await self.retriever.retrieve_general_info(query)

    async def run(
        self, query: str, history: list = None, remembered_locations: list = None
    ) -> dict[str, Any]:
        """
        Execute full RAG pipeline.

        Args:
            query: User's question
            history: Prior messages [{"role", "content"}], oldest first, excluding query
            remembered_locations: parking options shown earlier, so follow-ups
                ("address?", "are those different?") can be answered from them

        Returns:
            Dict with 'intent', 'context', 'response', 'locations', 'metadata'
        """
        intent = await classify_intent(query, self.llm, history)

        if intent == "book_spot":
            # Signal that the reservation flow should start; the graph handles the reply
            return {
                "intent": "book_spot",
                "context": None,
                "response": None,
                "metadata": {"action": "start_reservation"},
            }

        retrieval_result = await self._retrieve(intent, query)
        context = retrieval_result.get("context", "")

        if self.enable_guardrails and context:
            context = self.guardrail.filter_retrieved_context(context)

        # Add earlier-shown options so follow-up questions have the data
        if remembered_locations:
            remembered = "\n".join(format_location_line(loc) for loc in remembered_locations)
            context += "\n\n[Parking options shown earlier]\n" + remembered

        chain = self.response_prompt | self.llm | StrOutputParser()
        response = await chain.ainvoke({
            "history": format_history(history) or "(none)",
            "context": context or "(no information found)",
            "query": query,
        })

        if self.enable_guardrails:
            response = self.guardrail.filter_llm_output(response)

        return {
            "intent": intent,
            "context": context,
            "response": response,
            "locations": retrieval_result.get("locations"),  # set for availability searches
            "metadata": {
                "source": retrieval_result.get("source", "unknown"),
                "num_results": retrieval_result.get("num_results", 0),
            },
        }

    def close(self):
        """Clean up resources."""
        self.retriever.close()
