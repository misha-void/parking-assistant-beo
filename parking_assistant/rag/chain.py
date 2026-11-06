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
- Asked to list or find parking, show every location in the Context on its own line
  with its price (RSD/hour), and say how many there are.
- Asked for the "cheapest" or "most expensive", compare the prices and give only the
  winner with its price. Do not list the others and do not state a total count.
- Never say "several" or "a few" when the Context lets you be exact.
- Each location line states its zone. Never work out a location's zone from the zone
  price list, and never claim a zone that is not stated on that location's own line.
- Greeted, or asked what you can do: say briefly that you answer questions about
  parking in Belgrade and can book a spot. Ignore the Context for that.
- If the Context lacks the answer, say so briefly and offer related help.
- Use the Conversation to resolve follow-ups; do not re-ask for details already given.
- Be concise. Never mention the words "context" or "retrieved".
- Only if you listed bookable locations, end with one short offer to book
  (e.g. "Would you like to book a spot?"); otherwise do not offer to book and do not
  end with filler such as "let me know if you need anything else".

Conversation so far:
{history}

Context:
{context}

User question: {query}

Your response:"""


class RAGChain:
    """Orchestrates the full RAG pipeline."""

    def __init__(self, llm, classifier_llm=None, enable_guardrails: bool = True):
        self.llm = llm
        # Routing is a classification, so it belongs on a deterministic model. Callers
        # that pass only `llm` keep the previous single-model behaviour.
        self.classifier_llm = classifier_llm or llm
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
        self, query: str, history: list = None, remembered_locations: list = None,
        note: str = "",
    ) -> dict[str, Any]:
        """
        Execute full RAG pipeline.

        Args:
            query: User's question
            history: Prior messages [{"role", "content"}], oldest first, excluding query
            remembered_locations: parking options shown earlier, so follow-ups
                ("address?", "are those different?") can be answered from them
            note: an extra instruction for this turn, e.g. that a booking is under way

        Returns:
            Dict with 'intent', 'context', 'response', 'locations', 'metadata'
        """
        intent = await classify_intent(query, self.classifier_llm, history)

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

        # Add earlier-shown options so follow-up questions have the data. Skipped when
        # retrieval produced its own list, which would otherwise appear twice in the
        # context, possibly with different spot counts.
        if remembered_locations and not retrieval_result.get("locations"):
            remembered = "\n".join(format_location_line(loc) for loc in remembered_locations)
            context += "\n\n[Parking options shown earlier]\n" + remembered

        if note:
            context += "\n\n[Note for this turn]\n" + note

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
