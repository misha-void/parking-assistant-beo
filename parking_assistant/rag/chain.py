"""RAG chain orchestration - ties together intent classification, retrieval, and response generation."""

from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from parking_assistant.rag.intent_classifier import classify_intent
from parking_assistant.rag.retriever import Retriever
from parking_assistant.guardrails.pii_filter import get_guardrail


RAG_RESPONSE_PROMPT = """You are a helpful parking assistant for Belgrade, Serbia.

Use the following context to answer the user's question. Be concise, friendly, and accurate.

If the context doesn't contain enough information to fully answer the question, say so honestly and offer to help with related information.

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
    
    async def run(self, query: str) -> Dict[str, Any]:
        """
        Execute full RAG pipeline.
        
        Args:
            query: User's question
        
        Returns:
            Dict with 'intent', 'context', 'response', 'metadata'
        """
        # Step 1: Classify intent
        intent = await classify_intent(query, self.llm)
        
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

        chain = self.response_prompt | self.llm | StrOutputParser()
        
        response = await chain.ainvoke({
            "context": context,
            "query": query
        })
        
        return {
            "intent": intent,
            "context": context,
            "response": response,
            "metadata": {
                "source": retrieval_result.get("source", "unknown"),
                "num_results": retrieval_result.get("num_results", 0)
            }
        }
    
    def close(self):
        """Clean up resources."""
        self.retriever.close()
