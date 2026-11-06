"""Parking Assistant - AI chatbot for parking information and reservations.

This package provides:
- RAG-based information retrieval over static docs (Milvus) and live data (SQLite)
- An interrupt-driven reservation flow orchestrated with LangGraph
- PII guardrails on retrieved context and generated answers
- An admin approval agent, with a Streamlit app for the human override
"""

__version__ = "0.1.0"
