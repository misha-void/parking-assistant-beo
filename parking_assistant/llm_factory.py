"""Factory for the chat and embedding models used across the project."""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def _require_api_key(api_key: str | None) -> str:
    """Return the OpenAI key, falling back to the environment."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not found. Set it in .env or pass it explicitly.")
    return key


def get_chat_model(
    model: str = CHAT_MODEL,
    temperature: float = 0.7,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Conversational model; the warmer default suits free-text answers."""
    return ChatOpenAI(model=model, temperature=temperature, api_key=_require_api_key(api_key))


def get_extraction_model(
    model: str = CHAT_MODEL,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Deterministic model for classification, parsing and value extraction.

    Sampling noise makes intent routing and slot values irreproducible, so those steps
    use temperature 0 rather than the conversational default.
    """
    return get_chat_model(model=model, temperature=0.0, api_key=api_key)


def get_embeddings(
    model: str = EMBEDDING_MODEL,
    api_key: str | None = None,
) -> OpenAIEmbeddings:
    """Embedding model backing the Milvus vector store."""
    return OpenAIEmbeddings(model=model, api_key=_require_api_key(api_key))
