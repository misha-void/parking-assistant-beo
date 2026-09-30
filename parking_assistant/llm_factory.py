"""Factory for creating LLM and embedding models."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_chat_model(
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    api_key: Optional[str] = None
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance.
    
    Args:
        model: OpenAI model name (default: gpt-4o-mini)
        temperature: Sampling temperature 0-1 (default: 0.7)
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
    
    Returns:
        ChatOpenAI instance configured with the specified parameters
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in .env file or pass as parameter."
            )
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key
    )


def get_embeddings(
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None
) -> OpenAIEmbeddings:
    """
    Create an OpenAIEmbeddings instance.
    
    Args:
        model: OpenAI embedding model name (default: text-embedding-3-small)
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
    
    Returns:
        OpenAIEmbeddings instance configured with the specified parameters
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found. Set it in .env file or pass as parameter."
            )
    
    return OpenAIEmbeddings(
        model=model,
        api_key=api_key
    )
