"""Pytest configuration and fixtures."""

import pytest
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is in path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def vector_store():
    """Create a vector store instance for tests."""
    from parking_assistant.rag.vector_store import MilvusVectorStorage
    store = MilvusVectorStorage()
    yield store
    store.close()


@pytest.fixture(scope="session")
def db_session():
    """Create a database session for tests."""
    from parking_assistant.db.database import SessionLocal
    session = SessionLocal()
    yield session
    session.close()
