"""Pytest configuration: run the suite against a throwaway database."""

import os
import tempfile
from pathlib import Path

# Point the app at a scratch SQLite file BEFORE anything imports the database module,
# and before load_dotenv(), which would otherwise supply the developer's DATABASE_URL.
# Without this the tests create and delete rows in the database used for development.
TEST_DB_PATH = Path(tempfile.gettempdir()) / "parking_assistant_test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ.setdefault("MCP_API_KEY", "test-key")

import pytest  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # never overrides the variables set above


@pytest.fixture(scope="session", autouse=True)
def seeded_database():
    """Create and seed the scratch database once for the whole session."""
    TEST_DB_PATH.unlink(missing_ok=True)
    from data_ingestions.seed_dynamic_db import seed_database

    seed_database()
    yield
    TEST_DB_PATH.unlink(missing_ok=True)


@pytest.fixture
def db_session():
    """A session bound to the scratch database."""
    from parking_assistant.db.database import SessionLocal

    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture(scope="session")
def vector_store():
    """Milvus vector store. Needs an API key and ingested data."""
    from parking_assistant.rag.vector_store import MilvusVectorStorage

    store = MilvusVectorStorage()
    yield store
    store.close()


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless RUN_INTEGRATION_TESTS=1.

    They need a live OpenAI key and ingested Milvus data, so the default `pytest` run
    stays green on a fresh checkout.
    """
    if os.getenv("RUN_INTEGRATION_TESTS") == "1":
        return
    skip = pytest.mark.skip(
        reason="integration test; set RUN_INTEGRATION_TESTS=1 with a valid OPENAI_API_KEY"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
