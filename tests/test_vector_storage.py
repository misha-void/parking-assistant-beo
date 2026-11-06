"""Tests for the Milvus vector store.

These are integration tests: they need a valid OPENAI_API_KEY (for the embeddings) and
the static documents already ingested. Run them with RUN_INTEGRATION_TESTS=1.
"""

import pytest

from parking_assistant.rag.vector_store import MilvusVectorStorage

pytestmark = pytest.mark.integration

RESULT_KEYS = {"text", "source_file", "chunk_index", "score"}


def test_vector_store_initializes(vector_store):
    assert vector_store.embedding_dim == 1536


@pytest.mark.parametrize(
    "query,collection,min_score",
    [
        ("What are the parking rules in zone A?", "parking_zone_rules", 0.5),
        ("How much does parking cost?", "parking_pricing", 0.4),
        ("What is a garage?", "parking_locations_info", 0.4),
        ("Can tourists park here?", "parking_faq", 0.4),
    ],
)
def test_search_returns_relevant_results(vector_store, query, collection, min_score):
    results = vector_store.search(collection_name=collection, query=query, top_k=2)

    assert results, f"no results for {query!r}"
    assert results[0]["score"] >= min_score, f"low relevance: {results[0]['score']}"
    assert RESULT_KEYS <= set(results[0])


def test_every_expected_collection_is_searchable(vector_store):
    for collection in MilvusVectorStorage.ALL_COLLECTIONS:
        results = vector_store.search(collection, "parking", top_k=1)
        assert results, f"{collection} returned no results"
        assert RESULT_KEYS <= set(results[0])
