"""Tests for Milvus vector storage."""

import pytest


def test_vector_storage_initialization(vector_store):
    """Test that vector storage initializes correctly."""
    assert vector_store is not None
    assert vector_store.embedding_dim == 1536


@pytest.mark.parametrize("query,collection,min_score", [
    ("What are the parking rules in zone A?", "parking_zone_rules", 0.5),
    ("How much does parking cost?", "parking_pricing", 0.4),
    ("What is a garage?", "parking_locations_info", 0.4),
    ("Can tourists park here?", "parking_faq", 0.4),
])
def test_vector_search(vector_store, query, collection, min_score):
    """Test semantic search returns relevant results."""
    results = vector_store.search(
        collection_name=collection,
        query=query,
        top_k=2
    )
    
    assert len(results) > 0, f"No results for query: {query}"
    
    # Check first result has good relevance
    assert results[0]['score'] >= min_score, f"Low relevance score: {results[0]['score']}"
    
    # Check result structure
    assert 'text' in results[0]
    assert 'source_file' in results[0]
    assert 'chunk_index' in results[0]


def test_all_collections_exist(vector_store):
    """Test that all expected collections exist."""
    expected_collections = [
        "parking_zone_rules",
        "parking_pricing",
        "parking_locations_info",
        "parking_faq"
    ]
    
    for collection in expected_collections:
        # Try searching - will fail if collection doesn't exist
        results = vector_store.search(collection, "test query", top_k=1)
        assert results is not None
