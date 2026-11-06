"""Labelled evaluation dataset: test queries plus ground truth for retrieval and routing.

Relevance is defined per source file (data/static/*.md), because chunk ids are not stable
between ingestions. Each static file is indexed as its own Milvus collection:

    zone_system.md        -> parking_zone_rules
    pricing_and_hours.md  -> parking_pricing
    parking_types.md      -> parking_locations_info
    faq.md                -> parking_faq
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalQuery:
    """A test query with its ground-truth labels."""

    query: str
    expected_intent: str
    expected_source: str  # "vector_db" | "sql" | "hybrid" | "none"
    # Static files whose chunks are relevant. Empty = answer is not in the vector DB
    # (SQL / reservation queries), so the query is skipped in Recall@K / Precision@K.
    relevant_sources: frozenset[str] = frozenset()
    expected_keywords: tuple[str, ...] = ()  # should appear in the generated answer


EVAL_QUERIES = [
    EvalQuery(
        query="What are the rules for parking in zone A?",
        expected_intent="general_info",
        expected_source="vector_db",
        relevant_sources=frozenset({"zone_system.md"}),
        expected_keywords=("30 minutes", "purple", "zone a"),
    ),
    EvalQuery(
        query="How much does parking cost in the red zone?",
        expected_intent="pricing_hours",
        expected_source="hybrid",
        relevant_sources=frozenset({"pricing_and_hours.md"}),
        expected_keywords=("red", "zone", "rsd", "hour"),
    ),
    EvalQuery(
        query="What's the difference between a garage and a car park?",
        expected_intent="general_info",
        expected_source="vector_db",
        relevant_sources=frozenset({"parking_types.md", "faq.md"}),
        expected_keywords=("garage", "car park", "covered", "outdoor"),
    ),
    EvalQuery(
        query="When is parking free in zone 3?",
        expected_intent="pricing_hours",
        expected_source="hybrid",
        relevant_sources=frozenset({"zone_system.md", "pricing_and_hours.md"}),
        expected_keywords=("zone 3", "green", "free", "5 pm"),
    ),
    EvalQuery(
        query="Can tourists park in Belgrade?",
        expected_intent="general_info",
        expected_source="vector_db",
        relevant_sources=frozenset({"faq.md"}),
        expected_keywords=("tourist", "park", "belgrade"),
    ),
    EvalQuery(
        query="How can I pay for parking, can I use SMS?",
        expected_intent="general_info",
        expected_source="vector_db",
        relevant_sources=frozenset({"pricing_and_hours.md", "faq.md"}),
        expected_keywords=("sms", "app"),
    ),
    EvalQuery(
        query="Can I extend my parking time in the white zone?",
        expected_intent="general_info",
        expected_source="vector_db",
        relevant_sources=frozenset({"zone_system.md", "pricing_and_hours.md", "faq.md"}),
        expected_keywords=("60", "extension"),
    ),
    EvalQuery(
        query="What is a reserved garage and how does booking one work?",
        expected_intent="general_info",
        expected_source="vector_db",
        relevant_sources=frozenset({"parking_types.md", "faq.md"}),
        expected_keywords=("reserved", "advance"),
    ),
    # Dynamic data: answered from SQL, no vector-DB ground truth.
    EvalQuery(
        query="Show me available garages",
        expected_intent="availability_search",
        expected_source="sql",
        expected_keywords=("garage", "available", "spots"),
    ),
    EvalQuery(
        query="Are there parking spots available near Clinical Centre?",
        expected_intent="availability_search",
        expected_source="sql",
        expected_keywords=("clinical centre", "available"),
    ),
    # Reservation flow: handled by the pipeline, no retrieval.
    EvalQuery(
        query="I want to book a parking spot",
        expected_intent="book_spot",
        expected_source="none",
    ),
]
