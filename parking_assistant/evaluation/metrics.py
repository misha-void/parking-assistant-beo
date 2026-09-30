"""Evaluation metrics for RAG system performance and accuracy."""

import time
import asyncio
from typing import List, Dict, Tuple
from dataclasses import dataclass
from statistics import mean, median

from parking_assistant.rag.chain import RAGChain
from parking_assistant.llm_factory import get_chat_model


@dataclass
class EvaluationQuery:
    """Test query with expected properties."""
    query: str
    expected_intent: str
    expected_keywords: List[str]  # Keywords that should appear in response
    expected_source: str  # "vector_db", "sql", "hybrid"


# Test dataset for evaluation
TEST_QUERIES = [
    EvaluationQuery(
        query="What are the rules for parking in zone A?",
        expected_intent="general_info",
        expected_keywords=["30 minutes", "purple", "zone a"],
        expected_source="vector_db"
    ),
    EvaluationQuery(
        query="How much does parking cost in the red zone?",
        expected_intent="pricing_hours",
        expected_keywords=["red", "zone", "rsd", "hour"],
        expected_source="hybrid"
    ),
    EvaluationQuery(
        query="Show me available garages",
        expected_intent="availability_search",
        expected_keywords=["garage", "available", "spots"],
        expected_source="sql"
    ),
    EvaluationQuery(
        query="What's the difference between a garage and a car park?",
        expected_intent="general_info",
        expected_keywords=["garage", "car park", "covered", "outdoor"],
        expected_source="vector_db"
    ),
    EvaluationQuery(
        query="I want to book a parking spot",
        expected_intent="book_spot",
        expected_keywords=[],
        expected_source="none"
    ),
    EvaluationQuery(
        query="Are there parking spots available near Clinical Centre?",
        expected_intent="availability_search",
        expected_keywords=["clinical centre", "available"],
        expected_source="sql"
    ),
    EvaluationQuery(
        query="When is parking free in zone 3?",
        expected_intent="pricing_hours",
        expected_keywords=["zone 3", "green", "free", "5 pm"],
        expected_source="hybrid"
    ),
    EvaluationQuery(
        query="Can tourists park in Belgrade?",
        expected_intent="general_info",
        expected_keywords=["tourist", "park", "belgrade"],
        expected_source="vector_db"
    ),
]


@dataclass
class PerformanceMetrics:
    """Performance/latency metrics."""
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for RAG responses."""
    intent_accuracy: float  # % correct intent classification
    keyword_recall: float   # % of expected keywords found in responses
    source_accuracy: float  # % correct data source routing


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    performance: PerformanceMetrics
    accuracy: AccuracyMetrics
    num_queries: int
    failures: List[Dict]


async def evaluate_single_query(
    chain: RAGChain,
    test_query: EvaluationQuery
) -> Tuple[float, Dict]:
    """
    Evaluate a single query.
    
    Returns:
        Tuple of (latency_ms, result_dict)
    """
    start_time = time.perf_counter()
    
    try:
        result = await chain.run(test_query.query)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return latency_ms, result
    
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return latency_ms, {
            "error": str(e),
            "intent": None,
            "response": None,
            "metadata": {}
        }


def calculate_keyword_recall(response: str, expected_keywords: List[str]) -> float:
    """Calculate recall of expected keywords in response."""
    if not expected_keywords:
        return 1.0  # No keywords to check
    
    response_lower = response.lower()
    
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    
    return found / len(expected_keywords)


async def evaluate_rag_system(test_queries: List[EvaluationQuery] = None) -> EvaluationReport:
    """
    Run full evaluation on RAG system.
    
    Args:
        test_queries: List of test queries (uses TEST_QUERIES if None)
    
    Returns:
        EvaluationReport with all metrics
    """
    if test_queries is None:
        test_queries = TEST_QUERIES
    
    llm = get_chat_model(temperature=0.0)  # Deterministic for evaluation
    chain = RAGChain(llm, enable_guardrails=True)
    
    latencies = []
    intent_correct = 0
    source_correct = 0
    keyword_recalls = []
    failures = []
    
    print(f"🧪 Evaluating RAG system with {len(test_queries)} queries...\n")
    
    try:
        for i, test_query in enumerate(test_queries, 1):
            print(f"[{i}/{len(test_queries)}] Testing: \"{test_query.query}\"")
            
            latency_ms, result = await evaluate_single_query(chain, test_query)
            latencies.append(latency_ms)
            
            # Check for errors
            if "error" in result:
                failures.append({
                    "query": test_query.query,
                    "error": result["error"]
                })
                print(f"  ❌ Failed: {result['error']}")
                continue
            
            # Check intent accuracy
            if result.get("intent") == test_query.expected_intent:
                intent_correct += 1
                print(f"  ✓ Intent: {result['intent']}")
            else:
                print(f"  ✗ Intent: got {result.get('intent')}, expected {test_query.expected_intent}")
            
            # Check source routing
            actual_source = result.get("metadata", {}).get("source", "unknown")
            if actual_source == test_query.expected_source or test_query.expected_source == "none":
                source_correct += 1
                print(f"  ✓ Source: {actual_source}")
            else:
                print(f"  ✗ Source: got {actual_source}, expected {test_query.expected_source}")
            
            # Check keyword recall
            response = result.get("response", "")
            if response and test_query.expected_keywords:
                recall = calculate_keyword_recall(response, test_query.expected_keywords)
                keyword_recalls.append(recall)
                print(f"  • Keyword recall: {recall:.2%}")
            
            print(f"  • Latency: {latency_ms:.0f}ms\n")
        
        # Calculate metrics
        latencies_sorted = sorted(latencies)
        n = len(latencies)
        
        performance = PerformanceMetrics(
            mean_latency_ms=mean(latencies),
            median_latency_ms=median(latencies),
            p95_latency_ms=latencies_sorted[int(0.95 * n)] if n > 0 else 0,
            p99_latency_ms=latencies_sorted[int(0.99 * n)] if n > 0 else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0
        )
        
        accuracy = AccuracyMetrics(
            intent_accuracy=intent_correct / len(test_queries),
            keyword_recall=mean(keyword_recalls) if keyword_recalls else 0.0,
            source_accuracy=source_correct / len(test_queries)
        )
        
        return EvaluationReport(
            performance=performance,
            accuracy=accuracy,
            num_queries=len(test_queries),
            failures=failures
        )
    
    finally:
        chain.close()


def print_evaluation_report(report: EvaluationReport):
    """Print formatted evaluation report."""
    print("\n" + "=" * 80)
    print("📊 RAG SYSTEM EVALUATION REPORT")
    print("=" * 80)
    
    print(f"\n📈 Performance Metrics ({report.num_queries} queries):")
    print(f"  Mean latency:      {report.performance.mean_latency_ms:.0f} ms")
    print(f"  Median latency:    {report.performance.median_latency_ms:.0f} ms")
    print(f"  P95 latency:       {report.performance.p95_latency_ms:.0f} ms")
    print(f"  P99 latency:       {report.performance.p99_latency_ms:.0f} ms")
    print(f"  Min/Max:           {report.performance.min_latency_ms:.0f} ms / {report.performance.max_latency_ms:.0f} ms")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"  Intent classification: {report.accuracy.intent_accuracy:.1%}")
    print(f"  Keyword recall:        {report.accuracy.keyword_recall:.1%}")
    print(f"  Source routing:        {report.accuracy.source_accuracy:.1%}")
    
    if report.failures:
        print(f"\n❌ Failures ({len(report.failures)}):")
        for failure in report.failures:
            print(f"  • {failure['query']}: {failure['error']}")
    else:
        print(f"\n✅ No failures!")
    
    print("\n" + "=" * 80)


async def run_evaluation():
    """Main evaluation entry point."""
    report = await evaluate_rag_system()
    print_evaluation_report(report)
    return report


if __name__ == "__main__":
    asyncio.run(run_evaluation())
