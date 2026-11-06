"""Evaluation metrics for the RAG system: retrieval quality, answer accuracy and latency."""

import asyncio
import math
import sys
import time
from dataclasses import dataclass, field
from statistics import mean

from parking_assistant.evaluation.eval_dataset import EVAL_QUERIES, EvalQuery
from parking_assistant.llm_factory import get_chat_model
from parking_assistant.rag.chain import RAGChain
from parking_assistant.rag.vector_store import MilvusVectorStorage

K_VALUES = (3, 5)
LINE = "=" * 80


# --------------------------------------------------------------------------- pure metric maths

def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: smallest value with at least p% of the data at or below it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def precision_at_k(retrieved_sources: list[str], relevant: frozenset[str], k: int) -> float:
    """Precision@K = (top-K retrieved chunks whose source_file is relevant) / K.

    Duplicate chunks from the same file each count, because each is a retrieved chunk.
    """
    top_k = retrieved_sources[:k]
    return sum(1 for source in top_k if source in relevant) / k


def recall_at_k(retrieved_sources: list[str], relevant: frozenset[str], k: int) -> float:
    """Recall@K = (distinct relevant source_files in the top-K chunks) / (# relevant files).

    Ground truth is file-level, so recall asks "how many of the files that hold the answer
    did we reach?", not "how many relevant chunks did we find?".
    """
    found = relevant & set(retrieved_sources[:k])
    return len(found) / len(relevant)


def keyword_recall(response: str, expected_keywords: tuple[str, ...]) -> float:
    """Fraction of expected keywords that appear in the generated answer."""
    if not expected_keywords:
        return 1.0
    response_lower = response.lower()
    return sum(1 for kw in expected_keywords if kw.lower() in response_lower) / len(
        expected_keywords
    )


# --------------------------------------------------------------------------- result containers

@dataclass
class RetrievalResult:
    """Retrieval metrics for one labelled query, keyed by K."""

    query: str
    retrieved_sources: list[str]
    precision: dict[int, float]
    recall: dict[int, float]


@dataclass
class QueryResult:
    """End-to-end (intent, routing, answer, latency) outcome for one query."""

    query: str
    latency_ms: float
    error: str | None = None
    intent: str | None = None
    intent_ok: bool = False
    source: str | None = None
    source_ok: bool = False
    keyword_recall: float | None = None


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    retrieval: list[RetrievalResult]
    queries: list[QueryResult]
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def failures(self) -> list[QueryResult]:
        """Queries whose end-to-end run raised an error."""
        return [q for q in self.queries if q.error]

    def macro(self, metric: str, k: int) -> float:
        """Macro average of 'precision' or 'recall' at K over the labelled queries."""
        values = [getattr(r, metric)[k] for r in self.retrieval]
        return mean(values) if values else 0.0


# --------------------------------------------------------------------------- computation

def search_top_k(store: MilvusVectorStorage, query: str, k: int) -> list[dict]:
    """Search every collection and merge by score (same strategy as Retriever).

    Takes k hits from each collection so that the merged top-k is exact.
    """
    hits = []
    for collection in store.ALL_COLLECTIONS:
        hits.extend(store.search(collection, query, top_k=k))
    hits.sort(key=lambda hit: hit["score"], reverse=True)
    return hits[:k]


def evaluate_retrieval(
    store: MilvusVectorStorage,
    queries: list[EvalQuery],
    k_values: tuple[int, ...] = K_VALUES,
) -> list[RetrievalResult]:
    """Compute Precision@K and Recall@K for each query that has relevant_sources."""
    results = []
    max_k = max(k_values)
    for item in queries:
        if not item.relevant_sources:
            continue
        hits = search_top_k(store, item.query, max_k)
        sources = [hit["source_file"] for hit in hits]
        results.append(
            RetrievalResult(
                query=item.query,
                retrieved_sources=sources,
                precision={k: precision_at_k(sources, item.relevant_sources, k) for k in k_values},
                recall={k: recall_at_k(sources, item.relevant_sources, k) for k in k_values},
            )
        )
    return results


async def evaluate_single_query(chain: RAGChain, item: EvalQuery) -> QueryResult:
    """Run one query end-to-end and compare intent, source routing and keywords."""
    start = time.perf_counter()
    try:
        result = await chain.run(item.query)
    except Exception as error:  # a failed query is recorded, not fatal
        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResult(item.query, latency_ms, error=str(error))
    latency_ms = (time.perf_counter() - start) * 1000

    source = result.get("metadata", {}).get("source", "unknown")
    outcome = QueryResult(
        query=item.query,
        latency_ms=latency_ms,
        intent=result.get("intent"),
        intent_ok=result.get("intent") == item.expected_intent,
        source=source,
        source_ok=item.expected_source == "none" or source == item.expected_source,
    )
    response = result.get("response")
    if response and item.expected_keywords:
        outcome.keyword_recall = keyword_recall(response, item.expected_keywords)
    return outcome


async def evaluate_rag_system(queries: list[EvalQuery] | None = None) -> EvaluationReport:
    """Run retrieval and end-to-end evaluation and return the data (no printing)."""
    queries = queries or EVAL_QUERIES

    store = MilvusVectorStorage()
    try:
        retrieval = evaluate_retrieval(store, queries)
    finally:
        store.close()

    chain = RAGChain(get_chat_model(temperature=0.0), enable_guardrails=True)
    try:
        results = [await evaluate_single_query(chain, item) for item in queries]
    finally:
        chain.close()

    latencies = [r.latency_ms for r in results]
    latency_summary = {
        "mean": mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
    }
    return EvaluationReport(retrieval, results, latency_summary)


# --------------------------------------------------------------------------- reporting

def print_evaluation_report(report: EvaluationReport) -> None:
    """Print the evaluation report to the console."""
    print(f"\n{LINE}\nRAG SYSTEM EVALUATION REPORT\n{LINE}")

    print(f"\nRetrieval quality ({len(report.retrieval)} labelled queries)")
    header = "".join(f"  P@{k:<3} R@{k:<3}" for k in K_VALUES)
    print(f"  {'Query':<58}{header}")
    for r in report.retrieval:
        scores = "".join(f"  {r.precision[k]:<5.2f} {r.recall[k]:<5.2f}" for k in K_VALUES)
        print(f"  {r.query[:56]:<58}{scores}")
    averages = "".join(
        f"  {report.macro('precision', k):<5.2f} {report.macro('recall', k):<5.2f}"
        for k in K_VALUES
    )
    print(f"  {'MACRO AVERAGE':<58}{averages}")

    answered = [q for q in report.queries if not q.error]
    total = len(report.queries)
    keyword_scores = [q.keyword_recall for q in answered if q.keyword_recall is not None]
    print(f"\nEnd-to-end accuracy ({total} queries)")
    print(f"  Intent classification: {sum(q.intent_ok for q in answered) / total:.1%}")
    print(f"  Source routing:        {sum(q.source_ok for q in answered) / total:.1%}")
    print(f"  Answer keyword recall: {mean(keyword_scores) if keyword_scores else 0.0:.1%}")

    lat = report.latency_ms
    print("\nLatency")
    print(f"  Mean {lat['mean']:.0f} ms | p50 {lat['p50']:.0f} ms | p95 {lat['p95']:.0f} ms "
          f"| p99 {lat['p99']:.0f} ms")
    print(f"  Min {lat['min']:.0f} ms | Max {lat['max']:.0f} ms")

    misses = [q for q in answered if not (q.intent_ok and q.source_ok)]
    if misses:
        print("\nMismatches")
        for q in misses:
            print(f"  - {q.query}: intent={q.intent} source={q.source}")
    if report.failures:
        print(f"\nFailures ({len(report.failures)})")
        for q in report.failures:
            print(f"  - {q.query}: {q.error}")
    print(f"\n{LINE}")


async def run_evaluation() -> EvaluationReport:
    """Main evaluation entry point."""
    print(f"Evaluating RAG system with {len(EVAL_QUERIES)} queries...")
    report = await evaluate_rag_system()
    print_evaluation_report(report)
    return report


def _self_check() -> None:
    """Offline sanity checks of the metric maths (no network)."""
    data = list(range(1, 101))
    assert (percentile(data, 50), percentile(data, 95), percentile(data, 99)) == (50, 95, 99)
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8], 95) == 8 and percentile([1, 2, 3, 4], 50) == 2
    assert percentile([7], 95) == 7 and percentile([], 95) == 0.0

    relevant = frozenset({"a.md", "b.md"})
    retrieved = ["a.md", "x.md", "a.md", "b.md", "x.md"]
    assert precision_at_k(retrieved, relevant, 3) == 2 / 3
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert precision_at_k(retrieved, relevant, 5) == 3 / 5
    print("metrics self-check passed")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        asyncio.run(run_evaluation())
