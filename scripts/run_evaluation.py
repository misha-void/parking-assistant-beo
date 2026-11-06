"""
Run RAG system evaluation (Precision@K, Recall@K, accuracy, latency p50/p95/p99).

Usage:
    python scripts/run_evaluation.py
    python -m parking_assistant.evaluation.metrics --self-check   # offline metric check
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parking_assistant.evaluation.metrics import run_evaluation  # noqa: E402

if __name__ == "__main__":
    asyncio.run(run_evaluation())
