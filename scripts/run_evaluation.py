"""
Run RAG system evaluation and generate report.

Usage:
    python scripts/run_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from parking_assistant.evaluation.metrics import run_evaluation


if __name__ == "__main__":
    print("Starting RAG system evaluation...\n")
    asyncio.run(run_evaluation())
