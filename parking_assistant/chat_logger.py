"""Append-only chat logging for quality review. Writes JSONL to .logs/ (never breaks the chat)."""

import os
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(os.getenv("CHAT_LOG_DIR", ".logs"))


def log_turn(session_id: str, user_message: str, assistant_response: str) -> None:
    """Append one conversation turn to .logs/chat.jsonl. Failures are swallowed."""
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session": session_id,
            "user": user_message,
            "assistant": assistant_response,
        }
        with open(LOG_DIR / "chat.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging must never interrupt the conversation
