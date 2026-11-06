"""Chat-history formatting shared by the intent classifier and the RAG chain."""


def format_history(history: list = None, limit: int = 6) -> str:
    """Format the last `limit` chat messages as 'User: ...' / 'Assistant: ...' lines."""
    return "\n".join(
        f"{'User' if message.get('role') == 'user' else 'Assistant'}: {message.get('content', '')}"
        for message in (history or [])[-limit:]
    )
