# Stage 1 — Environment Setup & Project Skeleton

**Date**: 2024 (Session 1)  
**Status**: In Progress  
**Owner**: AI (scaffold) + Human (secrets, verification)

---

## Decision

Set up the foundational project structure, Python 3.12 environment, and a minimal working Streamlit + LangChain skeleton to validate the toolchain before implementing any business logic.

---

## Scope of This Step

1. **Package layout**: `parking_assistant/` with subpackages for future modules (`rag/`, `guardrails/`, `db/`, `graph/`, `ui/`).
2. **Virtual environment**: Rebuild `.venv` on Python 3.12 (currently 3.14, which has poor ecosystem compatibility).
3. **Initial dependencies** (minimal set, not full stack):
   - `langchain`
   - `langgraph`
   - `openai`
   - `streamlit` (for demo/testing UI)
   - `fastapi` + `uvicorn` (backend API for programmatic access, enables future Telegram bot)
   - `python-dotenv`
4. **Skeleton code**:
   - `parking_assistant/llm_factory.py`: Factory function to instantiate `ChatOpenAI` from environment config.
   - `parking_assistant/service.py`: Core chat service (business logic) that both UIs will call — keeps logic DRY.
   - `app.py`: Streamlit chat UI calling `service.py`.
   - `api.py`: FastAPI app with a `/chat` endpoint calling `service.py`.
   - `.env.example`: Template showing required environment variables (`OPENAI_API_KEY`).
   - `main.py`: Updated to be a note pointing to both interfaces.

---

## Rationale

- **Python 3.12 instead of 3.14**: LangChain, Milvus SDKs, and Presidio don't yet have stable wheels for 3.14. 3.12 is the current sweet spot for ML/AI tooling.
- **Dual interface (Streamlit + FastAPI)**: Streamlit for interactive demos/testing; FastAPI for programmatic access (future Telegram bot, webhooks, or other clients). Shared business logic in `service.py` avoids duplication.
- **Minimal dependencies initially**: Only adding what we'll actually use in the skeleton. Milvus, Presidio, SQLAlchemy, etc. added incrementally in later steps as we implement those features — avoids dependency bloat and keeps troubleshooting simpler.
- **Trivial LangChain call**: Not implementing any real prompt/chain yet — just proving the OpenAI integration works end-to-end before we build RAG/graph logic on top of it.

---

## Alternatives Considered & Rejected

| Alternative | Why Rejected |
|---|---|
| Keep Python 3.14 | Too new; wheels for key dependencies not yet available, would cause friction |
| Start with full dependency list (Milvus, Presidio, SQLAlchemy, etc.) | Premature — we haven't designed those modules yet; add incrementally per sub-task |
| CLI REPL interface | Human prefers GUI; Streamlit provides better UX for testing |
| Streamlit-only (no FastAPI) | Rejected — human wants to build Telegram bot later, needs API endpoints |

---

## Delegation

### AI implements:
- Directory structure (`parking_assistant/` package + subpackage stubs)
- `requirements.txt` (add `fastapi` + `uvicorn` to dependency list)
- Rebuild `.venv` with Python 3.12, upgrade pip, install dependencies
- `parking_assistant/__init__.py`, `parking_assistant/llm_factory.py`, `parking_assistant/service.py`
- `app.py` (Streamlit skeleton)
- `api.py` (FastAPI skeleton with `/chat` POST endpoint)
- `.env.example`
- Update `main.py`

### Human implements:
- Copy `.env.example` → `.env`, add real `OPENAI_API_KEY`
- Run `streamlit run app.py` to verify Streamlit UI works
- Run `uvicorn api:app --reload` and test `/chat` endpoint (e.g., via `curl` or Postman) to verify FastAPI works

**Rationale for split**: All scaffolding/boilerplate delegated to AI for speed; human owns secrets management and verification. No interesting design decisions in this step — it's pure setup.

---

## Package Structure (Proposed)

```
parking_assistant/
├── __init__.py
├── llm_factory.py       # LLM/embedding model instantiation from env config
├── rag/                 # (future) Retrieval logic, Milvus adapter
│   └── __init__.py
├── guardrails/          # (future) Presidio PII filtering
│   └── __init__.py
├── db/                  # (future) SQLAlchemy models for dynamic data
│   └── __init__.py
├── graph/               # (future) LangGraph state graph (nodes/edges)
│   └── __init__.py
└── ui/                  # (future) Streamlit-specific helpers if needed
    └── __init__.py
```

Root-level files:
- `app.py` — Streamlit entry point
- `main.py` — Note pointing to `streamlit run app.py`
- `requirements.txt` — Dependencies
- `.env.example` — Template for environment variables
- `.env` — (gitignored) Actual secrets
- `docs/` — Architecture and decision logs

---

## Open Questions / Follow-ups

- None for this step (skeleton is intentionally trivial).
- Next step after verification: decide how to split static vs. dynamic data, design vector DB schema, implement retrieval.

---

## Success Criteria

- [x] `.venv` rebuilt on Python 3.12.7, dependencies installed without errors
- [ ] Human creates `.env` with real OpenAI API key
- [ ] `streamlit run app.py` launches, displays a chat interface
- [ ] Typing a message sends it to OpenAI via LangChain, response appears in chat
- [ ] `uvicorn api:app --reload` launches, `/health` endpoint responds  
- [ ] POST `/chat` endpoint works (test via curl or http://localhost:8000/docs)
- [x] No business logic implemented yet — just proving the toolchain works
