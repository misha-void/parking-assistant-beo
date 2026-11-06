# CLAUDE.md — working notes for this project

Parking Lot Chat Assistant (Belgrade) — a four-stage course project done at work, not production
code. Read this file first: it records the decisions that are not obvious from the code.

Everything agent-related lives under `.claude/` plus this file. See `docs/architecture.md` for the
design and `README.md` for setup and run commands.

---

## 1. Confirmed decisions

- **Goal**: chatbot for Belgrade parking — information Q&A, a guided reservation flow, and
  human-in-the-loop approval before a booking is recorded.
- **Python 3.12** in `.venv` (3.13+ is still too new for parts of the ecosystem).
- **Orchestration**: the whole conversation is one LangGraph `StateGraph`
  (`parking_assistant/graph/pipeline.py`) with an in-memory `MemorySaver`. Nodes:
  `route_message` → `collect_slot`* → `confirm_with_user` → `save_reservation` →
  `request_approval`. Every question to the user is an `interrupt()`, resumed per session id;
  chat history is threaded into the RAG and intent path. Slots and their LLM helpers are in
  `graph/slots.py`.
- **Interface**: Streamlit only — `app.py` (chat) and `admin_app.py` (admin override). The
  FastAPI chat backend (`api.py`) was deleted in Stage 4 to keep the codebase small.
- **Vector DB**: Milvus Lite, embedded, called directly from `rag/retriever.py`. There is no
  adapter layer; swapping to Pinecone/Weaviate would mean editing that module.
- **Dynamic data**: SQLite via SQLAlchemy — locations, prices, working hours, availability,
  reservations. Static documents are embedded into Milvus.
- **LLM**: OpenAI only, via `llm_factory.py` — `gpt-4o-mini` and `text-embedding-3-small`.
  `get_chat_model()` (temp 0.7) answers conversationally; `get_extraction_model()` (temp 0) does
  classification, slot interpretation and the approval review, so routing and extracted values are
  reproducible.
- **Approval**: an LLM agent (`admin_agent.auto_review_reservation`) approves or refuses with a
  stored reason, and the MCP recording happens on approval. `AUTO_APPROVE=false` hands every
  decision to the human in `admin_app.py`.
- **Guardrails**: Presidio filters retrieved context *and* generated output, for real secrets only
  (email / card / IBAN / IP). PERSON, PHONE_NUMBER and LOCATION are deliberately excluded — they
  false-positive on Belgrade street names and on the user's own booking details (spot counts like
  "1800/2500" were once redacted as `[PHONE]`).
- **Evaluation**: own scripts — Recall@K, Precision@K, latency p50/p95 (`evaluation/`,
  `scripts/run_evaluation.py`). RAGAS was considered and dropped as overkill.
- **Language**: English only.

### Open question

- `mcp_server/` is a plain REST service filling the MCP role, **not** a Model Context Protocol
  implementation. Confirm against the course brief whether a protocol-level MCP server is required.

Any decision that changes the above belongs back in this section.

---

## 2. Two bugs worth not repeating

- **Config read at import time is read before `.env` exists.** A module-level
  `AUTO_APPROVE = os.getenv(...)` silently ignored the setting, and `admin_app.py` /
  `mcp_server/main.py` read an empty `MCP_API_KEY` and 401'd every write. Entry points call
  `load_dotenv()` before importing project modules; flags are read inside the function.
- **`MILVUS_URI` is reserved by pymilvus** for its own HTTP connection and crashes on a file path.
  The project variable is `MILVUS_DB_PATH`.

---

## 3. How to work here

- Keep it concise and readable for review — course-work ready, not enterprise. No speculative
  abstraction layers.
- Generate code with Sonnet or Haiku; raise genuinely critical decisions instead of guessing.
- PEP 8, `ruff` config in `pyproject.toml` (line length 100).
- Commit each meaningful change as it lands.
- Decision specs are kept outside this repo. Do not create `docs/decisions/`.

The original brief asked for a learn-as-you-go loop — propose the smallest next step, agree who
writes what, then a short quiz at the end of a stage. All four stages are built, so this now
applies only to new work.

---

## 4. Stages

- [x] **Stage 1** — RAG + chatbot, vector DB, reservation collection, PII guardrails, evaluation
- [x] **Stage 2** — admin approval agent (LLM summary, notification, approve/refuse with a reason)
- [x] **Stage 3** — MCP service recording confirmed reservations to a text file
- [x] **Stage 4** — LangGraph orchestration of the whole pipeline, plus a cleanup pass: dead code
      removed, PEP 8, `docs/architecture.md`, real Recall@K/Precision@K, output guardrail, tests on
      a scratch DB that stay green without an API key
