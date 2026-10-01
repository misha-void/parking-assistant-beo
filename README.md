# Parking Assistant — Belgrade

AI chatbot for parking information and reservations in Belgrade, Serbia. Answers questions over a
RAG knowledge base, runs an interactive reservation flow, escalates to an admin for approval, and
records confirmed bookings via an MCP server — all orchestrated as a single **LangGraph** pipeline.

## Architecture

```
          Streamlit UI (app.py)
                  │  process_chat_message(session_id, text)
                  ▼
        LangGraph pipeline (parking_assistant/graph/pipeline.py)
                  │
   route_intent ──┼── (info question) ──► RAG agent ──► answer ──► END
        │ (book_spot)
        ▼
     collect ◄─┐  one slot per turn via interrupt()  (name, surname,
        │      └─ loop until complete                  car, location, dates)
        ▼
     confirm ── interrupt() "yes/no" ──► (no) END
        │ (yes)
        ▼
      save  ── writes Reservation (PENDING_APPROVAL) to SQLite
        │
        ▼
      admin ── escalate_to_admin() notifies admin (email, or console fallback) ──► END
               "submitted for approval"

   Out of band — admin_app.py (Streamlit):
      admin reviews pending requests ──► Confirm/Refuse ──► status=CONFIRMED/CANCELLED
                                                            ──► (on confirm) MCP server records it
```

The whole conversation is **one resumable graph run per session**. Each question to the user is a
LangGraph `interrupt()`; the next message resumes the run from a checkpoint (`MemorySaver`). State
lives in the checkpointer keyed by `session_id`, and recent chat history is threaded into the
RAG/intent path for conversation memory, so the UI holds no conversation logic.

### Components / agent & server logic

- **RAG agent** (`rag/`): intent classification (5 intents) → retrieval (Milvus for static docs,
  SQLite for dynamic data) → grounded answer. PII guardrail (Presidio) filters retrieved context.
- **Reservation slot-filling** (`graph/slot_collector.py`): Pydantic `ReservationSlots`, car-plate
  validation, LLM-based date parsing and location matching. Driven node-by-node by the graph.
- **Admin approval** (`admin/`): `escalate_to_admin` generates an LLM summary and notifies the admin
  (SMTP email, or prints to console if SMTP is unset). A human admin approves in **`admin_app.py`**
  (a minimal Streamlit UI listing pending requests with slot details + Confirm/Refuse buttons), which
  calls `decide_reservation` to set the status and, on confirm, record to the MCP server.
  `record_decision` keeps the token-based confirm/refuse path (unit-tested).
- **MCP server** (`mcp_server/`): API-key-protected FastAPI service that appends confirmed
  reservations to a text file. The pipeline calls it via `mcp_server/client.py` (non-blocking if
  unreachable, so approval is never blocked).

## Setup

```bash
# 1. Python 3.12 virtual env (the ecosystem is not ready for 3.13+)
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # for the Presidio guardrail

# 3. Environment
cp .env.example .env                            # add OPENAI_API_KEY; MCP/SMTP vars optional

# 4. Initialize data
python data_ingestions/seed_dynamic_db.py       # seed SQLite
python data_ingestions/ingest_static_to_milvus.py  # embed static docs into Milvus
```

Relevant env vars: `OPENAI_API_KEY` (required); `MCP_SERVER_URL`, `MCP_API_KEY` (MCP recording);
`SMTP_HOST/PORT/USER/PASSWORD`, `ADMIN_EMAIL`, `APP_BASE_URL` (email notifications — omit for console).

## Run

```bash
# Start the MCP server (separate terminal) so confirmed bookings are recorded
uvicorn mcp_server.main:app --port 8001

# Start the chat UI
streamlit run app.py

# Start the admin approval UI (separate terminal) to confirm/refuse requests
streamlit run admin_app.py --server.port 8502
```

Without the MCP server the chat still works; the recording step just logs that it was unreachable.
When a user submits a reservation it becomes **pending**; approve it in the admin app to confirm and
record it.

## Testing

```bash
pytest tests/                        # unit tests (slots, DB, guardrails, admin, MCP)
python scripts/run_evaluation.py     # RAG eval: Recall@K / Precision@K / latency
```

## Project structure

```
app.py                     # Streamlit chat UI (user-facing)
admin_app.py               # Streamlit admin UI (confirm/refuse pending reservations)
parking_assistant/
├── graph/
│   ├── pipeline.py        # LangGraph orchestration (Stage 4) — the pipeline above
│   └── slot_collector.py  # reservation slots + validation/parsing helpers
├── rag/                   # intent classifier, retriever, RAG chain
├── guardrails/            # Presidio PII filtering
├── admin/                 # admin approval agent + notifier
├── evaluation/            # RAG metrics
├── db/                    # SQLAlchemy models + session
├── llm_factory.py         # pluggable chat/embedding model factory
└── service.py             # thin wrapper: process_chat_message -> graph run_turn
mcp_server/                # FastAPI MCP service + client
data/                      # static docs (Milvus) + parking_locations.json (SQLite)
data_ingestions/           # seed / ingest scripts
tests/ · scripts/          # pytest suite · eval runner
```

## Stage status

- [x] Stage 1 — RAG + chatbot, vector DB, reservation collection, guardrails, evaluation
- [x] Stage 2 — Admin approval agent (LLM summary + email/console notify + approval token)
- [x] Stage 3 — MCP server (writes confirmed reservations to file)
- [x] Stage 4 — LangGraph orchestration of the full pipeline (chat → approval → MCP record)

## Dependencies

Python 3.12 · LangChain 0.3 / LangGraph 0.2 · OpenAI · Milvus Lite · SQLite + SQLAlchemy ·
Presidio · Streamlit · FastAPI (MCP server only).

Learning project — MIT License.
