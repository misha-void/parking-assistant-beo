# Parking Assistant — Belgrade

A chat assistant for parking in Belgrade: it answers questions from a RAG knowledge base,
walks the user through booking a spot, has an approval agent review the request, and records
confirmed bookings through a small MCP service. Built as a four-stage course project.

The whole conversation is orchestrated as a single **LangGraph** graph — see
[docs/architecture.md](docs/architecture.md) for the full design.

## How it works

```
 app.py (chat) ──▶ service.py ──▶ graph/pipeline.py ─┬─ RAG answer ─────────────▶ END
                                                     └─ collect slots (interrupts)
                                                        ─▶ confirm ─▶ save
                                                        ─▶ approval agent ──▶ MCP record
 admin_app.py (human override) ─────────────────────────▶ confirm / refuse ──▶ MCP record
```

- **Information Q&A** — intent classification picks the retrieval strategy: Milvus Lite for
  static documents, SQLite for live availability and per-location prices.
- **Booking** — six slots collected one question at a time. Each question is a LangGraph
  `interrupt()`, so the next message resumes the same graph run. Asking a question
  mid-booking gets answered, then the slot is asked again; saying "cancel" abandons it.
- **Approval** — by default an LLM agent reviews the request and approves or refuses it with
  a reason. Set `AUTO_APPROVE=false` to require the human in `admin_app.py` instead.
- **Recording** — on confirmation the booking is posted to the MCP service, which appends
  `Name | Car Number | Period | Approval Time` to a text file.

## Setup

```bash
# 1. Python 3.12 virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # for the Presidio guardrail

# 3. Environment
cp .env.example .env                            # then add your OPENAI_API_KEY

# 4. Data
python data_ingestions/seed_dynamic_db.py          # seed SQLite (re-runnable, but WIPES reservations)
python data_ingestions/ingest_static_to_milvus.py  # embed the static docs into Milvus
```

## Run

```bash
# Recording service (needs MCP_API_KEY set in .env)
uvicorn mcp_server:app --port 8001

# Chat UI
streamlit run app.py

# Admin approval UI (the human override)
streamlit run admin_app.py --server.port 8502
```

Set `MCP_API_KEY` in `.env` before starting the MCP service: both the service and whichever
process confirms a reservation read it, and the service rejects unauthenticated writes.

## Testing

```bash
pytest                                  # unit + integration of the graph; uses a scratch DB
RUN_INTEGRATION_TESTS=1 pytest          # also runs the live Milvus/OpenAI tests
python scripts/run_evaluation.py        # Recall@K / Precision@K / latency
```

The default run needs no API key and does not touch the development database. The tests
marked `integration` need `OPENAI_API_KEY` and ingested Milvus data, so they are skipped
unless `RUN_INTEGRATION_TESTS=1`.

## Project structure

```
app.py                      # Streamlit chat UI
admin_app.py                # Streamlit admin approval UI
parking_assistant/
├── graph/
│   ├── pipeline.py         # the LangGraph conversation graph
│   └── slots.py            # reservation slots + the LLM helpers that fill them
├── rag/                    # intent classifier, retriever, RAG chain, history helper
├── guardrails/             # Presidio PII filtering (context and output)
├── admin/                  # approval agent + notifier
├── evaluation/             # labelled dataset + Recall@K / Precision@K / latency
├── db/                     # SQLAlchemy models and session
├── llm_factory.py          # chat vs deterministic extraction models
├── service.py              # fail-safe entry point used by the UI
├── session_registry.py     # session → reservation, for status polling
└── chat_logger.py          # appends each turn to .logs/ for review
mcp_server.py               # the MCP recording service (FastAPI, run on its own)
data/                       # static docs (Milvus) + parking_locations.json (SQLite)
data_ingestions/            # seed and ingest scripts
docs/architecture.md        # full architecture
tests/ · scripts/           # pytest suite · evaluation runner
```

## Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | required |
| `AUTO_APPROVE` | `true` (default): the approval agent decides. `false`: wait for the admin app |
| `MCP_SERVER_URL`, `MCP_API_KEY` | the recording service |
| `MILVUS_DB_PATH` | Milvus Lite file (**not** `MILVUS_URI`, which pymilvus reserves) |
| `SMTP_*`, `ADMIN_EMAIL` | admin email; unset prints the notification to the console |
| `DATABASE_URL` | SQLite file (default `sqlite:///./parking_assistant.db`) |
| `MCP_RESERVATIONS_FILE` | where the service appends records (default `confirmed_reservations.txt`) |
| `CHAT_LOG_DIR` | per-turn chat logs (default `.logs`) |

## Stage status

- [x] **Stage 1** — RAG + chatbot, vector DB, reservation collection, PII guardrails, evaluation
- [x] **Stage 2** — Admin approval agent (LLM summary, notification, approve/refuse + reason)
- [x] **Stage 3** — MCP service recording confirmed reservations to file
- [x] **Stage 4** — LangGraph orchestration of the whole pipeline

Known limitations are listed in [docs/architecture.md](docs/architecture.md#9-known-limitations).

Course project, built for learning.
