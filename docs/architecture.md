# Architecture

Parking Lot Chat Assistant for Belgrade — a course project built in four stages. This
document describes the system as the code actually behaves.

## 1. Overview

```
                 ┌──────────────────────── Streamlit ────────────────────────┐
   user ───────▶ │  app.py  (chat)                   admin_app.py  (approve) │ ◀─── admin
                 └───────┬───────────────────────────────────┬───────────────┘
                         │ process_chat_message              │ decide_reservation
                         ▼                                   │
              parking_assistant/service.py                   │
                         │ run_turn                          │
                         ▼                                   ▼
         ┌─────── graph/pipeline.py (LangGraph) ──────┐   admin/admin_agent.py
         │  route_message → collect_slot* → confirm   │   ├─ escalate_to_admin  ──▶ email/console
         │     → save_reservation → request_approval  │   ├─ auto_review_reservation (LLM agent)
         └──────┬───────────────────┬─────────────────┘   └─ decide_reservation / _apply_decision
                │ RAG               │ ORM                              │ REST + API key
                ▼                   ▼                                  ▼
        rag/chain.py          SQLite (SQLAlchemy)              mcp_server.py (FastAPI)
        ├─ intent_classifier  ├─ parking_location                └─ confirmed_reservations.txt
        ├─ retriever          ├─ price_rule / working_hours
        │   ├─ Milvus Lite    ├─ availability
        │   └─ SQLite         └─ reservation
        └─ guardrails/pii_filter
```

## 2. The conversation graph

One LangGraph `StateGraph`, compiled with a `MemorySaver` checkpointer, holds **one
resumable run per chat session** (`graph/pipeline.py`).

| Node | Responsibility |
|---|---|
| `route_message` | Answers status questions from the database, answers information questions via RAG, or starts a booking (pre-filling the location the conversation already settled on). |
| `collect_slot` | Asks for the next missing slot, then interprets and validates the reply. Loops onto itself until every slot is filled. |
| `confirm_with_user` | Shows the summary with an estimated price and reads a yes / no / unclear reply. Unclear re-asks instead of cancelling. |
| `save_reservation` | Writes the reservation as `PENDING_APPROVAL`. |
| `request_approval` | Notifies the admin, then (by default) lets the approval agent decide. |

```
START ─▶ route_message ─┬─ not a booking ──────────────────────────────▶ END
                        └─ booking ─▶ collect_slot ⇄ collect_slot
                                           │ all slots filled
                                           ▼
                                   confirm_with_user ─┬─ no ──────────▶ END
                                           ▲          └─ yes
                                           │ unclear      ▼
                                           └───────  save_reservation ─▶ request_approval ─▶ END
```

**Interrupts.** Every question put to the user is a LangGraph `interrupt()`. The run
pauses, `run_turn` returns the interrupt text, and the next message resumes the run with
`Command(resume=...)`. The UI therefore holds no conversation logic.

**Slots** (`graph/slots.py`) are asked in order: `name`, `surname`, `car_number`,
`location_id`, `start_iso`, `end_iso`. Slot keys are identical to the `ReservationSlots`
field names, so there is no mapping to keep in sync.

### Where state lives

| Store | Contents | Lifetime |
|---|---|---|
| LangGraph checkpointer (`MemorySaver`) | slots, prompt prefix, shown locations, history, reservation id | process memory; lost on restart |
| Streamlit `session_state` | rendered messages, session id, announced statuses | browser session |
| `session_registry` (dict) | session id → reservation id, for status polling | process memory |
| SQLite | locations, prices, hours, availability, reservations | durable; the only channel between the two apps |
| `confirmed_reservations.txt` | confirmed bookings, written by the MCP server | durable |

## 3. Agent logic

**RAG agent** (`rag/`). `classify_intent` picks one of five intents, which selects the
retrieval strategy: Milvus for static documents, SQLite for live availability and prices,
or both. The answer is generated only from the retrieved context plus recent conversation.
Ranking questions ("cheapest") route to `availability_search`, because only that path
carries per-location prices.

**Slot interpreter** (`graph/slots.interpret_slot_reply`). One deterministic LLM call per
reservation turn classifies the reply as `answer`, `question` or `unclear` and normalises
its value (a name out of a sentence, a plate, a location name, an ISO date-time). It
**fails closed**: output that cannot be parsed becomes `unclear` and the slot is asked
again, never stored. A question is answered via RAG and the slot is re-asked in the same
turn.

**Approval agent** (`admin/admin_agent.auto_review_reservation`). Simulates the human
administrator: an LLM reviews the request and replies `APPROVE` or `REFUSE` plus a short
reason, which is stored on the reservation and shown to the user. It is given the current
time, the guest and plate, the location with its zone and rate, the period, the duration
and the zone's maximum stay, and it judges exactly what those support: a period in the
past, an end before the start, an implausible length, a stay over the zone maximum, and
an obviously fake name or plate. Unparseable output leaves the reservation pending for a
human. Controlled by `AUTO_APPROVE` (default `true`);
with it off, reservations simply wait for the admin app.

**Human override.** `admin_app.py` lists pending requests with their details and price and
offers Confirm / Refuse. Both paths go through `_apply_decision`, so the status change and
the MCP write behave identically however the decision is made.

**Determinism.** `llm_factory` exposes `get_chat_model()` (temperature 0.7) for
conversational answers and `get_extraction_model()` (temperature 0) for classification,
interpretation and review, so routing and extracted values are reproducible.

## 4. Server logic — the MCP service

`mcp_server.py` is a small FastAPI service with one endpoint,
`POST /reservations/confirmed`, protected by an `x-api-key` header compared in constant
time. It appends one line per booking under a lock:

```
Name | Car Number | Period | Approval Time
```

`parking_assistant/admin/mcp_client.py` posts to it when a reservation is confirmed. It never raises — a
failed write must not block an approval — but it checks the response status and returns
`False`, and the admin sees a warning, so a lost record cannot look like success.

> **Note.** This is a plain REST service fulfilling the "MCP server" role described in the
> brief; it does not implement the Model Context Protocol (no MCP SDK, no JSON-RPC, no
> tools). Confirm against the assignment whether a protocol-level MCP server is required.

## 5. Guardrails

`guardrails/pii_filter.py` wraps Microsoft Presidio and filters both the **retrieved
context** and the **generated answer** for a deliberately narrow set of real secrets
(`EMAIL_ADDRESS`, `CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`).

`PERSON`, `PHONE_NUMBER` and `LOCATION` are intentionally excluded: they false-positive on
Belgrade street names and on the user's own booking details, and an earlier version
redacted parking spot counts such as `1800/2500` as phone numbers.

## 6. Evaluation

`python scripts/run_evaluation.py` (needs an API key and ingested Milvus data) reports,
over the labelled queries in `evaluation/eval_dataset.py`:

- **Precision@K** — of the top-K retrieved chunks, the share whose source file is relevant.
- **Recall@K** — of the files that should have been reached, the share present in the top-K.
  Ground truth is file-level, which is stable even when chunk ids change. K = 3 and 5.
- **Intent accuracy** and **source-routing accuracy** end to end.
- **Latency** — mean, p50, p95, p99 by nearest rank.

Measured on 2026-11-06 (8 labelled retrieval queries, 11 end-to-end):

| Metric | Value |
|---|---|
| Precision@3 / Recall@3 | 0.79 / 0.77 |
| Precision@5 / Recall@5 | 0.80 / 0.90 |
| Intent classification | 81.8% |
| Source routing | 81.8% |
| Answer keyword recall | 85.0% |
| Latency mean / p50 / p95 | 2927 / 2789 / 5200 ms |

Recall@5 is well above Recall@3 because the pricing query needs a chunk that ranks
fourth; the two routing misses are both hybrid answers classified as a neighbouring
intent, which still retrieve the right sources.

**Milvus Lite holds an exclusive lock on its file**, so the evaluation cannot run while
the chat app is open. Stop the Streamlit app first.

## 7. Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | required, for chat and embeddings |
| `AUTO_APPROVE` | `true` (default) lets the approval agent decide; `false` waits for a human |
| `MCP_SERVER_URL`, `MCP_API_KEY` | recording confirmed reservations — must be set for **both** processes |
| `MILVUS_DB_PATH` | Milvus Lite file. Not `MILVUS_URI`, which pymilvus reserves for its own HTTP connection |
| `SMTP_*`, `ADMIN_EMAIL` | admin email; unset means the notification prints to the console |
| `DATABASE_URL` | SQLite file; the test suite overrides it with a scratch database |
| `MCP_RESERVATIONS_FILE` | file the service appends to (default `confirmed_reservations.txt`) |
| `CHAT_LOG_DIR` | per-turn chat logs (default `.logs`) |

## 8. Stage mapping

| Stage | Where |
|---|---|
| 1 — RAG, vector DB, reservation collection, guardrails, evaluation | `rag/`, `graph/slots.py`, `guardrails/`, `evaluation/` |
| 2 — Admin approval agent | `admin/admin_agent.py`, `admin/notifier.py`, `admin_app.py` |
| 3 — MCP server | `mcp_server.py` + `admin/mcp_client.py` |
| 4 — LangGraph orchestration | `graph/pipeline.py`, `service.py`, `app.py` |

## 9. Known limitations

- `MemorySaver` keeps conversations in memory, so a restart loses an in-flight booking
  (the pending row survives in SQLite). A persistent checkpointer would fix this.
- The status notification needs the chat tab to stay open; there is no push channel.
- A submitted reservation cannot be edited or cancelled by the user.
- No availability or double-booking check: a booking does not reserve a physical space.
- The approval agent judges only the fields listed above. It cannot check opening hours,
  capacity or overlapping bookings.
- `admin_app.py` has no authentication — anyone who can reach the port can approve.
- Milvus Lite is single-process: the chat app and the evaluation script cannot hold the
  database file at the same time.
