# AGENTS.md — Operating Instructions for CodeMie Developer on this Project

This file governs how the AI assistant ("agent") must work on the **Parking Lot Chat Assistant (Serbia)**
project. It is a learning project for the human developer (recovering AI/ML engineering skills after
long sick leave), so the process matters as much as the code. Read this file at the start of every
session before doing anything else.

---

## 0. Project Snapshot (confirmed decisions so far)

- **Goal**: Build an intelligent chatbot for parking lots in Serbia — info Q&A, reservation flow,
    human-in-the-loop confirmation. Built in 4 stages.
  - **Stage 2 (done)**: Admin approval agent (`parking_assistant/admin/`) — LangChain-generated
    summary, email notification (console fallback if SMTP unset) with confirm/refuse links, decision
    recorded via approval token.
  - **Stage 3 (done)**: MCP server (`mcp_server/`) — FastAPI service, API-key protected, writes
    confirmed reservations to a text file (`Name | Car Number | Period | Approval Time`); admin agent
    notifies it via `mcp_server/client.py` (non-blocking if unreachable).
- **Language**: Python 3.12 (rebuild `.venv`, currently on 3.14 which is too new for the ecosystem).
- **Orchestration**: LangChain + LangGraph.
- **Vector DB**: Milvus Lite (embedded, no infra), behind an interface/adapter so it can be swapped
  for Pinecone/Weaviate later without rewriting business logic.
- **Dynamic data**: SQLite via SQLAlchemy (availability, prices, working hours, reservations).
- **Static data**: Embedded into Milvus (general info, parking details, location, booking process docs).
- **LLM/Embeddings**: OpenAI confirmed (e.g., `gpt-4o-mini` + `text-embedding-3-small`), behind a
  pluggable factory so other providers can be tested later for eval purposes.
- **Guardrails**: Microsoft Presidio (pretrained NER + regex) for PII filtering on retrieved context
  and LLM output.
- **Evaluation**: Custom lightweight scripts — Recall@K, Precision@K, latency (p50/p95). RAGAS is a
  possible stretch goal later, not the default.
- **Language support**: English only (no bilingual EN/SR requirement).
- **Interface**: Streamlit chat UI only (`app.py`). The standalone FastAPI chat backend (`api.py`)
  was removed in Stage 4 to keep the codebase concise; the MCP server remains a FastAPI service.
- **Orchestration (Stage 4)**: the full conversation is a single LangGraph `StateGraph`
  (`parking_assistant/graph/pipeline.py`) with an in-memory `MemorySaver` checkpointer. User prompts
  are `interrupt()`s resumed per session; chat history is threaded into the RAG/intent path for
  conversation memory. The `admin` node escalates (email, or console fallback) and ends the turn with
  "submitted for approval". **Real human-in-the-loop approval** happens out of band in a Streamlit
  admin app (`admin_app.py`, confirm/refuse buttons over `admin_agent.decide_reservation`); the MCP
  recording happens on the admin's confirmation. `record_decision` keeps the token-based path.

Any future decision that changes the above MUST be reflected back into this section by the agent.

---

## 1. The Mandatory Per-Task Workflow

The human wants to *learn*, not just receive finished code. For **every task/sub-task** (roughly
each numbered item within a stage, e.g. "Stage 1, item 3: interactive features"), the agent MUST
follow this exact 5-step loop, in order, without skipping steps or collapsing them:

### Step 1 — Next Minimal Step(s)
- Before writing any code, propose the *smallest* reasonable next increment of work.
- Explain what it accomplishes and why it's the right size (not too big, not trivial busywork).
- Wait for human confirmation/adjustment before proceeding, unless explicitly told to keep going.

### Step 2 — Delegation Discussion
- Explicitly discuss: what should the AI implement/do, and what should the human do themselves.
- Default bias: delegate to the human anything that is core to the skill they're trying to
  re-acquire (e.g., writing the LangGraph node logic, prompt design, embedding/query logic,
  evaluation metric implementation). The agent should lean toward scaffolding, boilerplate,
  research/summarization, debugging assistance, and reviewing — not silently doing 100% of the
  "interesting" work unless the human asks for that explicitly for a given piece.
- Make this a real conversation/negotiation, not a formality — ask the human what they want to
  own for this particular step.

### Step 3 — Decision Spec Written to Markdown
- After the discussion converges, write (or append to) a decision spec `.md` file documenting:
  - what was decided, why, alternatives considered/rejected.
  - who implements what (AI vs. human) for this step.
  - any open questions / follow-ups.
- File location convention: outside this repo (human keeps decision specs separately). Do not
  create `docs/decisions/*.md` files going forward.
- Keep specs concise and skimmable — bullet points over prose, headers per section.

### Step 4 — Skill-Building Support
- Help the human acquire whatever skills/knowledge are needed to do their delegated part
  (per Step 2). This can include: pointing to concepts to learn, short explanations of key
  ideas/APIs, minimal illustrative examples (NOT the full solution), links to relevant docs,
  suggested exercises, or answering their clarifying questions.
- This step should build competence, not do the human's homework for them. If the human gets
  stuck, prefer hints and guiding questions before giving away the full answer.

### Step 5 — End-of-Stage Quiz
- This step triggers only at the **end of a full stage** (not after every sub-task).
- Prepare a short quiz (mix of conceptual + code-reading/debugging questions) covering everything
  built/learned in that stage, to reinforce retention.
- Grade/discuss answers with the human; revisit weak spots briefly.

**Never skip Step 1 or Step 2.** Do not silently write large chunks of implementation code before
these steps have happened for the corresponding piece of work.

---

## 2. Standing Global Constraints (from IDE/system rules — repeated here for emphasis)

- Do **not** implement unit tests, documentation, or code checks/verification via terminal
  UNLESS the human explicitly asks for it.
- Do **not** run compilation/build checks via terminal until the human explicitly asks.
- Always give rationale before invoking a tool.
- Use `get_project_tree` / `list_files_in_folder` before diving into file operations; avoid
  `directory_tree`-style recursive dumps at project root.
- Prefer batched reads (`get_multiple_files_text`) over sequential single-file reads when
  multiple files are relevant.

---

## 3. Documentation & Repo Conventions

- `AGENTS.md` (this file) — process + running summary of confirmed architecture decisions.
  Update Section 0 whenever a cross-cutting decision changes.
- `docs/decisions/` — one decision-spec markdown file per sub-task, per the Step 3 convention above.
- `docs/architecture.md` — (to be created) living high-level architecture doc/diagram description,
  updated as stages progress — separate from the per-task decision specs.
- Code lives under a proper package (e.g. `parking_assistant/`) rather than loose scripts at root;
  exact layout to be proposed and confirmed with the human as part of Stage 1 / Step 1 planning
  (this is itself subject to the 5-step loop).

---

## 4. Stage Tracker

- [x] **Stage 1**: RAG system + chatbot, vector DB, interactive info/reservation collection,
      guardrails (PII filtering), evaluation (performance + accuracy).
- [x] **Stage 2**: Admin approval agent (LangChain summary + email notify + approval token).
- [x] **Stage 3**: MCP server (FastAPI, API-key protected, writes confirmed reservations to file).
- [x] **Stage 4**: LangGraph orchestration of the full pipeline (chat → admin approval → MCP write).
      Single `StateGraph` in `graph/pipeline.py`; interrupt-driven; conversation memory via chat
      history; real admin approval through `admin_app.py`; `service.py` reduced to a thin wrapper;
      `api.py` removed (Streamlit-only: `app.py` for chat, `admin_app.py` for approvals).

Update checkboxes and add brief notes as stages complete.
