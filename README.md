# Parking Lot Assistant - AI Fast Track Project

This project is a learning exercise in AI Engineering, focusing on RAG, evaluation, and guardrails.

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies (already done during scaffold)
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 3. Run the Application

**Option A: Streamlit UI (interactive demo)**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**Option B: FastAPI Backend (for programmatic access / future Telegram bot)**
```bash
uvicorn api:app --reload
```
Then:
- API docs: http://localhost:8000/docs
- Test with curl:
  ```bash
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "Hello, what are your working hours?"}'
  ```

## Project Structure

```
parking_assistant/          # Main package
├── __init__.py
├── llm_factory.py         # LLM/embedding model instantiation
├── service.py             # Shared chat service logic
├── rag/                   # (Future) Retrieval logic, Milvus adapter
├── guardrails/            # (Future) Presidio PII filtering
├── db/                    # (Future) SQLAlchemy models
├── graph/                 # (Future) LangGraph state machine
└── ui/                    # (Future) Streamlit-specific helpers

app.py                     # Streamlit interface
api.py                     # FastAPI interface
requirements.txt           # Dependencies
.env.example               # Environment variable template
docs/                      # Documentation & decision logs
```

## Current Status

**Stage 1 - Task 1: Environment & Skeleton** ✅
- Python 3.12 venv with LangChain/LangGraph/FastAPI/Streamlit
- Dual interface (Streamlit + FastAPI)
- Trivial LLM integration (no RAG/guardrails yet)

Next: RAG system, vector DB, guardrails, evaluation...

## Documentation

See `docs/decisions/` for detailed decision logs per task.
See `AGENTS.md` for development process & AI assistant instructions.
