# Quick Start Guide - Stage 1 Complete System

## Prerequisites

- Python 3.12 installed
- OpenAI API key

## Setup (5 minutes)

### 1. Install Dependencies

```bash
# Make sure venv is active
source .venv/bin/activate

# Install all packages (including Presidio for guardrails)
pip install -r requirements.txt

# Download spaCy model for NLP (required by Presidio)
python -m spacy download en_core_web_sm
```

### 2. Configure API Key

```bash
# Already done if you have .env with OPENAI_API_KEY
# If not:
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 3. Initialize Databases

```bash
# This was already run, but if you need to reset:
rm parking_assistant.db
rm -rf milvus_data/

# Recreate:
python data_ingestions/seed_dynamic_db.py
python data_ingestions/ingest_static_to_milvus.py
```

## Run the Application

### Option 1: Streamlit UI (Recommended)

```bash
streamlit run app.py
```

Browser opens to http://localhost:8501

**Try these queries:**
1. "What are the parking rules in zone A?"
2. "How much does parking cost in the red zone?"
3. "Show me available garages"
4. "I want to book a parking spot"

For #4, the bot will guide you through:
- Name, surname
- Car number (e.g., BG123AB)
- Location selection
- Start/end dates

### Record confirmed reservations (optional)

Run the MCP server in a separate terminal so confirmed bookings are written to file:

```bash
uvicorn mcp_server.main:app --port 8001
```

The chat works without it; the recording step just logs if the server is unreachable.

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

Expected: 10+ tests pass (vector storage, database, guardrails, reservation)

### Run Evaluation

```bash
python scripts/run_evaluation.py
```

Expected output:
```
📊 RAG SYSTEM EVALUATION REPORT
📈 Performance Metrics (8 queries):
  Mean latency:      1500 ms
  P95 latency:       2000 ms
🎯 Accuracy Metrics:
  Intent classification: 87.5%
  Keyword recall:        75.0%
  Source routing:        87.5%
✅ No failures!
```

## Common Issues

### "No module named 'presidio_analyzer'"
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_sm
```

### "No module named 'pkg_resources'"
Already fixed with `setuptools==75.6.0` pin in requirements.txt

### Streamlit shows "Thinking..." forever
Check console for errors. Common: missing OPENAI_API_KEY or Milvus connection issue.

### Reservation slot collector repeats questions
Session state might be lost. Refresh browser (Streamlit) or check server logs.

## What to Expect

### Information Queries
Bot retrieves from vector DB or SQL, applies guardrails, generates contextual response.

### Availability Queries
Bot queries SQL database, formats results as list.

### Reservation Flow
Multi-turn conversation:
1. Collects all required info
2. Shows confirmation summary
3. Saves to database with status=PENDING_APPROVAL
4. Returns confirmation message with ID

### Guardrails
If retrieved context contains PII (names, emails, phones), it's filtered before reaching the LLM.

## Next Steps

1. **Test thoroughly** - try edge cases, wrong inputs, cancellations
2. **Review code** - check RAG chain logic, slot collector state machine
3. **Run evaluation** - see metrics, identify weak spots
4. **Answer quiz** - `docs/STAGE1_COMPLETE.md` has 5 questions to test understanding

## Files Created in This Stage

**Core Logic:**
- `parking_assistant/rag/` - Intent classifier, retriever, RAG chain
- `parking_assistant/graph/slot_collector.py` - Reservation conversation
- `parking_assistant/guardrails/pii_filter.py` - Presidio integration
- `parking_assistant/evaluation/metrics.py` - Performance & accuracy metrics
- `parking_assistant/service.py` - Main orchestration (updated)

**Data:**
- `parking_assistant/db/models.py` - Added Reservation model
- `parking_assistant.db` - SQLite with 5 tables (including reservations)
- `milvus_data/` - 4 collections, 68 embedded chunks

**Tests:**
- `tests/test_guardrails.py` - PII detection tests
- `tests/test_reservation.py` - Slot collection tests
- (Existing: test_vector_storage.py, test_database.py)

**Docs:**
- `README.md` - Complete project overview
- `docs/STAGE1_COMPLETE.md` - Summary + quiz
- This file

---

**Stage 1 is complete and ready for testing!** 🎉
