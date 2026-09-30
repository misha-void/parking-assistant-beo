# Stage 1 Complete - Summary & Quiz

## ✅ What Was Implemented

### Task 1: Environment & Skeleton
- Python 3.12 venv with LangChain/LangGraph
- Dual interface: Streamlit + FastAPI
- Pluggable LLM factory (OpenAI)

### Task 2: Data Layer
- **Vector DB**: Milvus Lite with 4 collections (68 chunks embedded)
  - parking_zone_rules
  - parking_pricing
  - parking_locations_info
  - parking_faq
- **SQL DB**: SQLite with 5 tables
  - ParkingLocation (8 locations)
  - PriceRule (5 zones)
  - WorkingHours (33 records)
  - Availability (24 snapshots)
  - **Reservation** (new - for bookings)

### Task 3: RAG Chain + Reservations
- **Intent Classification**: 5 intents (general_info, pricing_hours, availability_search, book_spot, mixed)
- **Retriever**: Queries vector DB, SQL, or hybrid based on intent
- **RAG Chain**: Orchestrates intent → retrieval → response
- **Slot Collector**: Interactive conversation to collect:
  - Name, surname, car number
  - Location preference → matched location_id
  - Start/end dates (natural language parsing)
- **Reservation Flow**: Saves to DB with status=PENDING_APPROVAL, shows confirmation message

### Task 4: Guardrails
- **Microsoft Presidio** integration
- PII detection: names, emails, phones, credit cards, etc.
- Filters retrieved context before sending to LLM
- Validates user inputs (with allowed entities during reservation)

### Task 5: Evaluation
- **Performance metrics**: mean, median, p95, p99 latency
- **Accuracy metrics**: 
  - Intent classification accuracy
  - Keyword recall (expected keywords in responses)
  - Source routing accuracy (correct data source used)
- Test dataset: 8 diverse queries
- Evaluation script: `python scripts/run_evaluation.py`

---

## 📊 Architecture Overview

```
User Query
    ↓
Intent Classifier (LLM)
    ↓
┌───────────────┬────────────────┬──────────────┐
│ general_info  │ pricing_hours  │ availability │
│               │                │    _search   │
│ Vector DB     │ Hybrid:        │ SQL:         │
│ (Milvus)      │ Vector + SQL   │ Locations +  │
│               │                │ Availability │
└───────────────┴────────────────┴──────────────┘
    ↓
Guardrails (Presidio PII filter)
    ↓
RAG Chain (context + query → LLM)
    ↓
Response to User

Special case: book_spot intent
    ↓
Slot Collector (conversation state machine)
    ↓
Save Reservation (SQL) → Confirmation message
```

---

## 🧪 How to Test

### 1. Run Streamlit UI
```bash
streamlit run app.py
```

Try these queries:
- "What are the parking rules in zone A?" (general_info → vector DB)
- "How much does red zone cost?" (pricing_hours → hybrid)
- "Show me available garages" (availability_search → SQL)
- "I want to book a spot" (book_spot → slot collector)

### 2. Run Evaluation
```bash
python scripts/run_evaluation.py
```

Expected output:
- Intent accuracy: ~87.5% (7/8 correct)
- Keyword recall: ~75%+
- Latency: <2000ms for most queries

### 3. Run Tests
```bash
pytest tests/ -v
```

Tests cover:
- Vector storage (search, collections exist)
- Database (models, queries, joins)
- Guardrails (PII detection, anonymization)
- Reservation (slot filling, validation)

---

## 📝 Key Files to Review

**Core RAG:**
- `parking_assistant/rag/chain.py` - Main orchestration
- `parking_assistant/rag/intent_classifier.py` - Intent detection
- `parking_assistant/rag/retriever.py` - Multi-source retrieval

**Reservation:**
- `parking_assistant/graph/slot_collector.py` - Slot-filling logic
- `parking_assistant/db/models.py` - Reservation model (line 121+)
- `parking_assistant/service.py` - State management

**Guardrails:**
- `parking_assistant/guardrails/pii_filter.py` - Presidio wrapper

**Evaluation:**
- `parking_assistant/evaluation/metrics.py` - All metrics
- `scripts/run_evaluation.py` - Runner script

---

## 🎓 Stage 1 Quiz

### Question 1: Intent Classification
**Why do we use an LLM for intent classification instead of keyword matching?**

<details>
<summary>Answer</summary>
LLM understands semantic meaning, not just keywords. Example:
- "Where can I leave my car?" → availability_search (no keyword "parking" or "available")
- "Book me a space" → book_spot (no keyword "reservation")

Keyword matching would fail on paraphrases, synonyms, and natural language variations.
</details>

### Question 2: Data Split Strategy
**Why do we store zone rules in Milvus (vector DB) but price rules in SQLite?**

<details>
<summary>Answer</summary>
- **Zone rules** = unstructured text (policies, explanations) → best for semantic search ("What are zone A rules?" → retrieve relevant text chunks)
- **Price rules** = structured data (hourly_rate_rsd, max_duration_minutes) → best for exact queries and filtering ("Show me zones under 100 RSD/hour" → SQL WHERE clause)

Hybrid queries get best of both: semantic understanding + precise filtering.
</details>

### Question 3: Guardrails
**What problem do guardrails solve in our RAG system?**

<details>
<summary>Answer</summary>
Prevent **PII leakage** from vector DB. Example scenario:

1. Someone embeds a document containing: "Contact John Smith at john@example.com for zone A permits"
2. User asks: "How do I get a zone A permit?"
3. Without guardrails: LLM response includes John's email → privacy violation
4. With guardrails: Email replaced with [EMAIL] before reaching LLM

Also validates user inputs to warn if they accidentally share sensitive info.
</details>

### Question 4: Slot Collection
**Why do we use a conversation state machine for reservations instead of asking for all info in one prompt?**

<details>
<summary>Answer</summary>
**User experience**: Easier to answer one question at a time than fill a long form.

**Validation per slot**: Can immediately validate car number format, parse dates, match locations.

**Error recovery**: If user makes a mistake, can correct one field without re-entering everything.

**LLM token efficiency**: Shorter prompts per turn = faster + cheaper.

Trade-off: Requires session state management (we use in-memory dict with session_id).
</details>

### Question 5: Evaluation Metrics
**We measure "keyword recall" (% of expected keywords found in responses). Why not use semantic similarity instead?**

<details>
<summary>Answer</summary>
**Simplicity**: Keyword recall is easy to compute, no extra LLM calls or embedding comparisons.

**Interpretability**: "Response contains 3/4 expected keywords" is clear. Semantic similarity score of 0.82 needs context.

**Speed**: No inference overhead, can evaluate 100s of queries quickly.

**Good enough for Stage 1**: More sophisticated metrics (BLEU, ROUGE, LLM-as-judge) can be added in Stage 2 if needed.

Trade-off: Misses paraphrases ("vehicle" vs "car"), but catches major missing content.
</details>

---

## 🚀 Next Steps (Stage 2+)

Potential improvements:
- **LangGraph state machine**: Replace simple slot collector with proper graph (nodes for each state, conditional edges)
- **Text-to-SQL**: Let LLM generate SQL for complex availability queries
- **Multi-language**: Add Serbian language support
- **Admin UI**: Human approval workflow for reservations
- **Streaming responses**: Use LangChain streaming for real-time response tokens
- **Advanced evaluation**: RAGAS framework, LLM-as-judge for response quality

---

## 📦 Dependencies Installed

All in `requirements.txt`:
```
langchain==0.3.14
langgraph==0.2.60
langchain-openai==0.2.14
pymilvus==2.4.9
sqlalchemy==2.0.36
presidio-analyzer==2.2.355
presidio-anonymizer==2.2.355
fastapi==0.115.6
streamlit==1.41.1
pytest==8.4.0
```

Plus compatibility pins:
- setuptools==75.6.0 (for pkg_resources)
- marshmallow==3.23.1 (for environs)

---

**Stage 1 is complete!** 🎉

All required functionality implemented, tested, and documented. Ready for you to review and test the full system.
