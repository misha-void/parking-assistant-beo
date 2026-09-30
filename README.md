"""
Parking Assistant - Belgrade
AI chatbot for parking information and reservations in Belgrade, Serbia.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# Download spaCy model for guardrails
python -m spacy download en_core_web_sm
```

### 2. Setup Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Initialize Data
```bash
# Seed SQL database
python data_ingestions/seed_dynamic_db.py

# Ingest static docs to vector DB
python data_ingestions/ingest_static_to_milvus.py
```

### 4. Run Application
```bash
# Streamlit UI (recommended for testing)
streamlit run app.py

# Or FastAPI backend
uvicorn api:app --reload
```

## Features

✅ **RAG System**
- Intent classification (5 intents)
- Hybrid retrieval (Milvus + SQLite)
- Context-aware responses

✅ **Reservation System**
- Interactive slot-filling conversation
- Collects: name, surname, car number, location, dates
- Human-in-loop approval simulation

✅ **Guardrails**
- PII detection using Microsoft Presidio
- Filters sensitive data from retrieved context
- Validates user inputs

✅ **Evaluation**
- Performance metrics (latency p50/p95/p99)
- Accuracy metrics (intent, keyword recall, source routing)
- Test dataset with 8 queries

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_guardrails.py -v

# Run evaluation
python scripts/run_evaluation.py
```

## Project Structure

```
parking_assistant/
├── rag/                    # RAG components
│   ├── intent_classifier.py
│   ├── retriever.py
│   └── chain.py
├── graph/                  # Reservation slot collector
│   └── slot_collector.py
├── guardrails/             # PII filtering
│   └── pii_filter.py
├── evaluation/             # Metrics & evaluation
│   └── metrics.py
├── db/                     # Database models
│   ├── models.py
│   └── database.py
└── service.py              # Main service layer

data/
├── static/                 # Markdown docs (vector DB)
└── parking_locations.json  # Parking locations (SQL)

data_ingestions/            # Setup scripts
tests/                      # Pytest test suite
scripts/                    # Evaluation runner
```

## Conversation Examples

### Information Query
```
User: What are the parking rules in zone A?
Bot: Zone A (Purple Zone) has a maximum parking time of 30 minutes with no extensions available...
```

### Availability Search
```
User: Show me available garages
Bot: Here are the available garages:
     • UŠĆE Shopping Center: 1,800/2,500 spots available
     • Delta City Garage: 620/800 spots available
     ...
```

### Reservation Flow
```
User: I want to book a spot
Bot: Great! I'll help you book a parking spot. What's your first name?
User: John
Bot: What's your surname?
User: Doe
Bot: What's your car license plate number?
User: BG123AB
Bot: Here are some available parking locations...
...
Bot: ✅ Reservation Submitted! Your reservation request (ID: #42) has been submitted for approval.
```

## Stage 1 Completion Status

✅ Task 1: Environment & skeleton  
✅ Task 2: Vector DB + SQL data layer  
✅ Task 3: RAG chain + reservation collector  
✅ Task 4: Guardrails (PII filtering)  
✅ Task 5: Evaluation metrics

**Next**: Stage 2 (LangGraph state machine, advanced reservation flow)

## Dependencies

Core:
- Python 3.12
- LangChain 0.3+
- LangGraph 0.2+
- OpenAI API

Data:
- Milvus Lite (vector DB)
- SQLite + SQLAlchemy

Guardrails:
- Microsoft Presidio

UI:
- Streamlit (chat interface)
- FastAPI (backend API)

## License

Learning project - MIT License
"""