# Data Ingestion Scripts

## Setup (Run Once)

```bash
# 1. Seed SQL database
python data_ingestions/seed_dynamic_db.py

# 2. Ingest static docs to Milvus
python data_ingestions/ingest_static_to_milvus.py
```

## Reset

```bash
rm parking_assistant.db milvus_data/ -rf
# Then re-run setup commands above
```

## What's Created

- `parking_assistant.db` - SQLite with 8 locations, prices, hours, availability
- `milvus_data/` - Milvus Lite with 68 embedded chunks across 4 collections

## Troubleshooting

**Missing `pkg_resources`**: Fixed via pinned `setuptools==75.6.0` in requirements.txt  
**Marshmallow error**: Fixed via pinned `marshmallow==3.23.1`  
**Directory not exists**: Auto-created by `vector_store.py` now
