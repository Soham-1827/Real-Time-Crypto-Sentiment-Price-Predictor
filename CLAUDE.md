# Real-Time Crypto ML Pipeline - Project Guide

## Project Overview

Building an end-to-end real-time machine learning pipeline for cryptocurrency price prediction with sentiment analysis. The system ingests live data, processes it into features, trains/updates models continuously, serves predictions via API, and monitors model health.

**Owner**: Soham (CS student at ASU, AI/ML intern at Bayer)
**Timeline**: 8 weeks
**Budget**: $0-10/month (strictly free tier focused)

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│ Binance WebSocket│     │  Reddit RSS     │
│ (live prices)   │     │  (sentiment)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│           Upstash Redis Streams         │
│  streams: raw:prices, raw:reddit        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│        Python Stream Processors         │
│  - Price features (RSI, MA, volatility) │
│  - Sentiment features (FinBERT scores)  │
│  - Feature joining by timestamp         │
└────────────────┬────────────────────────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
┌─────────────┐   ┌─────────────┐
│  Supabase   │   │   River     │
│  (Postgres) │   │  (Online    │
│  - features │   │   Learning) │
│  - predictions│ └──────┬──────┘
└──────┬──────┘          │
       │                 │
       ▼                 ▼
┌─────────────────────────────────────────┐
│              FastAPI Server             │
│  - /predict/{symbol}                    │
│  - /features/{symbol}                   │
│  - /health                              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│    Streamlit Dashboard (HF Spaces)      │
│  - Live predictions chart               │
│  - Model accuracy metrics               │
│  - Evidently drift reports              │
└─────────────────────────────────────────┘
```

---

## Tech Stack

**Budget**: $10/month maximum

| Component | Tool | Tier | Cost | Limits |
|-----------|------|------|------|--------|
| Message Queue | Upstash Redis | **Pay-as-you-go** | ~$2-3/mo | 10K commands/day free + $0.2 per 100K extra |
| Stream Processing | Python + asyncio + Quix Streams | Free | $0 | N/A |
| Database | Supabase (Postgres) | Free | $0 | 500MB storage, 2GB transfer |
| ML Training | LightGBM | Free | $0 | N/A |
| Online Learning | River | Free | $0 | N/A |
| Model Serving | FastAPI on Railway | **Pro** | ~$5/mo | No sleep, always-on |
| Experiment Tracking | MLflow on Dagshub | Free | $0 | Free hosted |
| Monitoring | Evidently AI | Free | $0 | N/A |
| Dashboard | Streamlit on HF Spaces | Free | $0 | 16GB RAM |

### $10/month Budget Allocation (Recommended)

| Service | Upgrade | Cost | Why Worth It |
|---------|---------|------|--------------|
| **Railway** | Pro plan | ~$5 | API stays awake 24/7, no cold starts, looks professional in demos |
| **Upstash Redis** | Pay-as-you-go | ~$2-3 | 50K+ commands/day, no throttling during market volatility |
| **Buffer** | Emergency reserve | ~$2-3 | Unexpected overages, testing spikes |

### Alternative Budget Splits

**Option A: Always-on API focus ($10)**
- Railway Pro: $5
- Upstash upgraded: $3
- Supabase (stay free): $0
- Reserve: $2

**Option B: More data throughput ($10)**
- Railway (stay free, accept cold starts): $0
- Upstash Pro (200K commands/day): $7
- Reserve: $3

**Option C: GPU for FinBERT inference ($10)**
- Modal: Use $10 of free credits for GPU inference
- Everything else: Free tier
- Best if you want heavier NLP models

### When to Upgrade (Don't Pay Until You Need It)

| Phase | What to use | When to upgrade |
|-------|-------------|-----------------|
| Phase 1-3 | All free tiers | Stay free while building |
| Phase 4-5 | All free tiers | Stay free while training |
| Phase 6 | Upgrade Railway to Pro | When deploying API (need always-on) |
| Phase 7 | Upgrade Upstash if hitting limits | When dashboard drives more traffic |

**Pro tip**: Start 100% free, upgrade Railway only when you're ready to show the project to recruiters/interviewers. That's when cold starts matter.

---

## Project Structure

```
crypto-ml-pipeline/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── binance_ws.py        # WebSocket price ingestion
│   │   ├── reddit_rss.py        # RSS feed ingestion
│   │   └── config.py            # API keys, URLs
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── price_features.py    # Technical indicators
│   │   ├── sentiment_features.py # NLP processing
│   │   ├── feature_joiner.py    # Combine features by timestamp
│   │   └── stream_consumer.py   # Redis Streams consumer base
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── redis_client.py      # Upstash connection
│   │   ├── supabase_client.py   # Postgres connection
│   │   └── feature_store.py     # Feature retrieval functions
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── prepare_data.py      # Training data preparation
│   │   ├── train_lightgbm.py    # Batch training script
│   │   ├── online_learner.py    # River online learning
│   │   └── evaluate.py          # Model evaluation metrics
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI application
│   │   ├── predictor.py         # Prediction logic
│   │   └── schemas.py           # Pydantic models
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── drift_detector.py    # Evidently reports
│       └── alerts.py            # Alert logic
│
├── dashboard/
│   ├── app.py                   # Streamlit dashboard
│   └── components/              # Dashboard components
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_backtest_analysis.ipynb
│
├── tests/
│   ├── test_features.py
│   ├── test_api.py
│   └── test_storage.py
│
├── scripts/
│   ├── run_ingestion.py         # Start all ingestion
│   ├── run_processing.py        # Start all processors
│   └── backfill_features.py     # Historical data backfill
│
├── docker-compose.yml           # Local dev environment
├── requirements.txt
├── .env.example
├── README.md
└── CLAUDE.md                    # This file
```

---

## Development Phases

### Phase 1: Data Ingestion (Current Phase)
**Status**: Not started
**Goal**: Get real-time data flowing into Redis Streams

Tasks:
- [ ] Set up Binance WebSocket connection for BTC, ETH price streams
- [ ] Set up Reddit RSS polling for r/cryptocurrency, r/wallstreetbets
- [ ] Create Upstash Redis account and database
- [ ] Implement producers that push to Redis Streams
- [ ] Test data flow end-to-end

Key files to create:
- `src/ingestion/binance_ws.py`
- `src/ingestion/reddit_rss.py`
- `src/storage/redis_client.py`

Success criteria:
- Both scripts running continuously
- Data visible in Upstash console
- No rate limit issues

### Phase 2: Stream Processing
**Status**: Not started
**Goal**: Transform raw data into ML features

Tasks:
- [ ] Build price feature processor (MA, RSI, volatility, momentum)
- [ ] Build sentiment processor (TextBlob initially, FinBERT later)
- [ ] Build feature joiner (align by timestamp)
- [ ] Implement Redis Streams consumer groups

Key files to create:
- `src/processing/price_features.py`
- `src/processing/sentiment_features.py`
- `src/processing/feature_joiner.py`

Success criteria:
- Combined feature vectors flowing to `features:combined` stream
- Features correctly aligned by timestamp

### Phase 3: Database & Feature Storage
**Status**: Not started
**Goal**: Persist features for training

Tasks:
- [ ] Set up Supabase project
- [ ] Design and create database schema
- [ ] Implement persistence consumer
- [ ] Build feature retrieval functions

Key files to create:
- `src/storage/supabase_client.py`
- `src/storage/feature_store.py`

Success criteria:
- 24+ hours of continuous data in database
- Feature retrieval functions working

### Phase 4: Model Training
**Status**: Not started
**Goal**: Train baseline model with experiment tracking

Tasks:
- [ ] Set up MLflow on Dagshub
- [ ] Prepare training data with proper time splits
- [ ] Train LightGBM baseline
- [ ] Log experiments and compare

Key files to create:
- `src/training/prepare_data.py`
- `src/training/train_lightgbm.py`
- `src/training/evaluate.py`

Success criteria:
- Model logged in MLflow with metrics
- Test accuracy above random baseline (>52%)

### Phase 5: Online Learning
**Status**: Not started
**Goal**: Add continuous model updates

Tasks:
- [ ] Learn River library basics
- [ ] Implement online learning loop
- [ ] Track rolling accuracy
- [ ] Optional: hybrid LightGBM + River approach

Key files to create:
- `src/training/online_learner.py`

Success criteria:
- Model updating with each new observation
- Rolling accuracy tracked and stable

### Phase 6: Model Serving
**Status**: Not started
**Goal**: Deploy prediction API

Tasks:
- [ ] Build FastAPI application
- [ ] Implement prediction endpoint
- [ ] Add prediction logging
- [ ] Deploy to Railway

Key files to create:
- `src/serving/app.py`
- `src/serving/predictor.py`
- `src/serving/schemas.py`

Success criteria:
- Live API returning predictions
- Predictions logged to database

### Phase 7: Dashboard & Monitoring
**Status**: Not started
**Goal**: Visualize and monitor

Tasks:
- [ ] Build Streamlit dashboard
- [ ] Add Evidently drift reports
- [ ] Deploy to HF Spaces
- [ ] Optional: set up alerts

Key files to create:
- `dashboard/app.py`
- `src/monitoring/drift_detector.py`

Success criteria:
- Live dashboard accessible publicly
- Drift reports generating

### Phase 8: Documentation & Polish
**Status**: Not started
**Goal**: Portfolio-ready project

Tasks:
- [ ] Write comprehensive README
- [ ] Create architecture diagram
- [ ] Add unit tests
- [ ] Clean up code structure

Success criteria:
- Project presentable to interviewers
- All demos working

---

## Key Technical Decisions

### Why Redis Streams over Kafka/Redpanda?
- Upstash offers generous free tier (10K commands/day)
- Simpler setup for single-developer project
- Sufficient for portfolio-scale throughput
- Can migrate to Kafka later if needed

### Why LightGBM + River hybrid?
- LightGBM: Fast, memory-efficient, great for tabular data
- River: Enables online learning without full retraining
- Hybrid gives best of both: stable base + adaptive updates

### Why not use a proper feature store (Feast)?
- Overkill for single-user project
- Adds infrastructure complexity
- SQLite/Postgres with good functions achieves same goal
- Can add Feast in Phase 9 if extending project

### Target variable design
- Predicting: `price_direction_1h` (binary: up/down)
- Why 1 hour: Long enough for signal, short enough for feedback loop
- Alternative targets to explore: `price_change_magnitude`, `volatility_spike`

---

## Environment Variables

```env
# Redis (Upstash)
UPSTASH_REDIS_URL=redis://...
UPSTASH_REDIS_TOKEN=...

# Supabase
SUPABASE_URL=https://...
SUPABASE_KEY=...

# MLflow (Dagshub)
MLFLOW_TRACKING_URI=https://dagshub.com/...
MLFLOW_TRACKING_USERNAME=...
MLFLOW_TRACKING_PASSWORD=...

# Optional: Binance (not needed for public streams)
BINANCE_API_KEY=
BINANCE_SECRET=
```

---

## Data Schemas

### Redis Streams

**raw:prices**
```json
{
  "symbol": "BTCUSDT",
  "price": "43250.50",
  "volume": "1234.56",
  "timestamp": "1704672000000"
}
```

**raw:reddit**
```json
{
  "subreddit": "cryptocurrency",
  "title": "Bitcoin hits new high...",
  "score": 1234,
  "num_comments": 567,
  "created_utc": "1704672000",
  "url": "https://..."
}
```

**features:combined**
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "1704672000",
  "price": 43250.50,
  "ma_5m": 43200.25,
  "ma_15m": 43150.00,
  "rsi_14": 65.5,
  "volatility_1h": 0.023,
  "momentum": 0.015,
  "sentiment_score": 0.72,
  "sentiment_volume": 45,
  "reddit_mentions": 123
}
```

### Supabase Tables

**features**
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| symbol | text | Trading pair |
| timestamp | timestamptz | Feature timestamp |
| price | float | Current price |
| ma_5m | float | 5-min moving average |
| ma_15m | float | 15-min moving average |
| rsi_14 | float | 14-period RSI |
| volatility_1h | float | 1-hour volatility |
| momentum | float | Price momentum |
| sentiment_score | float | Aggregated sentiment |
| sentiment_volume | int | Number of posts |
| created_at | timestamptz | Insert timestamp |

**predictions**
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary key |
| symbol | text | Trading pair |
| timestamp | timestamptz | Prediction timestamp |
| prediction | int | 0 or 1 |
| confidence | float | Model confidence |
| actual | int | Actual outcome (filled later) |
| model_version | text | Model identifier |
| features_used | jsonb | Feature snapshot |
| created_at | timestamptz | Insert timestamp |

---

## Common Issues & Solutions

### Rate Limiting
- **Upstash**: Batch writes using XADD with multiple fields
- **Reddit RSS**: Poll every 5 minutes, not more frequently
- **Supabase**: Batch inserts (50-100 records per call)

### Time Alignment
- Always use UTC timestamps
- Align to minute boundaries for joining
- Handle late-arriving data with small buffer window

### Look-ahead Bias
- Never use future data in features
- Shift features: `df['ma'] = df['price'].shift(1).rolling(5).mean()`
- Use TimeSeriesSplit, never random shuffle

### Memory Management
- Process in small batches, not full DataFrames
- Clear old data from Redis periodically
- Use generators for large data iteration

---

## Resources & Documentation

- [Binance WebSocket API](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
- [Redis Streams Tutorial](https://redis.io/docs/data-types/streams-tutorial/)
- [Upstash Python SDK](https://github.com/upstash/redis-py)
- [River Documentation](https://riverml.xyz/)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

## Commands Reference

```bash
# Start ingestion (run in separate terminals or use tmux)
python scripts/run_ingestion.py --source binance
python scripts/run_ingestion.py --source reddit

# Start processing
python scripts/run_processing.py

# Train model
python -m src.training.train_lightgbm

# Run API locally
uvicorn src.serving.app:app --reload

# Run dashboard locally
streamlit run dashboard/app.py

# Generate drift report
python -m src.monitoring.drift_detector

# Run tests
pytest tests/ -v
```

---

## Notes for Claude Code

When helping with this project:

1. **Current phase**: Check the "Development Phases" section for current status
2. **Budget constraint**: Strictly free tier - no paid services
3. **Learning focus**: Explain concepts, don't just write code
4. **Code style**: 
   - Type hints everywhere
   - Docstrings for functions
   - Clear variable names
   - Modular, testable functions
5. **Testing**: Suggest tests for critical functions
6. **Error handling**: Robust error handling for network/API calls
7. **Logging**: Use proper logging, not print statements

When suggesting next steps:
- Reference specific files to create/modify
- Explain why certain approaches are chosen
- Point out potential pitfalls
- Suggest how to verify each step works

---

## Progress Log

*(Update this as you complete tasks)*

| Date | Phase | What was done | Notes |
|------|-------|---------------|-------|
| 2026-01-11 | Phase 1 | ✅ Set up project structure (folders, requirements.txt, .env) | All core directories created |
| 2026-01-11 | Phase 1 | ✅ Created Upstash Redis account and connected successfully | Free tier, REST API |
| 2026-01-11 | Phase 1 | ✅ Built Redis client helper (src/storage/redis_client.py) | Reusable connection module |
| 2026-01-11 | Phase 1 | ✅ Built Binance WebSocket ingestion (src/ingestion/binance_ws.py) | Live BTC price streaming working! |
| 2026-01-11 | Phase 1 | ✅ Tested end-to-end data flow: Binance → Script → Redis | Data flowing successfully |

---

## Questions to Explore

- How to handle market closed hours (crypto is 24/7, but sentiment varies)?
- What's the optimal feature lag for prediction?
- How to detect and handle concept drift in financial data?
- Should predictions be continuous (regression) or categorical (up/down/neutral)?
- How to backtest without look-ahead bias?