# ✈ British Airways ML Pipeline

![CI](https://github.com/sukhmansaran/airline-analytics-ml-system/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Node](https://img.shields.io/badge/Node.js-18%2B-green?logo=node.js&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

> Production-style ML system that scrapes airline reviews, extracts sentiment signals, and predicts booking conversion — exposed via a scalable FastAPI backend and interactive React dashboard.

---

## Problem & Impact

Airlines operate on thin margins where customer acquisition and retention are critical. Two high-value problems this project addresses:

- **Sentiment from reviews** — identify dissatisfied customers at scale from unstructured text, enabling targeted service improvements
- **Booking completion prediction** — predict which customers are likely to complete a booking, enabling smarter marketing spend and conversion optimisation

This simulates a real data science workflow: raw data → feature engineering → model tuning → production API → live dashboard. The same pipeline structure is used in revenue management and CRM systems at airlines.

---

## Highlights

- 🔍 **Web scraping** — Selenium + BeautifulSoup scraper for airline, seat, and lounge reviews from Skytrax
- 🧠 **Dual sentiment analysis** — VADER and RoBERTa scores averaged for robust Positive / Neutral / Negative classification
- 🛠 **Feature engineering** — 7 derived features on top of raw inputs (`extras_count`, `lead_per_duration`, `is_long_haul`, `is_weekend_flight`, etc.)
- 📊 **Booking prediction** — XGBoost, LightGBM, and CatBoost tuned with Optuna; best model auto-selected by ROC-AUC
- ⚡ **FastAPI backend** — Pydantic validation, auto Swagger docs, proper error handling
- 🖥 **React dashboard** — booking predictor, sentiment analyser with radar chart, live model metrics
- 🐳 **Production-ready** — Docker Compose, GitHub Actions CI/CD, nginx reverse proxy

---

## Architecture

Data flows from scraping → preprocessing → training → model artifacts → API inference → frontend visualisation.

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│  React UI   │────▶│              FastAPI Backend              │
│  (Vite)     │     │                                          │
│  :3000      │     │  POST /predict/booking                   │
└─────────────┘     │    └─▶ XGBoost / LightGBM / CatBoost    │
                    │                                          │
                    │  POST /sentiment/analyse                 │
                    │    └─▶ VADER + RoBERTa (averaged)        │
                    │                                          │
                    │  GET  /metrics/   GET /health            │
                    └──────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴──────────────────────┐
                    │           Training Pipeline               │
                    │                                          │
                    │  Scraper → CSV → Feature Engineering     │
                    │    → Optuna Tuning → model artifacts     │
                    └──────────────────────────────────────────┘
```

---

## Quick Start (Docker)

```bash
git clone https://github.com/sukhmansaran/airline-analytics-ml-system.git
cd airline-analytics-ml-system
docker-compose up --build
```

| | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API | http://localhost:8000 |
| Swagger | http://localhost:8000/docs |

> Sample data is included in `data/sample/` — no scraping needed to try it out.

---

## Model Performance

Trained on 50,000 customer booking records. Models tuned with Optuna (30 trials each), evaluated on a 30% held-out test set. ROC-AUC computed from predicted probabilities.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** ★ | 66.8% | 28.7% | 78.6% | 42.0% | **0.717** |
| LightGBM | 68.1% | 29.4% | 74.2% | 42.1% | 0.709 |
| CatBoost | 67.5% | 28.9% | 75.8% | 41.8% | 0.705 |

Baseline (raw features, default XGBoost params, no engineering): ROC-AUC ~0.67. Feature engineering + Optuna tuning improved this to 0.717 — a ~4–5 point gain.

**On the precision/recall trade-off:** the dataset is heavily imbalanced — booking completion is the minority class. High recall is intentionally prioritised: it's better to flag more potential completions (even with false positives) than to miss real ones. Class imbalance is handled via `scale_pos_weight` (XGBoost, LightGBM) and `auto_class_weights="Balanced"` (CatBoost).

**Top predictive features:**

| Feature | Description |
|---|---|
| `purchase_lead` | Days between booking and flight — strong signal for intent |
| `length_of_stay` | Longer stays correlate with higher completion |
| `flight_duration` | Long-haul bookings behave differently |
| `extras_count` | Customers who add extras are more committed |
| `lead_per_duration` | How far ahead booked relative to flight length |
| `booking_origin` | Origin country affects completion rates significantly |
| `is_long_haul` | Engineered flag — flight > 6h |

---

## Key Design Decisions

- **Why gradient boosting over deep learning?** Tabular data consistently performs better with tree-based models. Faster training, better interpretability, and no need for large datasets to generalise.
- **Why average VADER + RoBERTa?** VADER captures rule-based sentiment well for short, structured reviews. RoBERTa captures contextual semantics. Averaging the two improves robustness over either alone.
- **Why Optuna over GridSearchCV?** Optuna uses TPE (Bayesian optimisation) — it learns which hyperparameter regions are promising and focuses trials there. GridSearch blindly exhausts a fixed grid, which is inefficient for large search spaces.
- **Why FastAPI?** Async support, automatic request validation via Pydantic, and built-in Swagger docs with zero extra config.
- **Why Docker Compose?** Ensures reproducibility across environments and makes the full stack (API + frontend + nginx) runnable with a single command.

---

## Performance

- API response time: ~50–120ms (CPU inference, model already loaded)
- First RoBERTa call: ~1–2s (model loaded into memory at startup via `warm_up()`, cached for all subsequent requests)
- Training time: ~15–30 min on full 50k dataset with 30 Optuna trials per model

---

## API Reference

Full examples with curl + Python in [`docs/api_examples.md`](docs/api_examples.md).

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API status + model loaded flag |
| `POST` | `/predict/booking` | Predict booking completion |
| `POST` | `/sentiment/analyse` | Dual-model sentiment analysis |
| `GET` | `/metrics/` | Model evaluation metrics |
| `GET` | `/docs` | Swagger UI |

**Predict booking**
```bash
curl -X POST http://localhost:8000/predict/booking \
  -H "Content-Type: application/json" \
  -d '{"num_passengers":1,"sales_channel":"Internet","trip_type":"RoundTrip",
       "purchase_lead":30,"length_of_stay":7,"flight_hour":10,"flight_day":"Mon",
       "route":"LHRBKK","booking_origin":"United Kingdom","wants_extra_baggage":0,
       "wants_preferred_seat":0,"wants_in_flight_meals":0,"flight_duration":11.5}'
```
```json
{"booking_complete": 0, "probability": 0.7142, "model_used": "XGBoost"}
```

**Analyse sentiment**
```bash
curl -X POST http://localhost:8000/sentiment/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "The crew was fantastic and the flight was very smooth."}'
```
```json
{"avg_pos": 0.711, "avg_neg": 0.011, "avg_neu": 0.279, "pred_result": "Positive"}
```

---

## Project Structure

```
british-airways/
├── api/                        # FastAPI app
│   └── routes/
│       ├── predict.py          # POST /predict/booking
│       ├── sentiment.py        # POST /sentiment/analyse
│       └── metrics.py          # GET  /metrics/
├── frontend/                   # React + Vite dashboard
│   └── src/pages/              # Predict, Sentiment, Metrics, Health
├── nlp/
│   └── sentiment.py            # VADER + RoBERTa inference
├── scraping/                   # Selenium scraper + text cleaner
├── training/
│   ├── feature_engineering.py  # 7 derived features
│   ├── preprocess.py           # Label encoding + scaling
│   └── train.py                # Optuna tuning — XGBoost, LightGBM, CatBoost
├── scripts/
│   └── run_scraper.py          # Scraping entry point
├── data/sample/                # Sample CSVs for quick start
├── docs/api_examples.md        # curl + Python request examples
├── Makefile                    # make run / train / scrape / docker
├── Dockerfile
└── docker-compose.yml
```

---

## Local Setup

### Prerequisites
- Python 3.10+ · Node.js 18+ · Chrome _(scraping only)_

### 1 — Environment

```bash
python -m venv venv

# Windows
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2 — Train the model

```bash
# Quick start with sample data
make train-sample

# Full dataset (place customer_booking.csv in data/ first)
make train
```

### 3 — Run API + frontend

```bash
# Terminal 1
make run

# Terminal 2
cd frontend && npm install && npm run dev
```

### 4 — Scrape reviews _(optional)_

```bash
make scrape
```

---

## Future Improvements

- **Cloud deployment** — containerised deploy to AWS ECS or GCP Cloud Run with autoscaling
- **Model monitoring** — drift detection on input distributions and prediction confidence over time
- **SMOTE oversampling** — address class imbalance at the data level rather than just via class weights
- **Streaming sentiment** — Kafka consumer for real-time review ingestion and live sentiment scoring
- **Threshold tuning** — optimise decision threshold per business objective (precision vs recall trade-off)
- **Explainability** — SHAP values per prediction exposed via API for interpretable outputs

---

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Push / PR | Lint (ruff), API import check, frontend build, Docker smoke test |
| `deploy.yml` | Push to `main` / tag | Build + push image to Docker Hub |

Add `DOCKER_USERNAME` and `DOCKER_PASSWORD` to GitHub Secrets to enable deploy.

---

## Resume Bullet

> Built a production-style ML pipeline for airline analytics — combining NLP sentiment analysis (VADER + RoBERTa) and gradient boosting booking prediction (XGBoost ROC-AUC 0.717), tuned with Optuna, deployed via FastAPI with a React dashboard, Dockerized CI/CD, and 7 engineered features that improved baseline AUC by ~5 points.

---

## Troubleshooting

- **`uvicorn` not found** — use `python -m uvicorn api.main:app --reload` or `make run`
- **`ModuleNotFoundError`** — always run from the project root, never from inside a subfolder
- **503 on prediction** — model not loaded; run `make train-sample` first and check `/health`
- **RoBERTa slow on first use** — downloads ~500MB to `~/.cache/huggingface` once; fast after that
- **PowerShell `&&` not supported** — use `;` instead, or use the Makefile targets
- **CatBoost temp dir error** — ensure `%TEMP%` is writable; training creates `catboost/` subdirs there automatically
