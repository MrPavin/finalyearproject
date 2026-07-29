# Context-Aware Multilingual Hate Speech Detection — Backend API

> **FastAPI · Python 3.12 · XLM-RoBERTa**

A production-ready REST API backend for detecting hate speech in multilingual text using a fine-tuned XLM-RoBERTa transformer model.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Testing](#testing)
- [Adding the Model](#adding-the-model)
- [Roadmap](#roadmap)

---

## Project Structure

```
backend/
├── app.py                          # FastAPI application factory & entry-point
├── predict.py                      # CLI / programmatic prediction helper
├── config.py                       # Pydantic Settings (reads from .env)
├── requirements.txt                # Pinned Python dependencies
├── .env                            # Environment variables (not committed)
├── .gitignore
│
├── models/
│   └── xlm_roberta_hate_model/     # Model weights directory (see below)
│
├── routes/
│   ├── __init__.py
│   └── prediction.py               # POST /predict, POST /predict/batch, GET /predict/model/info
│
├── services/
│   ├── __init__.py
│   └── model_service.py            # ModelService singleton (load / predict)
│
├── schemas/
│   ├── __init__.py
│   └── prediction_schema.py        # Pydantic request / response schemas
│
├── utils/
│   ├── __init__.py
│   └── helper.py                   # Sanitisation, timing, response builders
│
├── static/                         # Static file serving (mounted at /static)
├── logs/                           # Rotating log files (auto-created)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Pytest fixtures (TestClient)
    ├── test_health.py               # Health & root endpoint tests
    ├── test_prediction.py           # Prediction endpoint tests
    └── test_utils.py                # Unit tests for helper utilities
```

---

## Features

| Feature | Status |
|---|---|
| FastAPI with async support | ✅ |
| CORS middleware | ✅ |
| Swagger UI (`/docs`) & ReDoc (`/redoc`) | ✅ |
| API versioning (`/api/v1`) | ✅ |
| Health check endpoint (`/health`) | ✅ |
| Rotating file logging | ✅ |
| Global exception handlers | ✅ |
| Request ID tracing (`X-Request-ID` header) | ✅ |
| Pydantic v2 validation | ✅ |
| Modular project structure | ✅ |
| Unit & integration tests (pytest) | ✅ |
| XLM-RoBERTa inference | 🔜 Pending model weights |
| Batch prediction | 🔜 Pending model implementation |

---

## Prerequisites

- **Python 3.12**
- `pip` or `uv`

---

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd backend

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Server

```bash
# Development (auto-reload)
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API base** → http://localhost:8000
- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc
- **Health** → http://localhost:8000/health

---

## API Endpoints

### v1

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/predict/model/info` | Model metadata |
| `POST` | `/api/v1/predict/` | Single text prediction |
| `POST` | `/api/v1/predict/batch` | Batch prediction (1–32 texts) |

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "I hate this so much!", "language": "en"}'
```

---

## Configuration

All settings live in `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | Hate Speech Detection API | Display name |
| `DEBUG` | `False` | Enable debug mode |
| `ENVIRONMENT` | `development` | `development` / `production` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Server bind address |
| `ALLOWED_ORIGINS` | `localhost:3000,...` | CORS allowed origins |
| `MODEL_DIR` | `models/xlm_roberta_hate_model` | Path to model weights |
| `MAX_SEQUENCE_LENGTH` | `512` | Max tokens per input |
| `PREDICTION_THRESHOLD` | `0.5` | Default confidence threshold |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_health.py -v
```

---

## Adding the Model

1. Place your fine-tuned XLM-RoBERTa weights in `models/xlm_roberta_hate_model/`.
2. Uncomment the `torch` / `transformers` lines in `requirements.txt` and reinstall.
3. Implement `ModelService.load()` and `ModelService.predict()` in `services/model_service.py`.

---

## Roadmap

- [ ] Implement XLM-RoBERTa tokenization and inference
- [ ] Add Roman Kannada transliteration pre-processing
- [ ] Integrate language detection (langdetect / lingua)
- [ ] Add rate limiting middleware
- [ ] Dockerise the application
- [ ] Add authentication (API key / JWT)
