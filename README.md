# Cryptocurrency Price Prediction Service

This project is a machine learning microservice that predicts cryptocurrency prices using historical data from Kaggle.

## Project Goals
- Download and preprocess cryptocurrency OHLC-V data
- Engineer relevant features for price prediction
- Train and optimize ML models for price forecasting
- Deploy a FastAPI service for real-time predictions
- Implement monitoring and continuous learning pipeline

## Directory Structure
```
├── data/                  # Data storage
│   ├── raw/               # Raw data from Kaggle
│   ├── processed/         # Cleaned, processed datasets
│   ├── external/          # External data sources
│   └── interim/           # Intermediate processed data
├── notebooks/             # Exploratory Jupyter notebooks
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── features/          # Feature engineering
│   ├── models/            # Model training and evaluation
│   ├── visualization/     # Visualization scripts
│   ├── config/            # Configuration files
│   └── app/               # FastAPI application
├── models/                # Saved model artifacts
├── reports/figures        # Generated figures and reports
├── tests/                 # Unit and integration tests
├── .env                   # Environment variables
├── dvc.yaml               # DVC pipeline configuration
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container setup
└── .flake8                # Linter configuration

**You are Cursor Claude 3.7 acting as Lead QA & DevOps for the crypto-price ML service.**

# 0 · Context reminder  (DO NOT SKIP)
Project already built in 9 phases:
  1-data download (Kaggle) · 2-cleaning · 3-features · 4-training/ONNX ·
  5-FastAPI · 6-continuous-learning endpoints · 7-monitoring ·
  8-docker/CI · 9-polish.
Unit tests exist, but branch- & edge-case coverage is incomplete.

# 1 · Global test policy
* Use **pytest 8 + pytest-cov** with `--cov=src --cov-branch --cov-report=term-missing --cov-fail-under=95`.
* Target: ≥ 95 % line **и** branch coverage for every module inside `src/`.
* Output HTML report to `reports/coverage/index.html`.
* After each phase below:  
  ▶ run tests, paste summary table, show uncovered lines, wait for my **continue**.

# 2 · Phase checkpoints
For each phase (1 → 9) ANALYSE code, then:
  a. **Identify untested branches & edge cases.**  
  b. **Add/extend tests**:
     • success path  
     • failure/error path (400/403 responses, file not found, wrong CSV schema, etc.)  
     • boundary values (min/max, empty DF, single-row DF, NaN bursts)  
     • performance smoke (< 1 s on 1 000 rows)  
     • security (SQL-injection-like payload, path traversal upload)  
     • property-based tests with Hypothesis where numeric ranges exist  
     • parametrized scenarios to avoid duplication.
  c. **Mock external deps** (network, file-system, MLflow server) via `unittest.mock` or `pytest-mock`.
  d. **Use fixtures** (`tmp_path`, `monkeypatch`) for isolation.
  e. **Refactor** prod code *only if* needed to unlock testability (keep behaviour).

# 3 · Additional tooling
* Install (if missing): `hypothesis`, `pytest-mock`, `pytest-asyncio`, `httpx`, `pytest-benchmark`.
* Integrate `pytest-benchmark` but **do not fail** on performance yet; just print.

# 4 · Reporting template per phase
After tests run, output:

Phase <n> – Test report
lua
Copy
Edit
<pytest summary block in fenced code>
`coverage: xx % lines, yy % branches`
Missing: <path:lines>.
Next actions: …
(then WAIT for my **continue**)
5 · CI enforcement
Update .github/workflows/ci.yml step:

yaml
Copy
Edit
- name: Run tests
  run: |
    pytest --cov=src --cov-branch --cov-report=xml --cov-fail-under=95
Fail pipeline below threshold.

6 · Start now
Begin with Phase 1 (data download).
Follow steps 2a-2e, run tests, report as per §4, then stop for my confirmation.

markdown
Copy
Edit


## Running Instructions

### Setup
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up Kaggle API token in `.env`
4. Download data: `python src/data/download_dataset.py`

### Running the API
```bash
# Development mode
uvicorn src.app.main:app --reload

# Production mode with Docker
docker-compose up -d
```

### Model Training
```bash
# Run the full DVC pipeline
dvc repro

# View experiment tracking
mlflow ui
```

## Monitoring
- API metrics: http://localhost:8000/metrics
- Prometheus: http://localhost:9090
- Grafana dashboards: http://localhost:3000 