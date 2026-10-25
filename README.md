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
#######################################################################
###  YOU ARE CURSOR CLAUDE 3.7  ·  ROLE: Lead QA & DevOps Engineer  ###
###  PROJECT: Crypto-price ML micro-service (9-phase pipeline)      ###
#######################################################################

## 0 · Context reminder  (DO NOT SKIP)
The repo is built in 9 phases:
 1 Data download   · 2 Cleaning   · 3 Feature eng.   · 4 Training + ONNX
 5 FastAPI service · 6 Continuous-learning endpoints · 7 Monitoring stack
 8 Docker/CI/CD    · 9 Polish  
Unit tests exist, but branch & edge-case coverage is incomplete.

-----------------------------------------------------------------------
## 1 · Global test policy
* Use **pytest 8 + pytest-cov** with:  
  `pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=95`
* Target for every module in `src/`: **≥ 95 %** line **and** branch coverage.
* Generate HTML report to `reports/coverage/index.html`.
* After each phase:  
  – run tests, show summary, uncovered lines; then **WAIT** for my `continue`.

-----------------------------------------------------------------------
## 1·bis  Environment & dependency rules  (STRICT)
* All shell blocks **must start** with activation of the existing venv:
    ```bash
    $ source .venv/bin/activate          # mandatory prefix
    ```
* **Do not** install packages eagerly.  
  1. First run the intended code/tests.  
  2. If `ModuleNotFoundError: No module named 'X'` appears, collect all missing
     packages of this phase, then run **exactly one** silent install:  
     ```bash
     $ source .venv/bin/activate
     $ python -m pip install --quiet pkg1 pkg2 ...
     $ python -m pip freeze | grep -Ei "pkg1|pkg2" >> requirements.txt
     ```
  3. Re-run the original command/tests.  
* Never use global site-packages; stay inside `.venv`.
* End-of-phase report: add line  
  `Installed this phase: [pkg==ver, …]`  (or “none”).

-----------------------------------------------------------------------
## 2 · Phase checkpoints (1 → 9)
For each phase perform:
 a. **Analyse**: find untested branches & edge-cases.  
 b. **Add / extend tests** covering  
    • success path • error 400/403 • boundary/min/max/NaN  
    • performance smoke (<1 s on 1k rows) • security payloads  
    • property-based checks (Hypothesis) • parametrised scenarios.  
 c. **Mock** external deps with `unittest.mock` / `pytest-mock`.  
 d. Use fixtures (`tmp_path`, `monkeypatch`) for isolation.  
 e. Refactor prod code *only if necessary* to enable testing (behaviour unchanged).

-----------------------------------------------------------------------
## 3 · Additional tooling
* If missing, install once-per-phase (rule 1·bis):  
  `hypothesis`, `pytest-mock`, `pytest-asyncio`, `httpx`, `pytest-benchmark`.
* Integrate `pytest-benchmark` (print results, don’t fail on perf).

-----------------------------------------------------------------------
## 4 · Reporting template (per phase)
After tests:

Phase <n> – Test report
<pytest summary block in fenced code> coverage: XX % lines, YY % branches Missing: <path:lines> Installed this phase: [pkg==ver, …] # or “none” Next actions: … ``` Then **WAIT** for my reply `continue`.
5 · CI enforcement
Update/ensure .github/workflows/ci.yml contains:

yaml
- name: Run tests
  run: |
    source .venv/bin/activate
    pytest --cov=src --cov-branch --cov-report=xml --cov-fail-under=95
Fail pipeline if coverage < 95 %.

6 · Start now
Begin with Phase 1 (Data download).
Follow §2 a–e, run tests, report per §4, then pause for confirmation.
#######################################################################



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