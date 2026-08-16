# 🎓 Student Performance Prediction — MLOps Project

**IT4V43 — Machine Learning Operations**

A complete, end-to-end MLOps project that predicts a student's final exam score using a regression pipeline. Every component is genuinely functional and verifiable.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Architecture](#4-architecture)
5. [MLOps Lifecycle](#5-mlops-lifecycle)
6. [Dataset](#6-dataset)
7. [Data Validation](#7-data-validation)
8. [DVC — Data Version Control](#8-dvc--data-version-control)
9. [Data Exploration](#9-data-exploration)
10. [Model Training](#10-model-training)
11. [MLflow — Experiment Tracking](#11-mlflow--experiment-tracking)
12. [Model Artifact](#12-model-artifact)
13. [SHAP — Interpretability](#13-shap--interpretability)
14. [FastAPI — Model Serving](#14-fastapi--model-serving)
15. [Streamlit — Frontend](#15-streamlit--frontend)
16. [Monitoring](#16-monitoring)
17. [Docker](#17-docker)
18. [GitHub Actions CI/CD](#18-github-actions-cicd)
19. [MLOps Maturity Level](#19-mlops-maturity-level)
20. [Installation](#20-installation)
21. [Usage](#21-usage)
22. [Testing](#22-testing)
23. [Results](#23-results)
24. [Limitations](#24-limitations)
25. [Future Scope](#25-future-scope)
26. [IT4V43 Syllabus Mapping](#26-it4v43-syllabus-mapping)

---

## 1. Project Overview

This project implements a **Student Performance Prediction** system using a genuine MLOps pipeline. It demonstrates the full machine learning operational lifecycle — from raw data generation and validation through to containerised deployment and monitoring.

The ML task is **regression**: predicting a student's final exam score (0–100) based on six academic and behavioural features.

---

## 2. Problem Statement

Student performance prediction is a well-studied educational data mining problem. Identifying at-risk students early allows institutions to provide timely support.

In this project, we build a lightweight, interpretable regression system that:
- Predicts a continuous final exam score
- Categorises performance into five bands
- Explains which features drive each prediction (SHAP)
- Provides a REST API and web interface

---

## 3. Objectives

| # | Objective | Implementation |
|---|-----------|----------------|
| 1 | End-to-end ML pipeline | `src/train.py` orchestrates all stages |
| 2 | Data quality assurance | Pandera schema validation |
| 3 | Dataset reproducibility | DVC + fixed random seed |
| 4 | Experiment tracking | MLflow with real logged runs |
| 5 | Model interpretability | SHAP values and plots |
| 6 | REST API serving | FastAPI with Prometheus metrics |
| 7 | Web UI | Streamlit calling FastAPI |
| 8 | Containerisation | Docker + Docker Compose |
| 9 | CI/CD automation | GitHub Actions |
| 10 | Quality gate | Automated metric threshold check |

---

## 4. Architecture

```
Developer
    │
    ▼
GitHub (code + DVC metadata)
    │
    ▼
GitHub Actions CI/CD
    ├─ Install dependencies
    ├─ Generate dataset
    ├─ Validate data (Pandera)
    ├─ Run pre-training tests
    ├─ Train models (MLflow)
    ├─ Run all tests (+ model + API)
    ├─ Quality gate (R² ≥ 0.75, MAE ≤ 8)
    ├─ Generate SHAP plots
    └─ Build Docker images
           │
           ▼
   Docker Compose
   ┌─────────────────────────────────┐
   │  mlflow (port 5000)             │
   │  backend/FastAPI (port 8000)    │
   │  frontend/Streamlit (port 8501) │
   └─────────────────────────────────┘
           │
           ▼
   End User (browser)
```

### Component Roles

| Component | Role |
|-----------|------|
| **Pandera** | Schema-based data validation before training |
| **DVC** | Tracks dataset versions like Git tracks code |
| **MLflow** | Logs parameters, metrics, and model artifacts per run |
| **Scikit-learn Pipeline** | Prevents data leakage; same preprocessing at train/inference |
| **Joblib** | Serialises the trained model and preprocessor |
| **SHAP** | Explains which features most influence predictions |
| **FastAPI** | Serves predictions via REST API; exposes Prometheus metrics |
| **Streamlit** | Web frontend; calls FastAPI via HTTP |
| **Prometheus** | Collects request counts, latency, and error metrics |
| **Docker Compose** | Orchestrates all services locally |
| **GitHub Actions** | Automates the test → train → gate → build pipeline |

---

## 5. MLOps Lifecycle

```
Data Generation (generate_data.py)
        ↓
Data Validation (Pandera)
        ↓
Data Versioning (DVC)
        ↓
Preprocessing (Scikit-learn Pipeline)
        ↓
Model Training (Linear Regression + Random Forest)
        ↓
Experiment Tracking (MLflow)
        ↓
Model Evaluation (MAE, RMSE, R²)
        ↓
Quality Gate (R² ≥ 0.75, MAE ≤ 8.0)
        ↓
Model Artifact (Joblib)
        ↓
SHAP Interpretability
        ↓
Model Serving (FastAPI)
        ↓
Web Frontend (Streamlit)
        ↓
Monitoring (Prometheus)
        ↓
CI/CD Automation (GitHub Actions)
```

---

## 6. Dataset

> ⚠️ **This dataset is entirely synthetic.** No real student records are used.

### Generation

The dataset is created by `src/generate_data.py` using `numpy.random.default_rng(seed=42)`.

Feature relationships are based on educational domain knowledge:
- **study_hours** and **previous_exam_score** are the strongest predictors
- **attendance_percentage** and **assignment_completion** have moderate positive effects
- **sleep_hours** has a small positive effect
- **extracurricular_hours** has a small negative effect (diminishing returns)
- Gaussian noise (σ=5) simulates real-world variability

### Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `study_hours` | float | 0–14 | Daily hours spent studying |
| `attendance_percentage` | float | 0–100 | % of classes attended |
| `previous_exam_score` | float | 0–100 | Score on previous exam |
| `assignment_completion_percentage` | float | 0–100 | % of assignments completed |
| `sleep_hours` | float | 0–24 | Average nightly sleep |
| `extracurricular_hours` | float | 0–24 | Daily extracurricular hours |

### Target

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `final_exam_score` | float | 0–100 | Predicted final exam score |

### Dataset stats (typical)

| Metric | final_exam_score |
|--------|-----------------|
| Count | 1000 |
| Mean | ~65 |
| Std | ~14 |
| Min | ~20 |
| Max | ~100 |

---

## 7. Data Validation

**Tool:** [Pandera](https://pandera.readthedocs.io/)

File: [`src/data_validation.py`](src/data_validation.py)

### What is validated?

| Check | Details |
|-------|---------|
| Required columns | All 7 columns must be present |
| Numeric types | All columns coerced to float |
| Null values | No NaN/missing values allowed |
| Value ranges | e.g., attendance in [0, 100] |
| Target range | final_exam_score in [0, 100] |

### Two schemas

- `STUDENT_SCHEMA` — full schema for training (includes target column)
- `INFERENCE_SCHEMA` — inference schema (no target column required)

### Failure behaviour

Invalid data raises `pandera.errors.SchemaErrors` with a detailed failure report, immediately halting the pipeline.

```bash
python src/data_validation.py
# [data_validation] ✓ Schema validation passed. Rows: 1000
```

---

## 8. DVC — Data Version Control

**Tool:** [DVC](https://dvc.org/)

Files: [`dvc.yaml`](dvc.yaml), [`params.yaml`](params.yaml)

### Why DVC?

- **Dataset versioning**: Track which version of the data produced which model
- **Reproducibility**: `dvc repro` re-runs only changed pipeline stages
- **Data-Git separation**: Large data files don't pollute Git history

### Pipeline stages

```
generate → validate → train
```

### Commands

```bash
# Initialise (already done)
dvc init

# Track the dataset
dvc add data/raw/students.csv

# Run full pipeline
dvc repro

# View the pipeline DAG
dvc dag

# Check what has changed
dvc status
```

### Updating the dataset

```bash
# Regenerate with different seed or size
python src/generate_data.py --n 2000 --seed 99

# Re-track with DVC
dvc add data/raw/students.csv
git add data/raw/students.csv.dvc
git commit -m "Update dataset: 2000 rows, seed=99"
```

---

## 9. Data Exploration

File: [`notebooks/eda.py`](notebooks/eda.py)

Generates visualisations to:
- Understand feature distributions
- Inspect the target variable
- Identify correlations
- Spot data quality issues

### Generated plots (saved to `reports/`)

| Plot | Description |
|------|-------------|
| `feature_distributions.png` | Histogram for each feature |
| `target_distribution.png` | Final exam score histogram with mean/median |
| `correlation_matrix.png` | Pairwise Pearson correlation heatmap |
| `feature_vs_target.png` | Study hours and previous score vs final score |
| `scatter_matrix.png` | Pairplot of key features |

```bash
python notebooks/eda.py
```

---

## 10. Model Training

File: [`src/train.py`](src/train.py)

### Models trained

| Model | Hyperparameters |
|-------|----------------|
| Linear Regression | default (no regularisation) |
| Random Forest Regressor | n_estimators=100, max_depth=8, min_samples_split=5 |

### Selection criterion

Best model = highest R² on the test set.

### Preprocessing pipeline

1. `SimpleImputer(strategy="median")` — handles any missing values
2. `StandardScaler()` — standardises features to zero mean / unit variance

The same fitted pipeline is saved alongside the model and **reused at inference time** — no data leakage.

### Run training

```bash
python src/train.py
```

### Single pipeline command (DVC)

```bash
dvc repro
```

---

## 11. MLflow — Experiment Tracking

**Experiment name:** `student-performance-prediction`

**Tracking URI:** `mlruns/` (local directory, no server required for tracking)

### Logged per run

**Parameters:**
- `model_name` (e.g., `RandomForestRegressor`)
- model hyperparameters
- `random_seed`, `test_size`, `n_train`, `n_test`

**Metrics:**
- `mae`, `rmse`, `r2`

**Artifacts:**
- Actual vs Predicted scatter plot
- Residual plot
- Feature importance plot
- Model object (via `mlflow.sklearn.log_model`)
- `model_metadata.json`
- `best_model.joblib`

### View the UI

```bash
mlflow ui --backend-store-uri mlruns
# Open: http://localhost:5000
```

Or via Docker Compose — MLflow is available at `http://localhost:5000`.

---

## 12. Model Artifact

File: `models/best_model.joblib`

The best model is saved with Joblib alongside:
- `models/preprocessor.joblib` — the fitted preprocessing pipeline
- `models/model_metadata.json` — version, metrics, run ID, quality gate status

### Model metadata example

```json
{
  "model_name": "RandomForestRegressor",
  "model_version": "1.0.0",
  "mlflow_run_id": "abc123...",
  "random_seed": 42,
  "test_size": 0.2,
  "features": ["study_hours", ...],
  "metrics": {"mae": 3.8, "rmse": 5.1, "r2": 0.89},
  "quality_gate": {"min_r2": 0.75, "max_mae": 8.0, "passed": true}
}
```

> Note: MLflow Model Registry requires a database backend (PostgreSQL / SQLite) for production use. This project uses file-based tracking for simplicity, which is appropriate for local development and academic projects.

---

## 13. SHAP — Interpretability

File: [`src/explain.py`](src/explain.py)

### Explainer selection

| Model type | SHAP Explainer |
|-----------|---------------|
| Random Forest | `TreeExplainer` (fast, exact) |
| Linear Regression | `LinearExplainer` |
| Other | `KernelExplainer` (fallback) |

### Output

- `models/shap_summary.png` — beeswarm plot (shows direction and magnitude of feature effects)
- `models/shap_bar.png` — mean |SHAP| bar chart (global feature importance)

```bash
python src/explain.py
```

### Typical feature ranking (Random Forest)

1. `previous_exam_score` — strongest predictor
2. `study_hours`
3. `attendance_percentage`
4. `assignment_completion_percentage`
5. `sleep_hours`
6. `extracurricular_hours`

---

## 14. FastAPI — Model Serving

File: [`api/main.py`](api/main.py)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Regression prediction |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI (auto-generated) |

### Predict request

```json
{
  "study_hours": 5,
  "attendance_percentage": 85,
  "previous_exam_score": 72,
  "assignment_completion_percentage": 90,
  "sleep_hours": 7,
  "extracurricular_hours": 2
}
```

### Predict response

```json
{
  "predicted_score": 78.4,
  "performance": "VERY GOOD"
}
```

### Performance categories

| Score | Category |
|-------|----------|
| 90–100 | EXCELLENT |
| 75–89 | VERY GOOD |
| 60–74 | GOOD |
| 40–59 | AVERAGE |
| 0–39 | POOR |

### Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 15. Streamlit — Frontend

File: [`app/streamlit_app.py`](app/streamlit_app.py)

### Architecture

```
Streamlit (port 8501)
       ↓ HTTP POST /predict
FastAPI (port 8000)
       ↓
Saved ML Model (Joblib)
```

> **Important:** Streamlit does NOT load the ML model directly. All predictions come from the FastAPI backend via HTTP.

### Features

- Interactive sliders for all 6 features
- Predicted score display with gradient styling
- Performance category badge
- Model version and timestamp
- Feature importance explanation (SHAP ranking)
- Live model info from API sidebar

### Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## 16. Monitoring

**Tool:** `prometheus-client`

Prometheus metrics are exposed at `GET /metrics`.

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total prediction requests |
| `prediction_latency_seconds` | Histogram | End-to-end prediction latency |
| `api_errors_total` | Counter | Errors by endpoint label |
| `health_requests_total` | Counter | Total health check calls |

```bash
curl http://localhost:8000/metrics
```

To add a full Prometheus + Grafana stack, add it to `docker-compose.yml` (left as future scope to avoid overengineering).

---

## 17. Docker

### Files

- [`Dockerfile`](Dockerfile) — FastAPI backend
- [`Dockerfile.streamlit`](Dockerfile.streamlit) — Streamlit frontend
- [`docker-compose.yml`](docker-compose.yml) — orchestrates all services

### Services

| Service | Image | Port | Description |
|---------|-------|------|-------------|
| `mlflow` | python:3.11-slim | 5000 | MLflow tracking server |
| `backend` | Built from Dockerfile | 8000 | FastAPI |
| `frontend` | Built from Dockerfile.streamlit | 8501 | Streamlit |

### Run everything

```bash
# Build and start all services
docker compose up --build

# View logs
docker compose logs -f backend

# Stop
docker compose down
```

> **Prerequisites:** Train the model locally first (`python src/train.py`) so the model artifacts exist before building Docker images. Alternatively, add a training step to the Docker entrypoint.

---

## 18. GitHub Actions CI/CD

File: [`.github/workflows/mlops.yml`](.github/workflows/mlops.yml)

### Triggers

- Push to `main` or `develop`
- Pull request to `main`

### Pipeline stages

```
1. Checkout code
2. Set up Python 3.11
3. Install dependencies
4. Generate dataset
5. Validate data (Pandera)
6. Pre-training unit tests
7. Train models (MLflow)
8. All tests (data + preprocess + model + API)
9. Model quality gate (fail if R² < 0.75 or MAE > 8)
10. SHAP explanation generation
11. Upload MLflow runs (artifact)
12. Upload model artifacts
13. Build Docker images (separate job, after pipeline passes)
```

### Failure rule

Any step failure causes the workflow to fail. The quality gate (step 9) uses `raise_on_fail=True` — if the model doesn't meet thresholds, the pipeline stops and deployment does not proceed.

---

## 19. MLOps Maturity Level

Based on the course MLOps Maturity Model:

| Level | Description | This Project |
|-------|-------------|-------------|
| **Level 0** | Manual, ad-hoc ML scripts | ✅ Baseline covered |
| **Level 1** | Automated training pipeline | ✅ Implemented (DVC + MLflow) |
| **Level 2** | Automated model deployment | ✅ Implemented (Docker + CI/CD) |
| **Level 3** | Full MLOps with automated retraining | ❌ Out of scope |

### This project achieves **Level 2**

**Evidence:**
- ✅ Data validation before every training run
- ✅ Automated pipeline (`dvc repro` or `python src/train.py`)
- ✅ Experiment tracking with MLflow
- ✅ Automated testing with Pytest
- ✅ Quality gate that blocks deployment on failure
- ✅ Model served via FastAPI
- ✅ Containerised deployment via Docker Compose
- ✅ CI/CD via GitHub Actions

**Level 3 would additionally require:**
- Automated retraining triggered by data drift or scheduled events
- Model performance monitoring with automatic rollback
- A/B testing infrastructure

These are deliberately excluded to keep the project understandable and explainable.

---

## 20. Installation

### Prerequisites

- Python 3.11+
- pip
- Git
- Docker + Docker Compose (for containerised deployment)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd student-performance-mlops

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Initialise DVC
dvc init
```

---

## 21. Usage

### Option A — Run everything manually (recommended for development)

```bash
# Step 1: Generate dataset
python src/generate_data.py

# Step 2: Validate data
python src/data_validation.py

# Step 3: Explore data
python notebooks/eda.py

# Step 4: Train models
python src/train.py

# Step 5: Generate SHAP plots
python src/explain.py

# Step 6: Run tests
pytest

# Step 7: Start FastAPI
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Step 8: Start Streamlit (new terminal)
streamlit run app/streamlit_app.py

# Step 9: View MLflow
mlflow ui --backend-store-uri mlruns
```

### Option B — DVC pipeline

```bash
dvc repro   # runs generate → validate → train
```

### Option C — Docker Compose

```bash
# Build images and start all services
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Prometheus metrics | http://localhost:8000/metrics |

---

## 22. Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api --cov-report=term-missing

# Run specific module
pytest tests/test_data_validation.py -v
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

### Test coverage

| Test file | What it tests |
|-----------|--------------|
| `test_data_validation.py` | Pandera schema: valid, invalid ranges, missing columns, nulls |
| `test_preprocessing.py` | Pipeline: split ratios, no leakage, imputation, reproducibility |
| `test_model.py` | Quality gate logic, metrics, score categorisation, model artifact |
| `test_api.py` | /health, /model-info, /predict, validation errors, /metrics |

---

## 23. Results

> Results from actual training run on synthetic dataset (1000 rows, seed=42).

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | ~4.5 | ~5.8 | ~0.84 |
| Random Forest | ~3.5 | ~4.6 | ~0.89 |

**Best model: RandomForestRegressor** (selected automatically by highest R²)

**Quality gate: PASSED** (R² ≥ 0.75 ✓, MAE ≤ 8.0 ✓)

> Note: Exact values are logged to MLflow. Run `mlflow ui` to view full results.

---

## 24. Limitations

1. **Synthetic data** — Results are realistic but not from real student records. Real-world performance would depend on actual data quality.
2. **MLflow registry** — File-based MLflow tracking is used (appropriate for local/academic use). A production registry would use a database backend.
3. **No data drift detection** — The system does not monitor for distribution shift in production inputs.
4. **No authentication** — The API has no authentication (deliberate for academic simplicity).
5. **Level 2, not Level 3** — No automated retraining triggers.
6. **Prometheus only** — Grafana dashboards are not configured (intentionally kept simple).
7. **Single-node deployment** — Docker Compose runs on one machine; no Kubernetes or horizontal scaling.

---

## 25. Future Scope

| Enhancement | Tool | Justification |
|-------------|------|--------------|
| Data drift detection | Evidently AI | Monitor production inputs vs training distribution |
| Automated retraining | DVC + cron | Trigger training when drift is detected (Level 3) |
| Grafana dashboards | Grafana + Prometheus | Visualise metrics over time |
| MLflow registry | PostgreSQL backend | Formal model versioning and staging/production promotion |
| Hyperparameter tuning | Optuna | Systematic search instead of manual tuning |
| Feature store | Feast | Centralised feature management for multiple models |

---

## 26. IT4V43 Syllabus Mapping

| IT4V43 Topic | Project Implementation | File(s) |
|--------------|----------------------|---------|
| **MLOps lifecycle** | Data → Validation → Versioning → Training → Tracking → Testing → Serving → Deployment → Monitoring | All |
| **Automation** | `dvc repro` runs full pipeline; GitHub Actions automates CI/CD | `dvc.yaml`, `mlops.yml` |
| **Continuous X** | CI on every push; automated test → train → quality gate → build | `.github/workflows/mlops.yml` |
| **Versioning** | Git for code; DVC for data; Joblib + metadata for models | `dvc.yaml`, `model_metadata.json` |
| **Experiment tracking** | MLflow logs params, metrics, plots, and model per run | `src/train.py`, `mlruns/` |
| **Testing** | Pytest: data tests, preprocessing tests, model tests, API tests | `tests/` |
| **Monitoring** | Prometheus metrics at `/metrics` (request count, latency, errors) | `api/main.py` |
| **Reproducibility** | Fixed seeds, pinned requirements, DVC pipeline, saved preprocessor | `requirements.txt`, `params.yaml` |
| **Deployment** | Docker + Docker Compose; `docker compose up --build` | `Dockerfile`, `docker-compose.yml` |
| **Data validation** | Pandera schema: types, ranges, nulls, required columns | `src/data_validation.py` |
| **Data exploration** | EDA plots: distributions, correlation, scatter | `notebooks/eda.py` |
| **Data version control** | DVC tracks `data/raw/students.csv` | `dvc.yaml` |
| **Model serving** | FastAPI REST API with auto-docs | `api/main.py` |
| **Model interpretability** | SHAP TreeExplainer / LinearExplainer; summary + bar plots | `src/explain.py` |
| **MLOps architecture** | Fully documented pipeline with component roles | This README |
| **MLOps maturity model** | Level 2: Automated training + deployment; Level 3 limitations documented | Section 19 |
| **CI/CD** | GitHub Actions: test → validate → train → quality gate → build | `.github/workflows/mlops.yml` |
| **Automated ML testing** | Quality gate with R² ≥ 0.75, MAE ≤ 8.0; fails CI on threshold breach | `src/evaluate.py`, `mlops.yml` |
| **Automated model deployment** | Docker images built in CI after quality gate passes | `.github/workflows/mlops.yml` |

---

## Project Structure

```
student-performance-mlops/
├── data/
│   ├── raw/students.csv         ← synthetic dataset (DVC tracked)
│   └── processed/               ← (reserved for future use)
├── src/
│   ├── __init__.py
│   ├── generate_data.py         ← synthetic dataset generator
│   ├── data_validation.py       ← Pandera schema validation
│   ├── preprocess.py            ← Scikit-learn Pipeline
│   ├── train.py                 ← full training pipeline
│   ├── evaluate.py              ← metrics + quality gate
│   ├── explain.py               ← SHAP interpretability
│   └── predict.py               ← inference module
├── api/
│   ├── __init__.py
│   └── main.py                  ← FastAPI application
├── app/
│   └── streamlit_app.py         ← Streamlit frontend
├── models/
│   ├── best_model.joblib        ← trained model (generated)
│   ├── preprocessor.joblib      ← fitted preprocessing pipeline
│   ├── model_metadata.json      ← version, metrics, run ID
│   ├── shap_summary.png         ← SHAP beeswarm plot
│   └── shap_bar.png             ← SHAP feature importance
├── tests/
│   ├── __init__.py
│   ├── conftest.py              ← shared fixtures
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_api.py
├── notebooks/
│   └── eda.py                   ← data exploration script
├── reports/                     ← EDA plots (generated)
├── mlruns/                      ← MLflow experiment runs (generated)
├── .github/
│   └── workflows/
│       └── mlops.yml            ← GitHub Actions CI/CD
├── Dockerfile                   ← FastAPI image
├── Dockerfile.streamlit         ← Streamlit image
├── docker-compose.yml
├── dvc.yaml                     ← DVC pipeline
├── params.yaml                  ← DVC parameters
├── requirements.txt
├── .gitignore
└── README.md
```

---

*IT4V43 — Machine Learning Operations | Student Performance MLOps Project*
