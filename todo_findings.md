# Project TODO & Findings

## 1. Project Structure & Missing Components
- **FastAPI service**: No `api/` directory or FastAPI application present. Need to create an inference service (`api/main.py`) exposing endpoints for predictions, health checks, and model metadata.
- **Cloud Storage integration**: No code to read/write from cloud storage (e.g., Google Cloud Storage or AWS S3). Add utilities to load raw data and write processed Parquet files using cloud SDKs.
- **GitHub CI/CD**: No GitHub Actions workflows for testing, linting, building Docker images, or deploying the FastAPI service. Create `.github/workflows/` with appropriate pipelines.

## 2. Configuration (`config/settings.py`)
- File contains duplicated and malformed import blocks, undefined variables, and stray comments (`LOG#file location…`).
- **Action items**:
  - Consolidate imports into a single block.
  - Define all required constants (paths, Spark configs, model paths, Cloud bucket URIs) with clear defaults.
  - Remove dead code/comments and add type hints.
  - Ensure values are loaded from environment variables or a `.env` file for flexibility.

## 3. ETL Pipeline (`etl/spark_pipeline.py`)
- Uses undefined functions/variables (`F` instead of `functions` alias, missing import for `functions as F`).
- Column rename logic may reference `COLUMN_MAP` that is not defined correctly in settings.
- **Data quality**: Basic validation present but could be extended (e.g., duplicate detection, outlier handling).
- **Action items**:
  - Fix imports (`from pyspark.sql import functions as F`).
  - Ensure `COLUMN_MAP` and `REQUIRED_COLUMNS` are correctly defined.
  - Add unit tests for each ETL step.
  - Implement optional parameter to write output directly to cloud storage.

## 4. Model Training (`models/train.py`)
- Several bugs:
  - `self.version` is referenced before being set (used in `save_best_model`).
  - `save_best_model` called without required `version` argument.
  - Path construction for `METRICLOG` and other constants malformed.
  - Duplicate `import numpy as np` and unused imports.
  - In `train_quantile_low_model`/`high_model`, `self.model.fit` called instead of the quantile model variable.
- **Action items**:
  - Refactor to use a clear versioning scheme (e.g., timestamp or git commit hash).
  - Correct model saving logic for quantile models.
  - Add proper exception handling and logging.
  - Write unit tests for hyperparameter tuning and model persistence.

## 5. Evaluation (`models/evaluate.py`)
- Minor issues:
  - `self.plot_metrics` uses `axs[i].ylabel` (should be `set_ylabel`).
  - `save_metrics` writes to `MODEL_DIR/f"model_metrics_{split}.json"` but `MODEL_DIR` may not exist.
  - `analyze_residuals` signature mismatched when called (`self.analyze_residuals(self.test, 'Test')` missing model argument).
- **Action items**:
  - Fix matplotlib usage errors.
  - Ensure directory creation before saving files.
  - Align method signatures with calls.
  - Add automated test coverage for evaluation utilities.

## 6. Monitoring & MLOps (`monitoring/*`)
- Files present but lack integration:
  - No scheduler to trigger drift detection after each model retrain.
  - No logging to a central observability platform.
  - `mlops_pipeline.py` is empty/placeholder.
- **Action items**:
  - Implement a periodic job (e.g., Airflow, Prefect, or simple cron) that runs `drift_detector.py` and updates `metrics_monitor.py`.
  - Connect alerts to a notification channel (e.g., Slack or email).
  - Add documentation on how to monitor model performance.

## 7. Code Quality & Documentation
- Missing type hints and docstrings in many modules.
- No `requirements.txt` pinning versions for Spark, LightGBM, FastAPI, etc.
- No tests (`tests/` directory absent).
- **Action items**:
  - Add comprehensive README with architecture diagram.
  - Introduce `pytest` test suite covering ETL, training, inference.
  - Add `mypy` and `flake8` configurations for linting.
  - Create `Dockerfile` for the FastAPI service and a `docker-compose.yml` for local development.

## 8. Security & Deployment
- Secrets (e.g., cloud credentials) are not managed securely.
- No containerization or deployment scripts.
- **Action items**:
  - Use environment variables or secret manager for credentials.
  - Provide Helm chart / Kubernetes manifests for production deployment.
  - Ensure HTTPS for FastAPI endpoints.

---

### Prioritization
1. **Fix configuration** (`settings.py`).
2. **Resolve immediate code errors** in ETL and training modules.
3. **Add FastAPI inference service**.
4. **Integrate Cloud Storage** for data I/O.
5. **Implement monitoring & MLOps pipeline**.
6. **Add testing, CI/CD, and documentation**.
7. **Containerize & secure deployment**.

This TODO captures all identified gaps and next steps to bring the project in line with the stated requirements.
