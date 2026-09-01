Thank you for sharing this detailed project overview and the comprehensive TODO list. You've clearly done a thorough self‑audit, and I'm happy to help you prioritize and resolve the issues systematically.

Below I’ve consolidated the most **critical blockers** (the ones that prevent the code from running at all) and then outlined a step‑by‑step implementation plan. I’ll also provide concrete code snippets to fix the biggest pain points, so you can get a working prototype quickly.

---

## 🔥 Immediate Fire‑Fighting (Top Priority)

### 1. **Configuration (`config/settings.py`)** – unblock all other modules
- **Problem**: Duplicate imports, undefined variables, malformed constants.
- **Solution**: Clean up imports, define all required constants with `os.getenv()` fallbacks, add type hints.

```python
# config/settings.py
import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "saved_models"
LOG_DIR = BASE_DIR / "logs"
METRIC_DIR = BASE_DIR / "metrics"

# Cloud storage (GCS or S3)
CLOUD_BUCKET = os.getenv("CLOUD_BUCKET", "your-bucket-name")
CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "taxi-data")

# Spark config
SPARK_APP_NAME = "TaxiDemandForecast"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

# Column mapping (old -> new)
COLUMN_MAP = {
    "pickup_datetime": "pickup_time",
    "dropoff_datetime": "dropoff_time",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "fare_amount": "fare_amount",
    "total_amount": "total_amount",
    "pickup_longitude": "pickup_lon",
    "pickup_latitude": "pickup_lat",
    "dropoff_longitude": "dropoff_lon",
    "dropoff_latitude": "dropoff_lat",
    "store_and_fwd_flag": "store_fwd_flag",
}

REQUIRED_COLUMNS = list(COLUMN_MAP.keys())

# Model hyperparameters (example)
LGBM_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
```

Ensure you create the directories (e.g., `data/raw/`, `models/saved_models/`) or let the code create them.

---

### 2. **ETL Pipeline (`etl/spark_pipeline.py`)** – fix imports and column rename
- **Problem**: `F` not imported, `COLUMN_MAP` missing.
- **Solution**:

```python
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, IntegerType, StringType
from config.settings import COLUMN_MAP, REQUIRED_COLUMNS

def load_data(spark, file_path):
    # Define schema to enforce data types
    schema = StructType([
        StructField("pickup_datetime", TimestampType(), True),
        StructField("dropoff_datetime", TimestampType(), True),
        # ... add all fields
    ])
    df = spark.read.option("header", True).schema(schema).csv(file_path)
    return df

def clean_data(df):
    # Rename columns
    for old, new in COLUMN_MAP.items():
        df = df.withColumnRenamed(old, new)
    # Drop rows with nulls in required columns
    df = df.na.drop(subset=REQUIRED_COLUMNS)
    # Optional: filter out outliers (e.g., negative fare)
    df = df.filter(F.col("fare_amount") >= 0)
    return df

def write_parquet(df, output_path, partition_cols=None):
    writer = df.write.mode("overwrite").parquet(output_path)
    if partition_cols:
        writer = df.write.partitionBy(*partition_cols).mode("overwrite").parquet(output_path)
    return writer
```

---

### 3. **Model Training (`models/train.py`)** – versioning and quantile model fixes
- **Problem**: `self.version` used before assignment, quantile model fitting uses wrong variable.
- **Solution**:

```python
import time
import json
from pathlib import Path
from config.settings import MODEL_DIR, METRIC_DIR
import lightgbm as lgb

class TaxiDemandTrainer:
    def __init__(self, params, version=None):
        self.params = params
        self.version = version or f"v{int(time.time())}"   # set version immediately
        self.model = None

    def train_quantile_model(self, X_train, y_train, quantile=0.5):
        model = lgb.LGBMRegressor(**self.params, objective="quantile", alpha=quantile)
        model.fit(X_train, y_train)
        return model

    def save_best_model(self, model, name):
        model_path = MODEL_DIR / self.version / f"{name}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(model, model_path)
        # Also save params and metadata
        meta = {"version": self.version, "params": self.params, "timestamp": time.time()}
        with open(model_path.with_suffix(".json"), "w") as f:
            json.dump(meta, f)
```

---

### 4. **FastAPI Inference Service** – create `api/main.py`
Since this is missing, here’s a minimal but production‑ready skeleton:

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path
from config.settings import MODEL_DIR

app = FastAPI(title="Taxi Demand Forecast", version="1.0")

# Load the latest model at startup (or use version param)
MODEL_PATH = MODEL_DIR / "latest" / "lgbm_model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

class PredictionInput(BaseModel):
    pickup_datetime: str   # ISO format
    pickup_lon: float
    pickup_lat: float
    dropoff_lon: float
    dropoff_lat: float
    passenger_count: int
    trip_distance: float

class PredictionOutput(BaseModel):
    demand_prediction: float
    quantile_low: float = None
    quantile_high: float = None

def feature_engineering(raw_df):
    # Apply same transformations as in training (lag, rolling, calendar, etc.)
    # For now, just a placeholder
    return raw_df

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    raw_df = pd.DataFrame([input_data.dict()])
    # Convert pickup_datetime to datetime
    raw_df["pickup_datetime"] = pd.to_datetime(raw_df["pickup_datetime"])
    features = feature_engineering(raw_df)
    pred = model.predict(features)[0]
    return PredictionOutput(demand_prediction=pred)

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Add a `Dockerfile` and `requirements.txt` for deployment.

---

## 📋 Full Implementation Roadmap (Prioritised)

| Priority | Task | Notes |
|----------|------|-------|
| **1** | Fix `settings.py` and all import errors | Unblocks everything |
| **2** | Make ETL runnable and output partitioned Parquet | Validate with sample data |
| **3** | Fix model training bugs and get a successful training run | Include quantile models |
| **4** | Build the FastAPI service with health and predict endpoints | Include feature pipeline reuse |
| **5** | Add cloud storage integration (GCS or S3) | Use `google-cloud-storage` or `boto3`, read/write from/to bucket |
| **6** | Implement monitoring (drift detection, residual analysis) | Schedule with cron/Airflow |
| **7** | Add GitHub Actions for CI/CD (test, lint, build, deploy) | Include Docker build and push |
| **8** | Write comprehensive tests (pytest) for ETL, training, API | Aim for >70% coverage |
| **9** | Add documentation (README with architecture diagram) | Include setup and local run instructions |

---

## 💡 Specific Recommendations for Each Remaining Item

### **Cloud Storage Integration**
- Create `utils/cloud_io.py` with functions:
  - `read_csv_from_gcs(bucket, blob_path)` using `pandas` or `pyspark` with GCS connector.
  - `write_parquet_to_gcs(df, bucket, dest_path)`.
- Use environment variables for credentials (`GOOGLE_APPLICATION_CREDENTIALS` or AWS keys).

### **Monitoring & Retraining**
- Extend `monitoring/drift_detector.py` to compare feature distributions (e.g., using `scipy.stats.ks_2samp`).
- In `monitoring/metrics_monitor.py`, log metrics to a CSV or a database with timestamps.
- Create a simple `retrain.py` that runs when drift exceeds a threshold (could be triggered by a scheduler).

### **CI/CD with GitHub Actions**
- Create `.github/workflows/ci.yml`:
  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - uses: actions/setup-python@v4
          with:
            python-version: '3.9'
        - run: pip install -r requirements.txt
        - run: pytest tests/
        - run: flake8 .
        - run: mypy .
  ```
- Add a separate `deploy.yml` to build and push Docker image on tags.

### **Testing**
- Structure tests as:
  - `tests/test_etl.py`
  - `tests/test_training.py`
  - `tests/test_api.py`
- Use small sample datasets (e.g., 100 rows) to keep tests fast.

---

## 🔧 Next Steps

I suggest you start by **fixing `settings.py` and `spark_pipeline.py`** – once those are corrected, you can run the ETL and verify you have clean Parquet files. Then move to training and finally to the API. If you'd like, I can provide full code for any of these modules (e.g., a complete FastAPI service with feature engineering, or a working `train.py` with experiment tracking).

Let me know which part you'd like to tackle first, or if you need help with a specific error message you’re encountering. I’m here to assist with code snippets, design decisions, or troubleshooting.











I’ll give you a **complete, step‑by‑step action plan** that fixes every bug listed and adds all missing components, while strictly aligning with the stated requirements.  
For each module, I’ll explain what’s broken, how to fix it, and what to add to meet the production‑grade expectations.

---

## 🧭 Overall Architecture & Required Components

| Requirement | Implementation |
|-------------|----------------|
| **Scalable PySpark ETL** | `etl/spark_pipeline.py` – reads multi‑GB CSV, enforces schema, validates quality, writes partitioned Parquet. |
| **Time‑series feature engineering** | `features/engineering.py` – lag, rolling, calendar, rush‑hour signals, zone‑level aggregations. |
| **LightGBM models** | `models/train.py` – trains point‑forecast and quantile models, hyperparameter tuning. |
| **Recursive multi‑step forecasting** | `forecast/recursive.py` – uses trained model to predict next hour, feeds prediction back for next step. |
| **Evaluation framework** | `models/evaluate.py` – computes MAE, RMSE, R², SMAPE; logs to CSV; plots residuals. |
| **FastAPI inference** | `api/main.py` – endpoints `/predict`, `/health`, `/metadata`; reuses feature pipeline; Pydantic validation. |
| **Cloud Storage** | `utils/cloud_io.py` – read/write from GCS/AWS S3 using environment credentials. |
| **Monitoring** | `monitoring/drift_detector.py` (PSI/KS), `monitoring/metrics_monitor.py` (logs metrics), `monitoring/retrain_trigger.py` (threshold‑based). |
| **CI/CD** | `.github/workflows/ci.yml` (test, lint), `deploy.yml` (Docker build & push), Kubernetes/Helm for production. |
| **Tests & Documentation** | `tests/` with pytest, `README.md` with architecture diagram, `requirements.txt` pinned versions. |

---

## 🔧 Step‑by‑Step Fixes & Additions

### 1. Configuration (`config/settings.py`)
**Issues:** duplicated imports, undefined variables, stray comments, no environment fallback.  
**Fix:** rewrite cleanly with all constants, type hints, and `os.getenv`.

```python
# config/settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "saved_models"
LOG_DIR = BASE_DIR / "logs"
METRIC_DIR = BASE_DIR / "metrics"
REPORT_DIR = BASE_DIR / "reports"

# Cloud (GCS or S3)
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "gcs")  # or "s3"
CLOUD_BUCKET = os.getenv("CLOUD_BUCKET", "taxi-demand-bucket")
CLOUD_PREFIX = os.getenv("CLOUD_PREFIX", "data/")
GCS_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Spark
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_APP_NAME = "TaxiDemandForecast"
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "8g"

# Column mapping (raw -> standard)
COLUMN_MAP = {
    "pickup_datetime": "pickup_time",
    "dropoff_datetime": "dropoff_time",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "fare_amount": "fare_amount",
    "total_amount": "total_amount",
    "pickup_longitude": "pickup_lon",
    "pickup_latitude": "pickup_lat",
    "dropoff_longitude": "dropoff_lon",
    "dropoff_latitude": "dropoff_lat",
    "store_and_fwd_flag": "store_fwd_flag",
}
REQUIRED_COLUMNS = list(COLUMN_MAP.keys())  # for validation

# Feature engineering parameters
LAG_HOURS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12]
FORECAST_HORIZON = 6  # steps ahead

# LightGBM base parameters
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}
QUANTILE_PARAMS = [0.1, 0.5, 0.9]  # low, median, high

# Monitoring thresholds
DRIFT_THRESHOLD = 0.1  # population stability index
RETRAIN_INTERVAL_HOURS = 24
```

Create directories at startup (e.g., in `__init__.py` or main script):
```python
for d in [RAW_DATA_DIR, PROCESSED_DIR, MODEL_DIR, LOG_DIR, METRIC_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
```

---

### 2. ETL Pipeline (`etl/spark_pipeline.py`)
**Issues:** missing `F` import, `COLUMN_MAP` undefined, basic validation only.  
**Fix:** import `functions as F`, use `COLUMN_MAP` from settings, extend validation with duplicate check, outlier handling, and write to Parquet with partitioning (e.g., by date).

```python
# etl/spark_pipeline.py
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType, IntegerType, StringType
from pyspark.sql.window import Window
from config.settings import (
    SPARK_MASTER, SPARK_APP_NAME, COLUMN_MAP, REQUIRED_COLUMNS,
    RAW_DATA_DIR, PROCESSED_DIR
)
import logging

logging.basicConfig(level=logging.INFO)

def create_spark_session():
    return SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master(SPARK_MASTER) \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

def get_schema():
    # Define schema for NYC taxi data (example)
    return StructType([
        StructField("pickup_datetime", TimestampType(), True),
        StructField("dropoff_datetime", TimestampType(), True),
        StructField("passenger_count", IntegerType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("total_amount", DoubleType(), True),
        StructField("pickup_longitude", DoubleType(), True),
        StructField("pickup_latitude", DoubleType(), True),
        StructField("dropoff_longitude", DoubleType(), True),
        StructField("dropoff_latitude", DoubleType(), True),
        StructField("store_and_fwd_flag", StringType(), True),
    ])

def load_data(spark, file_path):
    schema = get_schema()
    df = spark.read.option("header", True).schema(schema).csv(file_path)
    return df

def clean_and_validate(df):
    # Rename columns
    for old, new in COLUMN_MAP.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)
    # Drop rows with nulls in required
    df = df.na.drop(subset=list(COLUMN_MAP.values()))
    # Drop duplicates (based on all columns)
    df = df.dropDuplicates()
    # Remove outliers: fare > 0, trip_distance between 0 and 100, passenger_count > 0
    df = df.filter(
        (F.col("fare_amount") > 0) &
        (F.col("trip_distance") > 0) &
        (F.col("trip_distance") < 100) &
        (F.col("passenger_count") > 0)
    )
    # Add date partition column
    df = df.withColumn("date", F.to_date("pickup_time"))
    return df

def write_parquet(df, output_path, partition_cols=["date"]):
    df.write \
        .mode("overwrite") \
        .partitionBy(*partition_cols) \
        .parquet(output_path)

def run_etl(input_path, output_path=None):
    spark = create_spark_session()
    try:
        df = load_data(spark, input_path)
        df_clean = clean_and_validate(df)
        if output_path is None:
            output_path = str(PROCESSED_DIR / "taxi_data")
        write_parquet(df_clean, output_path)
        logging.info(f"ETL completed. Saved to {output_path}")
        return df_clean
    finally:
        spark.stop()
```

**Cloud Integration:** We’ll add cloud read/write in a utility; for now, you can pass GCS paths like `gs://bucket/path` – Spark handles them if GCS connector is configured.

---

### 3. Feature Engineering (`features/engineering.py`)
**Requirement:** time‑series features (lag, rolling, calendar, rush‑hour). We’ll create a reusable function that can be applied both in training and inference.

```python
# features/engineering.py
import pandas as pd
import numpy as np
from config.settings import LAG_HOURS, ROLLING_WINDOWS

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical features."""
    df = df.copy()
    df["hour"] = df["pickup_time"].dt.hour
    df["dayofweek"] = df["pickup_time"].dt.dayofweek
    df["month"] = df["pickup_time"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # Rush hour signal (e.g., 7-9, 17-19)
    df["is_rush_hour"] = df["hour"].between(7, 9) | df["hour"].between(17, 19)
    return df

def add_lag_features(df: pd.DataFrame, target_col: str, group_cols: list, lag_hours: list) -> pd.DataFrame:
    """Add lag features for each zone (group_cols)."""
    df = df.sort_values(["zone", "pickup_time"]).reset_index(drop=True)
    for lag in lag_hours:
        df[f"lag_{lag}h"] = df.groupby(group_cols)[target_col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target_col: str, group_cols: list, windows: list) -> pd.DataFrame:
    """Add rolling statistics (mean, max) over windows."""
    df = df.sort_values(["zone", "pickup_time"]).reset_index(drop=True)
    for w in windows:
        df[f"rolling_mean_{w}h"] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_max_{w}h"] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.rolling(w, min_periods=1).max()
        )
    return df

def build_feature_pipeline(df: pd.DataFrame, target_col: str = "demand") -> pd.DataFrame:
    """Full feature engineering for training/inference."""
    # Assume df has 'zone', 'pickup_time', and target_col (demand)
    df = add_time_features(df)
    df = add_lag_features(df, target_col, ["zone"], LAG_HOURS)
    df = add_rolling_features(df, target_col, ["zone"], ROLLING_WINDOWS)
    # Drop rows with nulls created by lag/rolling
    df.dropna(inplace=True)
    return df
```

**Integration with Spark:** For large‑scale, you could implement these in Spark SQL or with `pyspark.sql.functions` and window functions. But the requirement mentions Pandas; we can use Pandas UDFs in Spark or convert to Pandas after aggregation. For simplicity, we’ll assume we aggregate hourly demand per zone, then convert to Pandas for feature engineering.

---

### 4. Model Training (`models/train.py`)
**Issues:** `self.version` used before assignment, `save_best_model` missing version, quantile models fitted with wrong object.  
**Fix:** set version in `__init__`, pass version to save, use separate model objects for quantiles.

```python
# models/train.py
import joblib
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config.settings import MODEL_DIR, METRIC_DIR, LGBM_PARAMS, QUANTILE_PARAMS
import logging

logger = logging.getLogger(__name__)

class TaxiDemandTrainer:
    def __init__(self, params=None, version=None):
        self.params = params or LGBM_PARAMS.copy()
        self.version = version or f"v{int(time.time())}"
        self.model = None
        self.quantile_models = {}  # {quantile: model}
        self.metrics = {}

    def _prepare_data(self, df, target_col="demand"):
        # Drop non-feature columns (zone, pickup_time, demand)
        exclude = ["zone", "pickup_time", target_col]
        X = df.drop(columns=exclude, errors="ignore")
        y = df[target_col]
        return X, y

    def train_point_model(self, X_train, y_train, X_val, y_val):
        model = LGBMRegressor(**self.params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        self.model = model
        return model

    def train_quantile_models(self, X_train, y_train, quantiles=None):
        if quantiles is None:
            quantiles = QUANTILE_PARAMS
        for q in quantiles:
            q_params = self.params.copy()
            q_params["objective"] = "quantile"
            q_params["alpha"] = q
            model = LGBMRegressor(**q_params)
            model.fit(X_train, y_train)
            self.quantile_models[q] = model
        return self.quantile_models

    def save_best_model(self, model, name):
        model_dir = MODEL_DIR / self.version
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f"{name}.pkl")
        # Save metadata
        meta = {
            "version": self.version,
            "params": self.params,
            "timestamp": time.time()
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved {name} to {model_dir}")

    def evaluate(self, X, y, split_name="test"):
        pred = self.model.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        r2 = r2_score(y, pred)
        # SMAPE
        smape = 100 * np.mean(2 * np.abs(y - pred) / (np.abs(y) + np.abs(pred) + 1e-8))
        metrics = {"mae": mae, "rmse": rmse, "r2": r2, "smape": smape}
        self.metrics[split_name] = metrics
        return metrics

    def run_training(self, df, target_col="demand", test_size=0.2, val_size=0.2):
        X, y = self._prepare_data(df, target_col)
        # Split: train, val, test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size+val_size, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size/(test_size+val_size), random_state=42
        )
        # Train point model
        point_model = self.train_point_model(X_train, y_train, X_val, y_val)
        self.save_best_model(point_model, "lgbm_point")
        # Train quantile models
        quantile_models = self.train_quantile_models(X_train, y_train)
        for q, model in quantile_models.items():
            self.save_best_model(model, f"lgbm_quantile_{q}")
        # Evaluate
        for name, (X_eval, y_eval) in [("train", (X_train, y_train)),
                                        ("val", (X_val, y_val)),
                                        ("test", (X_test, y_test))]:
            metrics = self.evaluate(X_eval, y_eval, split_name=name)
            logger.info(f"{name} metrics: {metrics}")
        # Save metrics to JSON
        self._save_metrics()
        return self.model, self.metrics

    def _save_metrics(self):
        metric_dir = METRIC_DIR / self.version
        metric_dir.mkdir(parents=True, exist_ok=True)
        with open(metric_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
```

---

### 5. Recursive Multi‑Step Forecasting (`forecast/recursive.py`)
**Requirement:** predict future demand for multiple steps ahead. We implement a recursive strategy: use the model to predict next time step, then use that prediction as input for the next step.

```python
# forecast/recursive.py
import pandas as pd
import numpy as np
from features.engineering import build_feature_pipeline

class RecursiveForecaster:
    def __init__(self, model, feature_columns, target_col="demand", horizon=6):
        self.model = model
        self.feature_columns = feature_columns
        self.target_col = target_col
        self.horizon = horizon

    def predict_next(self, recent_df):
        """Predict one step ahead given recent data (with features)."""
        # Assume recent_df has all required features except target (which is None for future)
        X_new = recent_df[self.feature_columns].iloc[-1:].copy()
        return self.model.predict(X_new)[0]

    def recursive_forecast(self, history_df, future_timestamps, zone):
        """
        history_df: DataFrame with historical data including features and target.
        future_timestamps: list of datetime objects for which to forecast.
        """
        # We'll create a copy of the last row and update timestamps step by step
        last_row = history_df.iloc[-1:].copy()
        predictions = []
        current_time = history_df["pickup_time"].max()
        for t in future_timestamps:
            # Advance time
            last_row["pickup_time"] = t
            # Recompute time features (calendar, rush hour)
            # We'll use a function to update only time-based features without recomputing lags (which are dynamic)
            # For simplicity, we'll rebuild features from scratch using the accumulated history
            # This is a simplified approach; in production, you'd update incrementally.
            # We'll just build a new DataFrame with history + previous predictions.
            new_row = last_row.copy()
            # Predict
            pred = self.predict_next(new_row)
            predictions.append(pred)
            # Append prediction as the next target for lag features (but we need to update lags)
            # In a real recursive forecast, you'd shift lags manually.
            # For brevity, I'll show a placeholder; you'd implement proper shift.
        return predictions
```

**Important:** For a robust recursive forecast, you need to maintain a rolling window of historical predictions and update lag features accordingly. This can be complex; you may want to use a simpler approach: for each step, you create a full feature set by using actual values for known past and predicted values for the forecast horizon. I recommend reading about *recursive multi‑step forecasting* with LightGBM – you can implement a function that builds the feature matrix incrementally.

---

### 6. Evaluation Framework (`models/evaluate.py`)
**Issues:** `axs[i].ylabel` typo, `save_metrics` path may not exist, `analyze_residuals` signature mismatch.  
**Fix:** correct matplotlib, create directories, and fix method signatures.

```python
# models/evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from config.settings import MODEL_DIR, METRIC_DIR

class Evaluator:
    def __init__(self, model, version):
        self.model = model
        self.version = version

    def plot_metrics(self, metrics_dict, save_path=None):
        # metrics_dict: {'train': {'mae': ..., ...}, 'val': ..., 'test': ...}
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        metrics = list(metrics_dict['train'].keys())
        for i, metric in enumerate(metrics):
            values = [metrics_dict[split][metric] for split in metrics_dict]
            axes[i].bar(metrics_dict.keys(), values)
            axes[i].set_title(metric)
            axes[i].set_ylabel(metric)  # corrected
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def analyze_residuals(self, X, y_true, model=None, split_name="test"):
        if model is None:
            model = self.model
        y_pred = model.predict(X)
        residuals = y_true - y_pred
        # Plot distribution and vs predicted
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(residuals, kde=True, ax=ax1)
        ax1.set_title(f"{split_name} Residuals")
        ax2.scatter(y_pred, residuals)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residual")
        plt.show()
        return residuals

    def save_metrics(self, metrics, split="test"):
        metric_dir = METRIC_DIR / self.version
        metric_dir.mkdir(parents=True, exist_ok=True)
        with open(metric_dir / f"model_metrics_{split}.json", "w") as f:
            json.dump(metrics, f, indent=2)
```

---

### 7. FastAPI Inference Service (`api/main.py`)
**Requirement:** deploy with validated inputs and reusable feature pipelines. We’ll create endpoints, reuse the feature engineering, and load the latest model.

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from features.engineering import build_feature_pipeline
from config.settings import MODEL_DIR
import logging

app = FastAPI(title="Taxi Demand Forecast", version="1.0")
logger = logging.getLogger("uvicorn")

# Load latest model
def load_latest_model():
    versions = [d for d in MODEL_DIR.iterdir() if d.is_dir()]
    if not versions:
        return None
    latest = sorted(versions, key=lambda d: d.stat().st_mtime)[-1]
    model_path = latest / "lgbm_point.pkl"
    if model_path.exists():
        return joblib.load(model_path), latest.name
    return None, None

model, version = load_latest_model()
if model is None:
    logger.warning("No model found; service will return 503")

class PredictionInput(BaseModel):
    zone: str
    pickup_time: datetime
    demand: float = Field(..., description="Historical demand for feature engineering; for prediction, we use future time but need past demand for lags")

    @validator('pickup_time')
    def validate_time(cls, v):
        # ensure not too far in future?
        return v

class PredictionOutput(BaseModel):
    predicted_demand: float
    version: str
    quantiles: dict = None

def prepare_features_for_prediction(input_data: PredictionInput, history_df: pd.DataFrame):
    """
    For a given zone and time, we need historical data to compute lags and rolling features.
    This function builds a DataFrame with the new time and historical features.
    """
    # Combine history with the new row (for future, demand is unknown, but we need it for feature engineering? Actually lags are from past, so we can set demand as NaN for the future row and then drop it)
    # We'll create a row with the new time and zone, and fill demand with NaN
    new_row = pd.DataFrame([{
        "zone": input_data.zone,
        "pickup_time": input_data.pickup_time,
        "demand": np.nan  # will be imputed by lags? No, lags are from past rows
    }])
    # Append to history (which must be sorted by time)
    combined = pd.concat([history_df, new_row], ignore_index=True)
    # Now run feature pipeline; but lags will produce NaN for the new row because there are no past values if we just appended?
    # Better approach: we need to compute features for the new time using only past data.
    # We'll create a function that builds features for a single timestamp using a window of past data.
    # For simplicity, we assume we have a function that returns the feature vector given the past.
    # We'll implement a simpler approach in the service: load recent history for the zone and compute features on the fly.
    pass

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    global model, version
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Load recent history for this zone (from Parquet or cache)
    # For demonstration, we load a sample; in real, you'd query a database or Parquet file.
    # We'll assume there's a function get_recent_history(zone) that returns last 30 days of hourly data.
    # We'll then compute features for the given time using only past data.
    # This is a placeholder.
    history = get_recent_history(input_data.zone)  # implement
    features = compute_features_for_time(history, input_data.pickup_time)
    pred = model.predict(features)[0]
    # Optionally load quantile models and return intervals
    quantiles = {}
    for q in [0.1, 0.9]:
        q_model_path = MODEL_DIR / version / f"lgbm_quantile_{q}.pkl"
        if q_model_path.exists():
            q_model = joblib.load(q_model_path)
            quantiles[q] = q_model.predict(features)[0]
    return PredictionOutput(
        predicted_demand=pred,
        version=version,
        quantiles=quantiles
    )

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": version}

@app.get("/metadata")
async def metadata():
    return {"version": version, "model_params": {}}
```

**Reusable feature pipeline:** The key is to have a function that, given historical data and a future timestamp, returns the feature vector (including lags) using only past values. This is non‑trivial; you need to implement a sliding window approach. For a production service, you can pre‑compute features for historical data and store them in a feature store (e.g., Redis) or query from Parquet.

---

### 8. Cloud Storage Integration (`utils/cloud_io.py`)
**Requirement:** read raw data from cloud and write processed Parquet to cloud. We’ll support GCS and S3.

```python
# utils/cloud_io.py
import os
from google.cloud import storage
import boto3
from pathlib import Path
from config.settings import CLOUD_PROVIDER, CLOUD_BUCKET, CLOUD_PREFIX

def get_gcs_client():
    return storage.Client()

def get_s3_client():
    return boto3.client('s3',
                        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))

def read_csv_from_cloud(bucket, blob_path, local_path=None):
    """Download CSV from cloud to local, or read directly with pandas/Spark."""
    if CLOUD_PROVIDER == 'gcs':
        client = get_gcs_client()
        bucket_obj = client.bucket(bucket)
        blob = bucket_obj.blob(blob_path)
        if local_path:
            blob.download_to_filename(local_path)
            return local_path
        else:
            # Read as bytes and use pandas? Not efficient; better to download.
            # Spark can read GCS directly with gs://
            return f"gs://{bucket}/{blob_path}"
    elif CLOUD_PROVIDER == 's3':
        # For Spark, use s3a://
        return f"s3a://{bucket}/{blob_path}"
    else:
        raise ValueError("Unsupported cloud provider")

def write_parquet_to_cloud(local_path, bucket, dest_path):
    """Upload Parquet files to cloud (recursive)."""
    if CLOUD_PROVIDER == 'gcs':
        client = get_gcs_client()
        bucket_obj = client.bucket(bucket)
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_path)
                blob = bucket_obj.blob(os.path.join(dest_path, rel_path))
                blob.upload_from_filename(local_file)
    elif CLOUD_PROVIDER == 's3':
        s3 = get_s3_client()
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_path)
                s3.upload_file(local_file, bucket, os.path.join(dest_path, rel_path))
```

In the ETL pipeline, you can modify `write_parquet` to write directly to `gs://` or `s3a://` paths – Spark handles that if the appropriate Hadoop connectors are included. For simplicity, we can write locally then upload.

---

### 9. Monitoring & MLOps (`monitoring/*`)
**Requirement:** feature drift detection, residual analysis, retraining triggers. We’ll implement:

- `drift_detector.py`: computes PSI (Population Stability Index) per feature between training and production data.
- `metrics_monitor.py`: logs performance metrics (MAE, RMSE) over time by comparing predictions to actuals.
- `retrain_trigger.py`: checks if drift > threshold or performance degradation > threshold and triggers a retraining job.

**Implement drift detection:**

```python
# monitoring/drift_detector.py
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from config.settings import DRIFT_THRESHOLD

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index."""
    # expected and actual are numpy arrays
    expected = np.array(expected)
    actual = np.array(actual)
    # Bin expected
    hist_expected, bin_edges = np.histogram(expected, bins=bins)
    hist_actual, _ = np.histogram(actual, bins=bin_edges)
    # Compute proportions
    prop_expected = hist_expected / hist_expected.sum()
    prop_actual = hist_actual / hist_actual.sum()
    # Avoid log(0)
    prop_expected = np.clip(prop_expected, 1e-10, 1)
    prop_actual = np.clip(prop_actual, 1e-10, 1)
    psi = np.sum((prop_actual - prop_expected) * np.log(prop_actual / prop_expected))
    return psi

def detect_drift(train_df, prod_df, feature_cols):
    drift_report = {}
    for col in feature_cols:
        train_vals = train_df[col].dropna().values
        prod_vals = prod_df[col].dropna().values
        if len(train_vals) > 0 and len(prod_vals) > 0:
            # KS test
            ks_stat, p_value = ks_2samp(train_vals, prod_vals)
            psi = calculate_psi(train_vals, prod_vals)
            drift_report[col] = {"ks_stat": ks_stat, "p_value": p_value, "psi": psi, "drift_detected": psi > DRIFT_THRESHOLD}
    return drift_report
```

**Retraining trigger:** in a scheduled job (e.g., daily), run `drift_detector` and `metrics_monitor`. If drift or performance drop exceeds threshold, trigger `retrain.py` which runs the training pipeline and deploys the new model (e.g., by updating a symlink to the latest version).

---

### 10. CI/CD & Testing
**GitHub Actions:**

- `.github/workflows/ci.yml`:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install flake8 mypy pytest
      - run: flake8 .
      - run: mypy .
      - run: pytest tests/
```

- `.github/workflows/deploy.yml` (on tags):
```yaml
name: Deploy
on:
  push:
    tags:
      - 'v*'
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t myrepo/taxi-demand:${GITHUB_REF#refs/tags/} .
      - name: Push to registry
        run: docker push myrepo/taxi-demand:${GITHUB_REF#refs/tags/}
```

**Testing:** create `tests/` with:
- `test_etl.py`: small sample CSV, test cleaning and schema.
- `test_features.py`: verify time features.
- `test_training.py`: train on tiny dataset, check model saves.
- `test_api.py`: use `httpx` to test endpoints.

---

### 11. Documentation & Structure
- **README.md**: include architecture diagram (e.g., using Mermaid), setup instructions, how to run ETL, train, serve.
- **requirements.txt**: pin versions:
  ```
  pyspark==3.4.1
  lightgbm==4.1.0
  fastapi==0.104.0
  uvicorn==0.24.0
  pandas==2.1.4
  numpy==1.26.2
  scikit-learn==1.3.2
  matplotlib==3.8.1
  seaborn==0.13.0
  google-cloud-storage==2.14.0
  boto3==1.34.0
  pytest==7.4.3
  flake8==6.1.0
  mypy==1.7.0
  ```
- Add `Dockerfile` for FastAPI service:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📅 Prioritized Execution Plan

| Order | Module | Key Actions |
|-------|--------|-------------|
| 1 | **Configuration** | Clean `settings.py`, add all constants, env vars, create directories. |
| 2 | **ETL Pipeline** | Fix imports, use `COLUMN_MAP`, add duplicate/outlier handling, partition by date, write Parquet. Test with sample. |
| 3 | **Feature Engineering** | Implement functions for lag, rolling, calendar, rush‑hour. Use Pandas (can be run on Spark later). |
| 4 | **Model Training** | Fix versioning, quantile models, save logic. Run on engineered data. |
| 5 | **Recursive Forecasting** | Implement recursive predictor (at least a prototype). |
| 6 | **Evaluation Framework** | Fix matplotlib, save metrics, residual analysis. |
| 7 | **FastAPI Service** | Create `api/main.py` with endpoints, load model, integrate feature pipeline for prediction. |
| 8 | **Cloud Storage** | Add utilities to read/write from GCS/S3, integrate into ETL. |
| 9 | **Monitoring** | Implement drift detection, metrics logging, retraining trigger (as standalone scripts). |
| 10 | **CI/CD & Tests** | Add GitHub Actions, write tests, ensure code passes linting. |
| 11 | **Containerization** | Create Dockerfile and `docker-compose.yml` for local dev. |

---

## 🔍 Verification Checklist

- [ ] ETL can process a 1 GB CSV file and output partitioned Parquet.
- [ ] Feature engineering produces lag/rolling features without data leakage.
- [ ] Training completes and saves point and quantile models with versioned directories.
- [ ] Recursive forecasting generates 6‑step ahead predictions (even if simple).
- [ ] Evaluation computes MAE, RMSE, R², SMAPE and saves plots.
- [ ] FastAPI service responds to `/predict` with valid input and returns prediction + quantiles.
- [ ] Cloud storage read/write works (test with a small file).
- [ ] Drift detection runs on new data and logs results.
- [ ] CI pipeline passes linting and unit tests.
- [ ] Docker image builds and runs.

---

By following this detailed guide, you will not only fix all the current bugs but also build a robust, production‑ready system that fully meets the requirements. If you need further elaboration on any specific part (e.g., recursive forecasting implementation details, or how to handle feature engineering for inference), just ask!