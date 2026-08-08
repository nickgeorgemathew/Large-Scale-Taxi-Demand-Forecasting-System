import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from config.settings import LOG, BASELINE_METRICS_PATH
import json

# --- 1. Fake prediction logs, matching PredictionLogger's exact schema ---
n_rows = 500
zones = np.random.randint(1, 264, size=n_rows)
now = datetime.now()
timestamps = [now - timedelta(hours=np.random.uniform(0, 200)) for _ in range(n_rows)]

actual = np.random.poisson(lam=15, size=n_rows).astype(float)
# start with predictions close to actual (healthy model)
prediction = actual + np.random.normal(0, 2, size=n_rows)
residual = actual - prediction

fake_logs = pd.DataFrame({
    "timestamp": timestamps,
    "model_version": "v1",
    "residual": residual,
    "zone_id": zones,
    "prediction": prediction,
    "actual": actual,
})

Path(LOG).parent.mkdir(parents=True, exist_ok=True)
fake_logs.to_parquet(LOG, index=False)
print(f"Wrote {len(fake_logs)} fake prediction logs to {LOG}")

# --- 2. Fake baseline metrics (what detect_performance_drift compares against) ---
baseline = {"mae": 2.1, "rmse": 2.8, "r2": 0.85, "smape": 12.0}
Path(BASELINE_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(BASELINE_METRICS_PATH, "w") as f:
    json.dump(baseline, f, indent=2)
print(f"Wrote fake baseline metrics to {BASELINE_METRICS_PATH}")