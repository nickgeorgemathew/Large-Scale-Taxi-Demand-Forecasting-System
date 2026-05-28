from fastapi import FastAPI, Query
import pandas as pd
import json
from pathlib import Path
from predictor import TaxiForecaster
from config.settings import (
    MODEL_PATH, QUANTILE_LOW_MODEL_PATH, QUANTILE_HIGH_MODEL_PATH,
    FEATURES_PATH, RECENT_HISTORY, LOG, METRICLOG, SERVING_HALTED,
    VALID_ZONE_MIN, VALID_ZONE_MAX, HOTSPOTS
)
from monitoring.prediction_logger import PredictionLogger
from monitoring.metrics_monitor import MetricsMonitor
from monitoring.alert_manager import AlertManager
from monitoring.model_registry_manager import ModelRegistry

app = FastAPI(title="NYC Taxi Demand Forecast API")

# Initialize components
forecaster = TaxiForecaster(
    model_path=MODEL_PATH,
    feature_path=FEATURES_PATH,
    recent_history_path=RECENT_HISTORY,
    quantile_low_model_path=QUANTILE_LOW_MODEL_PATH,
    quantile_high_model_path=QUANTILE_HIGH_MODEL_PATH
)
logger = PredictionLogger()
monitor = MetricsMonitor()
alert_mgr = AlertManager()
registry = ModelRegistry()
valid_zone = list(range(VALID_ZONE_MIN, VALID_ZONE_MAX + 1))

@app.get('/')
def root():
    return {"message": "NYC Taxi Demand Forecast API"}

@app.get('/forecast/{zone_id}/{hours_ahead}')
def predict(zone_id: int, hours_ahead: int = Query(24, gt=0, le=72)):
    # Check if serving is halted
    halted_path = Path(SERVING_HALTED)
    if halted_path.exists():
        with open(halted_path, 'r') as f:
            flag = json.load(f)
        if flag.get("serving_halted", False):
            return {"serving": False, "halted_at": flag.get("halted_at"), "reason": flag.get("reason")}

    if zone_id not in valid_zone:
        return {"error": f"zone_id must be between {VALID_ZONE_MIN} and {VALID_ZONE_MAX}"}

    result = forecaster.predict(zone_id, hours_ahead)
    return [r.dict() for r in result]

@app.get('/hotspots/{timestamp}')
def get_hotspots(timestamp: str = None, top_n: int = 20):
    if timestamp is None:
        timestamp = pd.Timestamp.now().floor('h')
    else:
        timestamp = pd.to_datetime(timestamp).floor('h')
    all_zones = forecaster.build_feature_rows_all_zones(timestamp, valid_zone)
    result = forecaster.predict_hotspots(all_zones, timestamp)
    result = result.sort_values("predicted_demand", ascending=False)
    logger.save_logs_parquet(result.head(top_n).to_dict(orient="records"), HOTSPOTS)
    return result.head(top_n).to_dict(orient="records")

@app.get('/health')
def get_health():
    # Return the latest model condition log
    metric_path = Path(METRICLOG)
    if metric_path.exists():
        df = pd.read_parquet(metric_path)
        if not df.empty:
            latest = df.sort_values("timestamp", ascending=False).iloc[0].to_dict()
            return latest
    return {"status": "unknown", "message": "No health logs found"}