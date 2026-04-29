from fastapi import FastAPI,httpexception,Query
from predictor import TaxiForecaster
from config.settings import VALID_ZONE_MIN, VALID_ZONE_MAX,MODEL_PATH,QUANTILE_LOW_MODEL_PATH,RECENT_HISTORY,QUANTILE_HIGH_MODEL_PATH,HOTSPOTS,FEATURES_PATH,LOG
import numpy as np
import pandas as pd
from monitoring.prediction_logger import PredictionLogger
from monitoring.metrics_monitor import MetricsMonitor


app = FastAPI(title="NYC Taxi Demand Forecast API")
forecaster = TaxiForecaster(MODEL_PATH,FEATURES_PATH,RECENT_HISTORY,QUANTILE_LOW_MODEL_PATH,QUANTILE_HIGH_MODEL_PATH,)
logs=PredictionLogger()
metric=MetricsMonitor()
valid_zone=np.arange(VALID_ZONE_MIN,VALID_ZONE_MAX)   
  

@app.get('/')
def root():
  return("NYC Taxi Demand Forecast API")


@app.get('/forecast/{zone_id}/{hours_ahead}')
def predict(zone_id,hours_ahead:int=Query(24,gt=0)):
  
  if zone_id not in valid_zone:
    return(f"zone id entered is not valid,please enter zone_id between{VALID_ZONE_MIN} and {VALID_ZONE_MAX}")

    
  else:
    result=forecaster.predict(zone_id,hours_ahead)
    logs.save_logs_parquet(result,HOTSPOTS)
    return result.to_dict(orient="records")
  
  

@app.get('/hotspots/{timestamp}')
def get_hotspots(timestamp,top_n: int = 20):
    if timestamp is None:
      timestamp=pd.Timestamp.now().floor('h')
    else:
      timestamp=pd.to_datetime(timestamp).floor('h')
    all_zones=forecaster.build_feature_rows_all_zones(timestamp,valid_zone)
    result=forecaster.predict_hotspots(all_zones,timestamp)
    result = result.sort_values(
        "predicted_demand",
        ascending=False
    )
    logs.save_logs_parquet(result,HOTSPOTS)

    return result.head(top_n).to_dict(orient="records")



@app.get('/health')
def get_health():
 return metric.detect_performance_drift()

  
  
  
  
  
  
# GET PREVIOUS PREDICTIONS
# GET MODEL PERFORMANCE OR RETRAINING
