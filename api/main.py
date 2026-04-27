from fastapi import FastAPI,httpexception,Query
from predictor import TaxiForecaster
from config.settings import VALID_ZONE_MIN, VALID_ZONE_MAX,MODEL_PATH,QUANTILE_LOW_MODEL_PATH,RECENT_HISTORY,QUANTILE_HIGH_MODEL_PATH,HOTSPOTS,FEATURES_PATH
import numpy as np
import pandas as pd
from monitoring import prediction_logger


app = FastAPI(title="NYC Taxi Demand Forecast API")
forecaster = TaxiForecaster(MODEL_PATH,FEATURES_PATH,RECENT_HISTORY,QUANTILE_LOW_MODEL_PATH,QUANTILE_HIGH_MODEL_PATH,)
logs=prediction_logger()
valid_zone=np.arange(VALID_ZONE_MIN,VALID_ZONE_MAX)   
  

@app.get('/')
def root():
  return("NYC Taxi Demand Forecast API")


@app.get('/forecast/{zone_id}/{hours_ahead}')
def predict(zone_id,hours_ahead:int=Query(24,gt=0)):
  
  if zone_id not in valid_zone:
    return(f"zone id entered is not valid,please enter zone_id between{VALID_ZONE_MIN} and {VALID_ZONE_MAX}")

    
  else:
    return forecaster.predict(zone_id,hours_ahead)
  

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

    return result.head(top_n).to_dict(orient="records")



@app.get('/health')
def get_health():
  #get this from the mlops pipeline


 
  GET /forecast/{zone_id}?hours_ahead=24
    → validate zone_id (1-263)
    → call forecaster.predict(zone_id, hours_ahead)
    → return ForecastResponse
  
  GET /hotspots?timestamp=2024-01-06T18:00
    → for all 263 zones, get predicted demand at that timestamp
    → join with zone lat/lng centroids
    → return sorted list of top N zones by demand
    → this is what feeds the map
  
  GET /health
    → return {"status": "ok", "model_version": "v1"}
  
  
  
  GET PREVIOUS PREDICTIONS
  GET MODEL PERFORMANCE OR RETRAINING
  
  
  # Test your API with: uvicorn api.main:app --reload
  # Then visit localhost:8000/docs for auto-generated Swagger UI