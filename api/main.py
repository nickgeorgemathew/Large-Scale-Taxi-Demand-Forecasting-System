from fastapi import FastAPI,httpexception,Query
from predictor import TaxiForecaster
from config.settings import VALID_ZONE_MIN, VALID_ZONE_MAX
import numpy as np



app = FastAPI(title="NYC Taxi Demand Forecast API")
forecaster = TaxiForecaster()
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
def get_hotspots(timestamp):
    results=[]
    for zones in valid_zone:
      forecaster.predict_hotspots(zones,timestamp)
      results.append(forecaster.predict_hotspots(zones,timestamp))
    return results


@app.get('/health')
def get_health():
    

 
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