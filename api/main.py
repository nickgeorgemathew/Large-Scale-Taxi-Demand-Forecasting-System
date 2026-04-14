app = FastAPI(title="NYC Taxi Demand Forecast API")
  forecaster = TaxiForecaster()   ← loaded once at startup, not per request
  
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
  
  # Test your API with: uvicorn api.main:app --reload
  # Then visit localhost:8000/docs for auto-generated Swagger UI