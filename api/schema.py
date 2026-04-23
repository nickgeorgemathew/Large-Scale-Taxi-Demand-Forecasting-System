
from pydantic import BaseModel,Field
from datetime import datetime
from typing import List

class ForecastRequest(BaseModel):
    zone_id: int= Field(ge=1,le=263)
    hours_ahead: int=Field(default=24,ge=1 ,le=72)
  
class HourlyForecast(BaseModel):
    timestamp: datetime
    zone_id:int
    predicted_demand: float
    lower_bound: float
    upper_bound: float
    features:dict
  
class ForecastResponse(BaseModel):
    zone_id: int
    zone_name: str
    borough: str
    forecasts: list[HourlyForecast]
    model_version: str
    generated_at: datetime
  

class Hotspot(BaseModel):
    zone_id: int
    demand: float
    lat: float
    lng: float
    zone_name: str


class HotspotResponse(BaseModel):
    timestamp: datetime
    zones: list[Hotspot]
