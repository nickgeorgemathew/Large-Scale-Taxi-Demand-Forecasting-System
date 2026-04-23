import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import pyspark.pandas as ps
from datetime import datetime,timedelta
from schema import HourlyForecast
import joblib
import json
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    PUBLIC_HOLIDAYS_2022,
    TRAIN_END_DATE, VAL_END_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS
)


class TaxiForecaster:
    
    def __init__(self,model_path,feature_path,recent_history_path):
      self.model=joblib.load(model_path)

      with open(feature_path,"r") as F:
        self.feature_cols=json.load(F)
      self.recent_history_df=ps.read_csv(recent_history_path)#load recent_history_df from processed data  ← needed to build lag features
      # load zone_metadata_df
    
    def predict(self,zone_id, hours_ahead):
      """predict future demand.pass zone_id and how many hours ahead prediction should be done"""
      results = []
      history_buffer = self.recent_history_df[self.recent_history_df["zone_id"] == zone_id].copy()
      
      current_hour = pd.Timestamp.now().floor('h')
      
      for step in range(1, hours_ahead + 1):
        target_time = current_hour + timedelta(hours=step)
        
        # build one row of features for this (zone, time) pair
        features = self.build_feature_row(zone_id, target_time, history_buffer)
        
        pred = self.model.predict(features)[0]
        pred = np.clip(pred,0)  
        
        pred_low  = quantile_low_model.predict(features)[0]
        pred_high = quantile_high_model.predict(features)[0]
        
        results.append(HourlyForecast(
          timestamp=target_time,
          predicted_demand=pred,
          lower_bound=pred_low,
          upper_bound=pred_high
        ))
        
        # RECURSIVE FORECASTING: add this prediction to buffer
        # so next step's lag_1h uses it
        history_buffer=history_buffer.append({
          'timestamp': target_time,
          'demand': pred
        })
      
      return results
    
        
    def build_feature_row(self,zone_id, timestamp, history_buffer):
        row = {}
        history_buffer=history_buffer.copy()
        timestamp=pd.to_datetime(timestamp).floor('h')
        history_buffer['timestamp']=pd.to_datetime(history_buffer["timestamp"]).floor('h')
        row['hour_of_day'] = timestamp.hour
        row['day_of_week'] = timestamp.weekday()
        row['zone_id'] = zone_id
        row["month"]=timestamp.month
        row["is_weekend"]=(row["day_of_week"]>=5).astype(int)
        row["is_rush_am"]=int(row["hour_of_day"]>=7 and row["hour_of_day"]<=9)
        row["is_rush_pm"]=int(row["hour_of_day"]>=17 and row["hour_of_day"]<=19)
        row["is_night"]=int( (row["hour_of_day"] >= 22) or (row["hour_of_day"] <= 5)
        )

        holidays=pd.to_datetime(PUBLIC_HOLIDAYS_2022)
        row["is_holiday"]=int(pd.to_datetime(timestamp.date()) in holidays)
        # ...other temporal features
        
        # lags: pull from history_buffer
        lag_list = [1, 2, 3, 6, 24, 48, 168]

        for lag in lag_list:

            lookup_time = timestamp - timedelta(hours=lag)

            match = history_buffer.loc[
                history_buffer["timestamp"] == lookup_time,
                "demand"
            ]

            if len(match) > 0:
                row[f"lag_{lag}h"] = float(match.iloc[-1])
            else:
                row[f"lag_{lag}h"] = 0.0
        
        # rolling: compute from history_buffer
        values = []

        for i in range(1, 7):

            lookup_time = timestamp - timedelta(hours=i)

            match = history_buffer.loc[
                history_buffer["timestamp"] == lookup_time,
                "demand"
            ]

            if len(match) > 0:
                values.append(float(match.iloc[-1]))

        row["roll_mean_6h"] = np.mean(values) if values else 0.0
        
        row_df = pd.DataFrame([row])
        row_df = row_df.reindex(columns=self.feature_cols, fill_value=0)

        return row_df
