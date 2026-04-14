class TaxiForecaster:
    
    __init__:
      load model from artifacts/lgbm_demand_v1.pkl
      load feature_cols from artifacts/feature_cols.json
      load recent_history_df from processed data  ← needed to build lag features
      load zone_metadata_df
    
    predict(zone_id, hours_ahead):
      results = []
      history_buffer = recent_history_df[zone_id].copy()
      
      current_hour = datetime.now().floor('H')
      
      for step in range(1, hours_ahead + 1):
        target_time = current_hour + timedelta(hours=step)
        
        # build one row of features for this (zone, time) pair
        features = build_feature_row(zone_id, target_time, history_buffer)
        
        pred = model.predict(features)[0]
        pred = max(0, pred)   ← clip negative
        
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
        history_buffer.append({
          'hour_timestamp': target_time,
          'demand': pred
        })
      
      return results
    
    build_feature_row(zone_id, timestamp, history_buffer):
      row = {}
      row['hour_of_day'] = timestamp.hour
      row['day_of_week'] = timestamp.weekday()
      row['is_weekend'] = row['day_of_week'] >= 5
      row['zone_id'] = zone_id
      # ...other temporal features
      
      # lags: pull from history_buffer
      for lag in [1, 2, 3, 6, 24, 48, 168]:
        lookup_time = timestamp - timedelta(hours=lag)
        row[f'lag_{lag}h'] = history_buffer.get(lookup_time, 0)
      
      # rolling: compute from history_buffer
      last_6 = [history_buffer.get(timestamp - timedelta(hours=i), 0) for i in range(1,7)]
      row['roll_mean_6h'] = mean(last_6)
      
      return pd.DataFrame([row])[feature_cols]
```