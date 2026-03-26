import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path
import numpy as np
import time
import  joblib
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS
)



class evaluate:
    def __init__(self):
        self.metrics={}
    
    def evaluate_model(model, test_df, baseline_preds):
    
        y_true = test_df['demand']
        y_pred = model.predict(test_df[FEATURE_COLUMNS])
        y_pred = pd.clip(y_pred, min=0)  
        
        # Global metrics
        MAE  = mean_absolute_error(y_true, y_pred)
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        
        
        # Compare to baseline
        print(f"Model MAE: {MAE:.2f} | Baseline MAE: {baseline_MAE:.2f}")
        
        # Per-zone breakdown (critical for understanding where model fails)
        zone_errors = test_df.copy()
        abs=y_true - y_pred
        zone_errors['abs_error'] = -abs if abs <0 else abs
        per_zone = zone_errors.groupby('zone_id')['abs_error'].mean()
        
        # Identify: worst 10 zones → likely outlier zones (airports, stadiums)
        # These are good candidates for a separate specialized model
        
        # Per-hour breakdown
        zone_errors['hour'] = test_df['hour_of_day']
        per_hour = zone_errors.groupby('hour')['abs_error'].mean()
        plot per_hour → expect error peaks at 6-7am and 5-6pm (transition times)
        
        # Residuals over time
        plot y_pred vs y_true scatter (should hug diagonal)
        plot residuals (y_true - y_pred) over time (should be stationary, no drift)