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

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)





class Evaluate:
    def __init__(self):
        self.metrics={}
        self.baseline_metrics={}


    def compute_metrics(self,y_true, y_pred, label=''):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        smape=self.smape(y_true, y_pred)
        
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  SMAPE:   {smape:.4f}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2,'smape':smape}
    



    def baseline_naive_seasonal(self,df:pd.DataFrame,split:str=""):

        preds = df['lag_168h']
        metrics=self.compute_metrics(df['demand'], preds, label='Naive Seasonal Baseline')
        self.baseline_metrics[split]=metrics
        
        return metrics
    
    
    
    
    def smape(y_true, y_pred, epsilon=1e-10): 
        y_true = np.array(y_true) 
        y_pred = np.array(y_pred) 
        numerator = 2 * np.abs(y_true - y_pred) 
        denominator = np.abs(y_true) + np.abs(y_pred) + epsilon 
        smape_value = np.mean(numerator / denominator) * 100 
        return smape_value
    



    def evaluate_model(self,df:pd.DataFrame,model,split:str=""):
        y_true = df['demand']
        y_pred = model.predict(df[FEATURE_COLUMNS])
        y_pred = pd.clip(y_pred, min=0)  
        print(f"evaluation for {split}")
        baseline=self.baseline[split]
        # Global metrics
        MAE  = mean_absolute_error(y_true, y_pred)
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        SMAPE=self.smape(y_true,y_pred)
        model_metrics={'mae': MAE, 'rmse': RMSE, 'r2': r2,'smape':SMAPE}
        self.metrics[split]=model_metrics
        
         # Compare to baseline
        print(f"Model MAE: {MAE:.2f} | Baseline MAE: {baseline['mae']:.2f}")
        print(f"Model RMSE: {RMSE:.2f} | Baseline RMSE: {baseline['rmse']:.2f}")
        print(f"Model r2: {r2:.2f} | Baseline RMSE: {baseline['r2']:.2f}")
        print(f"Model SMAPE: {SMAPE:.2f} | Baseline SMAPE: {baseline['smape']:.2f}")
        
        
        
        
    def save_metrics(self):
        with open(MODEL_DIR/"model_metrics.json",'w') as f:
            json.dump(self.metrics,f)
        
        
        
        
        
        
        
    def analyze_residuals(self,df:pd.DataFrame,split_name:str):
    
        X = df[FEATURE_COLUMNS]
        y_true = df['demand']
        y_pred = self.model.predict(X)
        
        
        residuals = y_true - y_pred
        
        # By hour
        df['residual'] = residuals
        hourly_error = df.groupby('hour_of_day')['residual'].agg(['mean', 'std'])
        
        print(f"\n{'='*60}")
        print(f"  Error by Hour of Day({split_name})")
        print(f"{'='*60}")
        print(hourly_error)
        
        # By zone (find worst zones)
        zone_error = df.groupby('zone_id').agg({
            'residual': ['mean', 'std', 'count']
        }).round(2)
        zone_error.columns = ['mean_error', 'std_error', 'count']
        zone_error = zone_error.sort_values('mean_error', key=abs, ascending=False)
        
        print(f"\n{'='*60}")
        print(f"  Top 10 Worst Zones (Highest Error)")
        print(f"{'='*60}")
        print(zone_error.head(10))


        
        
        
        
        
        
        
        
        
        
        
        
        
        
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
        plot residuals (y_true - y_pred) over time (should be stationary, no drift)