from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from monitoring.prediction_logger import prediction_logger
from config.settings import LOG
from pathlib import Path
from datetime import datetime,timedelta
from models.evaluate import Evaluate





class metrics_monitor:
    
    
    
    def __init__(self):
        self.metrics
        
        

        
    def load_logs(self):
        self.pred_log=prediction_logger()
        self.log_path=Path(LOG)
        try:
            
            self.log_df=self.pred_log.load_prediciton_logs(self.log_path)
        
        except FileNotFoundError:
           
            exit()
            return("add logs or update log location")
            
    



    def compute_metrics(self,y_true,y_pred):
        self.evaluator=Evaluate()
        self.metrics=self.evaluator.compute_metrics(y_true,y_pred)
    
    
        

    def compute_rolling_metrics(self):
        #find a way to take timestamp,find date and calculate metrics
        
        timestamp=self.log_df["timestamp"]
        if  not pd.api.types.is_datetime64_any_dtype(timestamp):
            timestamp=pd.to_datetime(self.log_df["timestamp"])

        
        now=datetime.now()
        window_24h=now - timedelta(days=1)
        window_week=now - timedelta(days=7)

        df_24h=self.log_df[self.log_df[timestamp]>window_24h]
        df_week=self.log_df[self.log_df[timestamp]>window_week]

        metrics_24=self.compute_metrics(df_24h["actual"],df_24h["prediction"])
        metrics_week=self.compute_metrics(df_week["actual"],df_week["prediction"])
        

        

        
        pass
        
    def compare_with_baseline():
        #find how you can calculate the baseline
        pass