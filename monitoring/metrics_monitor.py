from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from monitoring.prediction_logger import prediction_logger
from config.settings import LOG
from pathlib import Path
from datetime import datetime,timedelta
from models.evaluate import Evaluate
import json





class MetricsMonitor:
    
    
    
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
            
    



    def compute_metrics(self,y_true,y_pred,label:str=""):

        self.evaluator=Evaluate()
        return self.evaluator.compute_metrics(y_true,y_pred,label)
    
    
    #SAVING FILES IN JSON FOR NOW,IF MORE EFFICIENT MOVE TO PARQUET

    def compute_rolling_metrics(self):

        #find a way to take timestamp,find date and calculate metrics
        
        timestamp=self.log_df["timestamp"]
        if  not pd.api.types.is_datetime64_any_dtype(timestamp):
            self.log_df["timestamp"]=pd.to_datetime(self.log_df["timestamp"])

        
        current_time=datetime.now()
        window_24h = current_time - timedelta(hours=24)
        window_week = current_time - timedelta(days=7)

        df_24h = self.log_df[self.log_df["timestamp"]>window_24h]
        df_24h=df_24h['actual'].dropna()

       
        if df_24h.shape()[0] <10:
            
            return(" not enough predictions in 24 hours metrics too unreliable")
        else:
            
            metrics_24=self.compute_metrics(df_24h["actual"],df_24h["prediction"],label=f"window_24h_{window_24h}")
             
            #check file path adding using config
            with open('model/artifacts/metric_history_24h.json','w') as f:
                json.dump(metrics_24,f)
        
        
        
        
        df_week = self.log_df[self.log_df["timestamp"]>window_week]
        metrics_week=self.compute_metrics(df_week["actual"],df_week["prediction"],label=f"window_week_{window_week}")
        
        #check file path adding using config
        with open('model/artifacts/metric_history_week.json','w') as f:
            json.dump(metrics_week,f)

        
        return metrics_week,metrics_24




        
    def compare_with_baseline(self):
        #find how you can calculate the baseline

        with open('model/artifacts/model_metrics_test.json','r') as f:
            baseline=json.load(f)
        
        baseline=pd.DataFrame(baseline)
        # how to calculate current metric
        
        metric_names = ["mae", "rmse", "r2", "smape"]
        flags={"mae_degrade":False, "rmse_degraded":False, "r2_degraded":False, "smape_degraded":False}
        
        for metric in metric_names:
            
            if self.metrics[metric]>baseline[metric]:
                
                print(f" current {metric}:{self.metrics[metric]} vs baseline {metric}:{self.metrics[metric]}")
                #check what to print
                #add retraining triggers
                print("retrain /fix model")
                flags[metric]=True
            
            else:
                
                print(f" current {metric}:{self.metrics[metric]} vs baseline {metric}:{self.metrics[metric]}")
                #check what to print
                print("model still usable")
        
        return flags


        