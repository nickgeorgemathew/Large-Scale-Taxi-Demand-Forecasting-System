from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from monitoring.prediction_logger import prediction_logger
from config.settings import LOG
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import pyspark.pandas as ps
from datetime import datetime,timedelta
from models.evaluate import Evaluate
import json
from config.settings import (METRICLOG)



class MetricsMonitor:
    
    
    
    def __init__(self):
        self.metrics
        
        

        
    def load_logs(self):
        self.pred_log=prediction_logger()
        self.log_path=Path(LOG)
        self.metric_path=Path(METRICLOG)
        try:
            
            self.log_df=self.pred_log.load_prediciton_logs(self.log_path)
        
        except FileNotFoundError:
           
            exit()
            return("add logs or update log location")
            
    

    def save_metrics_parquet(self,metrics,label,filename):
        df_metrics=pd.DataFrame(metrics)        
        # Load, Combine, and Overwrite
        try:    
            df_existing = pd.read_parquet(self.metric_path[filename])

            # perform a 'merge' with an 'indicator' to see which rows are new
            check_merge = pd.merge(
                df_metrics, 
                df_existing[[label]], 
                on=[label], 
                how='left', 
                indicator=True
            )
            # Check if any row in df_logs was found in df_existing
            is_duplicate = (check_merge['_merge'] == 'both').any()
            
            if is_duplicate:
                
                return "duplicate predictions"
            else:
                df_combined = pd.concat([df_existing, df_metrics], ignore_index=True)
                df_combined.to_parquet(self.log_path, engine='pyarrow')

        except FileNotFoundError:

            df_metrics.to_parquet(self.log_path,engine="pyarrow") 

        return df_metrics
    





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
            
            return(" not enough predictions in 24 hours, metrics too unreliable")
        else:
            
            metrics_24=self.compute_metrics(df_24h["actual"],df_24h["prediction"],label=f"window_24h_{window_24h}")
             
            #check file path adding using config
            with open('model/artifacts/metric_history_24h.json','w') as f:
                json.dump(metrics_24,f)
            self.save_metrics_parquet(metrics_24,label=f"window_24h_{window_24h}",file_name='metric_history_24h')
        
        
        
        
        df_week = self.log_df[self.log_df["timestamp"]>window_week]
        metrics_week=self.compute_metrics(df_week["actual"],df_week["prediction"],label=f"window_week_{window_week}")
        
        #check file path adding using config
        with open('model/artifacts/metric_history_week.json','w') as f:
            json.dump(metrics_week,f)
        self.save_metrics_parquet(metrics_week,label=f"window_week_{window_week}",file_name='metric_history_week')

        
        return metrics_week,metrics_24




        
    def compare_with_baseline(self,current_true,current_pred):
        #find how you can calculate the baseline

        with open('model/artifacts/model_metrics_test.json','r') as f:
            baseline=json.load(f)
        
        baseline=pd.DataFrame(baseline)
        # how to calculate current metric

        self.metrics= self.compute_metrics(current_true,current_pred)
        
        metric_names = ["mae", "rmse", "r2", "smape"]
        performance_flags={"mae_degrade":False, "rmse_degraded":False, "r2_degraded":False, "smape_degraded":False}
        
        for metric in metric_names:
            
            if self.metrics[metric]>baseline[metric]:
                
                print(f" current {metric}:{self.metrics[metric]} vs baseline {metric}:{baseline[metric]}")
                #check what to print
                #add retraining triggers
                print("retrain /fix model")
                performance_flags[metric]=True
            
            else:
                
                print(f" current {metric}:{self.metrics[metric]} vs baseline {metric}:{baseline[metric]}")
                #check what to print
                print("model still usable")
        
        return performance_flags
    
    
    
    
    def detect_performance_drift(self,filename,threshold:float=1.3):
        
        with open('model/artifacts/model_metrics_test.json','r') as f:
            baseline=json.load(f)
        
        
        training_metrics=pd.DataFrame(baseline)
        rolling_metrics = pd.read_parquet(self.metric_path[filename])
            
        
        metric_names = ["mae", "rmse", "r2", "smape"]
        performance_flags={"mae_drift":False, "rmse_drift":False, "r2_drift":False, "smape_drift":False}
        
        for metric in metric_names:
            
            if rolling_metrics[metric] > training_metrics[metric] * threshold:
                
                print(f" current {metric}:{rolling_metrics[metric]} vs baseline {metric}:{training_metrics[metric]}")
                performance_flags[metric]=True
            
            else:
                
                print(f" current {metric}:{rolling_metrics[metric]} vs baseline {metric}:{training_metrics[metric]}")
                #check what to print
                print("model still usable")
        
        return performance_flags


        