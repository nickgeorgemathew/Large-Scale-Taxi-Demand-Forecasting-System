import pandas as pd
from monitoring.prediction_logger import prediction_logger
from pathlib import Path
from datetime import datetime,timedelta
from metrics_monitor import MetricsMonitor
from models.evaluate import Evaluate
import json
import numpy as np
from retraining import retrain_pipeline
from config.settings import METRICLOG,LOG,PERFORMANCELOG
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,MONITORED_FEATURES,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS,
SPARK_APP_NAME, SPARK_SHUFFLE_PARTITIONS, SPARK_DRIVER_MEMORY)


def save_log_parquet(log,file_path):
        df_log=pd.DataFrame([log])        
        # Load, Combine, and Overwrite
        try:
            df_existing = pd.read_parquet(Path(file_path))
            df_combined = pd.concat([df_existing, df_log], ignore_index=True)
            df_combined.to_parquet(Path(file_path), engine='pyarrow')
        # create new file if it doesnot exist
        except FileNotFoundError:
            df_log.to_parquet(Path(file_path),engine="pyarrow") 

        return df_log




class AlertManager:
    

    def __init__(self):
        pass
        

    
    
    
    
    def log_state(self,flag,file_path):
        """ log the issue and severity with what action was taken with timestamp"""
        current_time=datetime.now()
        log={"timestamp":current_time,"severity":flag["severity"],"action":flag["action"]}
        save_log_parquet(log,file_path)
     
    
    
    def model_condition_log(self,flag,file_path):
        """ log the model condition and process what action was taken towards the model"""
        current_time=datetime.now()
        log={"timestamp":current_time,"severity":flag["severity"],"action triggered":flag["action"],"model state":flag["model_state"]}
        save_log_parquet(log,file_path)
        return log



    
    def assess_perfomance(self,performance_flags):
        """ process the performance flags"""
        performance=[k for k,val in performance_flags.items() if val]
        return performance
    
    def asses_drift(self,drift_flags):
        """ assess the feature and residual drift flags"""
        feature=[k for k,val in drift_flags.items() if val and k !="residual_drift"]
        residual=[k for k,val in drift_flags.items() if val and k =="residual_drift"]
        return feature,residual
        
    
    
    def assess_condition(self,performance_flags,drift_flags):
        """ decision engine that gives severity and action to take"""
        performance=self.assess_perfomance(performance_flags)
        feature,residual=self.asses_drift(drift_flags)
        
        condtion_flags={"severity":"","action":""}

        if performance and feature and residual:
            condtion_flags.update(severity="CRITICAL",action="RETRAIN MODEL IMMEDIATELY ! ROLLBACK")

        
        elif not performance and not feature and not residual:
            condtion_flags.update(severity="OK",action="NONE")
        
        
        elif performance and feature:
            condtion_flags.update(severity="RETRAIN",action="RETRAIN MODEL, PERFORMANCE AND FEATURE DRIFT DETECTED")

        
        elif feature and residual:
            condtion_flags.update(severity="WARNING",action="MONITOR FEATURES AND RESIDUAL")

        
        elif performance and residual:
            condtion_flags.update(severity="CRITICAL",action="INVESTIGATE AND RETRAIN (performance and residual) ")
        
        
        elif performance:
            condtion_flags.update(severity="WARNING",action="INVESTIGATE PERFORMANCE")
        
        elif feature:
            condtion_flags.update(severity="WATCH",action="MONITOR FEATURES")
        
        elif residual:
            condtion_flags.update(severity="WARNING",action="INVESTIGATE BIAS(residual)")
        
        

        self.log_state(condtion_flags,PERFORMANCELOG)
        return condtion_flags

    
    def trigger_action(self, condition_flags, model_registry, pipeline):
        """ trigger the next part of the mlops pipeline according to the model condition and call model_condition to log what action was done"""
        
        severity = condition_flags["severity"]
        action   = condition_flags["action"]

        if severity == "CRITICAL":
            pipeline.halt_serving()          # stop live predictions
            pipeline.trigger_retrain()       # kick off retraining job
            model_registry.rollback()        # revert to last stable version
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "ROLLED_BACK"},
                METRICLOG
            )

        elif severity == "RETRAIN":
            pipeline.trigger_retrain()
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "RETRAINING"},
                METRICLOG
            )

        elif severity == "WARNING":
            pipeline.increase_monitoring_frequency()
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "DEGRADED"},
                METRICLOG
            )

        elif severity == "WATCH":
            pipeline.flag_for_review()
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "WATCH"},
                METRICLOG
            )

        elif severity == "OK":
            self.model_condition_log(
                {"severity": severity, "action": "NONE", "model_state": "HEALTHY"},
                METRICLOG
            )

