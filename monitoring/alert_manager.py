import pandas as pd
from monitoring.prediction_logger import prediction_logger
from pathlib import Path
from datetime import datetime,timedelta
from metrics_monitor import MetricsMonitor
from models.evaluate import Evaluate
import json
import numpy as np
from config.settings import METRICLOG,LOG,PERFORMANCELOG
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,MONITORED_FEATURES,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS,
SPARK_APP_NAME, SPARK_SHUFFLE_PARTITIONS, SPARK_DRIVER_MEMORY)


def save_log_parquet(log,file_path):
        df_log=pd.DataFrame(log)        
        # Load, Combine, and Overwrite
        df_existing = pd.read_parquet(Path(file_path))
        if df_existing:
            df_combined = pd.concat([df_existing, df_log], ignore_index=True)
            df_combined.to_parquet(Path(file_path), engine='pyarrow')
        # create new file if it doesnot exist
        else:
            df_log.to_parquet(Path(file_path),engine="pyarrow") 

        return df_log




class AlertManager:

    def __init__(self):
        pass

    def log_state(self,flags,severity,action):
        """ log the issue and what action was taken with timestamp"""
        current_time=datetime.now()
        flag=[k for k,val in flags.items() if val]
        log={"timestamp":current_time,"issue flags":flag,"severity":severity,"action":action}
        save_log_parquet(log,PERFORMANCELOG)
     
    
    
    def model_condition(self,flags):


        pass
    def assess_perfomance(self):
        pass

    def asses_drift(self):
        pass
    def assess_condition(self,performance_flag,drift_flag):


        pass
    
    def trigger_action(self):
        pass

