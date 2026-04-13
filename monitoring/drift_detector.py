from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from monitoring.prediction_logger import prediction_logger
from config.settings import LOG
from pathlib import Path
from datetime import datetime,timedelta
from models.evaluate import Evaluate
import json
import numpy as np
from config.settings import METRICLOG,LOG
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,MONITORED_FEATURES,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS,
SPARK_APP_NAME, SPARK_SHUFFLE_PARTITIONS, SPARK_DRIVER_MEMORY)



def create_spark_session()->SparkSession:
    spark=(SparkSession.builder.appName(SPARK_APP_NAME).master("local[*]")
           .config("spark.driver.memory",SPARK_DRIVER_MEMORY)
           .config("spark.sql.shuffle.partitions",SPARK_SHUFFLE_PARTITIONS)
           .config("spark.sql.adaptive.enabled","true")
           .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=WARN")
        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    print(f"SparkSession created | version: {spark.version}")
    return spark



def calculate_psi(expected, actual, n_bins=10):
    """Calculates PSI between two populations."""
    # Define bins based on the expected population
    breakpoints = np.linspace(0, 100, n_bins + 1)
    
    # Calculate quantiles to get bin edges
    bins = np.percentile(expected, breakpoints)
    
    # Bin the data
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    
    # Handle zeros to avoid log(0)
    expected_percents[expected_percents == 0] = 1e-6
    actual_percents[actual_percents == 0] = 1e-6
    
    # Calculate PSI
    psi_val = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_val)


class DriftDetector:


    def __init__(self,spark):
        self.spark=spark
        

    def load_dataframe(self,file_path):
       
        df=self.spark.read.parquet(file_path)
        df=df.to_pandas()
        return df
    
    
    def compute_feature_drift(self,training,current,threshold):
        
        result={}
        for feature in MONITORED_FEATURES:
            
            psi=calculate_psi(training[feature],current[feature])
            
            if psi > threshold:
                result.update({f"{feature}_drift":True})
            else:
                result.update({f"{feature}_drift":False})
        
        return result


        
    
    
    def compute_residual_drift(self,df,threshold):
        
        residual=df["actual"]- df["prediction"]
        mean_residual=np.mean(residual)
        std_residual=np.std(residual)
        
        result={"residual_drift":bool(np.abs(mean_residual)>threshold)}

        return result


        
    
    def detect_drift(self,feature_drift,residual_drift):
        
        feature =[k if val is True else None for k,val in feature_drift ]
        residual=[k if val is True else None for k,val in feature_drift ]

        if feature or residual :
            print("drift detected !!")
            return feature_drift,residual_drift
        else:
            print("No drift detected")


        