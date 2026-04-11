from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import Functions as f
from pyspark.sql.types import(
    StructType,StructField,IntegerType,FloatType,TimestampType,stringType
)
from monitoring.prediction_logger import prediction_logger
from config.settings import LOG
from pathlib import Path
from datetime import datetime,timedelta
from models.evaluate import Evaluate
import json
from config.settings import METRICLOG,LOG
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS
)


class DriftDetector:


    def __init__(self):
        pass
        

    def load_training_distribution(self):
        df_training=pd.read_csv(PROCESSED_PATH)

        pass
    
    
    def compute_feature_drift(self):
        pass
    
    
    def compute_residual_drift(self):
        pass
    
    def detect_drift(self):
        pass