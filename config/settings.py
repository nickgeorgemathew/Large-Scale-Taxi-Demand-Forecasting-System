import os
from pathlib import Path




PROCESSED_PATH#where the processed nyc data  by etl/spark pipeline is stored
FEATURES_PATH#path to file that contains the selected features of the dataset used for prediciton and learning
LAG_HOURS#the list of hours when lag is choosen
ROLLING_WINDOWS#the hours/time gap of the rolling windows
TRAIN_END_DATE#date in the dataset which splits the data into training(splitting dataset)
VAL_END_DATE#date in the dataset which splits the data into validation(splitting dataset)
TEST_START_DATE#date in the dataset which splits the data into testing(splitting dataset)
TARGET_COLUMN#which column is the target/one which should be predicted
FEATURE_COLUMNS#list of column names which are used to train
MONITORED_FEATURES#features to be monitored to check if any drift or quality drop is happening
PUBLIC_HOLIDAYS_2022#rewrite as neccessary to the dataset ,to determine rush hours etc,a list of dates
RAW_DATA_PATH#where the raw nyc taxi data is stored(path)
RAW_DATA_FORMAT#which format the nyc taxi data raw is stored in
COLUMN_MAP#standardise column names from raw names:standard names
REQUIRED_COLUMNS#columns that should be checked to be present to ensure schema validation,it is  a list
VALID_ZONE_MIN#the min valid zone id
VALID_ZONE_MAX#the maximum valid zone id
MIN_FARE, #min fare to prevent outliers
MAX_FARE,#max fare to prevent outliers
MIN_DISTANCE,# to prevent outliers
MAX_DISTANCE,# to prevent outliers
MIN_PASSENGER, # to prevent outliers
MAX_PASSENGER,# to prevent outliers
DATA_START_DATE, #find from dataset
DATA_END_DATE,#find from dataset
TIME_GRANULARITY#used in df=df.withColumn("hour_timestamp",f.date_trunc(TIME_GRANULARITY,F.col("pickup_datetime"))),probably the min timestamp??
FILL_MISSING_ZEROS#true or false flag that tells the spark_pipeine to fill the missing zeros if true and not if false
SPARK_APP_NAME#name of the spark app when it is initialised
SPARK_SHUFFLE_PARTITIONS#used here.config("spark.sql.shuffle.partitions",SPARK_SHUFFLE_PARTITIONS),find why
SPARK_DRIVER_MEMORY# used here spark=(SparkSession.builder.appName(SPARK_APP_NAME).master("local[*]").config("spark.driver.memory",SPARK_DRIVER_MEMORY) find why
LOG#file location for storing parquet logs
METRICLOG #'model/artifacts/metric_history_24h.json','model/artifacts/metric_history_week.json'
PERFORMANCELOG# LOG THE PERFORMANCE ISSUES LIKE FEATURE DRIFT,METRIC DEGRADATION ETC WITH TIMESTAMP IN THIS LOCATION 
MODEL_PATH#path of where the model is stored
QUANTILE_LOW_MODEL_PATH#path of where the model is stored
QUANTILE_HIGH_MODEL_PATH#path of where the model is stored
HOTSPOTS#file path where to store the hotspots
RECENT_HISTORY#file location for storing parquet logs
SERVING_HALTED#location of json file containing flag of whether model is able to predict or not

# feature_cols = [
        #     'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
        #     'is_rush_am', 'is_rush_pm', 'zone_id', 'borough_encoded',
        #     'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_24h', 'lag_48h', 'lag_168h',
        #     'roll_mean_3h', 'roll_mean_6h', 'roll_mean_24h',
        #     'roll_std_6h', 'mean', 'std'   
        # ]