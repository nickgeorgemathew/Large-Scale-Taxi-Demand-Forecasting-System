import os
from pathlib import Path




PROCESSED_PATH="C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/data/processed/processed.parquet"#where the processed nyc data  by etl/spark pipeline is stored
FEATURES_PATH="C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/features_store/selected_features.parquet"#path to file that contains the data that has been feature engineered
ZONE_PATH=""#file path where zone lookup file is present
PATH_CURRENT_TRAIN_DATA="C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/yellow_tripdata_2022-03.parquet"
BASELINE_METRICS_PATH=Path("models/artifacts/model_metrics_test.json")#file path where baseline metrics of the trained model is stored
LAG_HOURS=[1, 2, 3, 6, 24, 48, 168]#the list of hours when lag is choosen
ROLLING_WINDOWS=[1,2,3,6,24,48,168]#the hours/time gap of the rolling windows
TRAIN_END_DATE="2022-03-22"#date in the dataset which splits the data into training(splitting dataset)
VAL_END_DATE="2022-03-26"#date in the dataset which splits the data into validation(splitting dataset)
TEST_START_DATE="2022-03-26"#date in the dataset which splits the data into testing(splitting dataset)
TARGET_COLUMN="demand"#which column is the target/one which should be predicted
FEATURE_COLUMNS=['hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday','is_rush_am', 'is_rush_pm', 'zone_id', 'borough_encoded','lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_24h', 'lag_48h', 'lag_168h','roll_mean_3h', 'roll_mean_6h', 'roll_mean_24h','roll_std_6h', 'roll_std_24h',          # <-- added 24h std
    'zone_mean_demand', 'zone_std_demand']#list of column names which are used to train
MONITORED_FEATURES=['hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday','is_rush_am', 'is_rush_pm', 'zone_id', 'borough_encoded','lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_24h', 'lag_48h', 'lag_168h','roll_mean_3h', 'roll_mean_6h', 'roll_mean_24h','roll_std_6h', 'mean', 'std']#features to be monitored to check if any drift or quality drop is happening
PUBLIC_HOLIDAYS_2022=[]#rewrite as neccessary to the dataset ,to determine rush hours etc,must be a list of dates
RAW_DATA_PATH="C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/yellow_tripdata_2022-03.parquet"#where the raw nyc taxi data is stored(path in str)
RAW_DATA_FORMAT="parquet"#which format the nyc taxi data raw is stored in "csv" or "parquet"
COLUMN_MAP = {
    "tpep_pickup_datetime": "pickup_datetime",
    "tpep_dropoff_datetime": "dropoff_datetime",
    "PULocationID": "zone_id",       # only if using pre-zoned data
}#standardise column names from raw names:standard names
REQUIRED_COLUMNS=[ "pickup_datetime", "dropoff_datetime",
       'passenger_count', 'trip_distance',"zone_id", 'fare_amount', 'extra',
       'total_amount', 'congestion_surcharge']#columns that should be checked to be present to ensure schema validation,it is  a list
VALID_ZONE_MIN=1#the min valid zone id
VALID_ZONE_MAX=265#the maximum valid zone id
MIN_FARE=0.01 #min fare to prevent outliers
MAX_FARE=1777#max fare to prevent outliers
MIN_DISTANCE=0.7# to prevent outliers
MAX_DISTANCE=286259.84# to prevent outliers
MIN_PASSENGER=1.0 # to prevent outliers
MAX_PASSENGER=8.0# to prevent outliers
DATA_START_DATE="2022-03-01" #find from dataset
DATA_END_DATE="2022-03-31"#find from dataset
TIME_GRANULARITY="hour"#the string you pass to date_trunc that tells it what unit to round down to.
FILL_MISSING_ZEROS=False#true or false flag that tells the spark_pipeine to fill the missing zeros if true and not if false
SPARK_APP_NAME="Taxi-Demand-Forecasting"#name of the spark app when it is initialised
SPARK_SHUFFLE_PARTITIONS="200"#used here.config("spark.sql.shuffle.partitions",SPARK_SHUFFLE_PARTITIONS),find why
SPARK_DRIVER_MEMORY="4g"# used here spark=(SparkSession.builder.appName(SPARK_APP_NAME).master("local[*]").config("spark.driver.memory",SPARK_DRIVER_MEMORY) find why
LOG=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/log/log.parquet")#file location for storing parquet logs
METRICLOG=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/metric_log/metric_log.parquet") #'model/artifacts/metric_history_24h.json','model/artifacts/metric_history_week.json'
MODELCONDITIONLOG=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/performance_log/model_performance_log.parquet")
PERFORMANCELOG=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/performance_log/performance_log.parquet")# LOG THE PERFORMANCE ISSUES LIKE FEATURE DRIFT,METRIC DEGRADATION ETC WITH TIMESTAMP IN THIS LOCATION 
MODEL_PATH=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/model")#path of where the model is stored
QUANTILE_LOW_MODEL_PATH=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/quantile_low_model")#path of where the model is stored
QUANTILE_HIGH_MODEL_PATH=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/quantile_high_model")#path of where the model is stored
HOTSPOTS=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/log/hotspots/hotspots.parquet")#file path where to store the hotspots
RECENT_HISTORY=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/log/RECENT_HISTORY.json")#file location for storing parquet logs
SERVING_HALTED=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/serving_halted/serving_halted.json")#location of json file containing flag of whether model is able to predict or not
MONITORING_CONFIG=Path("C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/monitoring_config/monitoring_config.json")#file that serves as flag for """Switch monitoring to high‑frequency mode via config file."""
#        config_path = Path(MONITORING_CONFIG)
#        if config_path.exists():
#           config = json.loads(config_path.read_text())
#        else:
#            config = {"interval_minutes": 30, "high_alert_mode": False}
#        config["interval_minutes"] = 5
#        config["high_alert_mode"] = True
#        config_path.write_text(json.dumps(config, indent=2))
#        logging.info("Monitoring frequency increased to 5 minutes")

# feature_cols = [
        #     'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
        #     'is_rush_am', 'is_rush_pm', 'zone_id', 'borough_encoded',
        #     'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_24h', 'lag_48h', 'lag_168h',
        #     'roll_mean_3h', 'roll_mean_6h', 'roll_mean_24h',
        #     'roll_std_6h', 'mean', 'std'   
        # ]