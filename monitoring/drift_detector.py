import numpy as np
import pandas as pd
from pyspark.sql import functions as f
from config.settings import SPARK_APP_NAME,SPARK_DRIVER_MEMORY,SPARK_SHUFFLE_PARTITIONS
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from config.settings import MONITORED_FEATURES,TEST_START_DATE,DATA_START_DATE,DATA_END_DATE,VAL_END_DATE,TRAIN_END_DATE,FEATURES_PATH,RAW_DATA_FORMAT,PATH_PREV_TRAIN_DATA,PROCESSED_PATH

def calculate_psi(expected, actual, n_bins=10):
    """Population Stability Index."""
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    expected_percents[expected_percents == 0] = 1e-6
    actual_percents[actual_percents == 0] = 1e-6
    psi_val = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_val)




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

class DriftDetector:

    def __init__(self, spark=None):
        self.spark = spark  # optional, not used for pandas

    def load_data(self,file_path):
        if RAW_DATA_FORMAT =="csv":
            df=pd.read_csv(file_path)
        elif RAW_DATA_FORMAT =="parquet":
            df=spark.read.parquet(file_path)
        return df
    def split_data(self,df):
            
            self.train=df.filter(f.col("hour_timestamp") < TRAIN_END_DATE)
            self.val=df.filter((f.col("hour_timestamp")>=TRAIN_END_DATE)&(f.col("hour_timestamp") < VAL_END_DATE))
            self.test=df.filter(f.col("hour_timestamp") >= TEST_START_DATE)
    
            train_max=self.train.select(f.max("hour_timestamp")).collect()[0][0]
            val_min=self.val.select(f.min("hour_timestamp")).collect()[0][0]
            val_max=self.val.select(f.max("hour_timestamp")).collect()[0][0]
            test_min=self.test.select(f.min("hour_timestamp")).collect()[0][0]
    
            assert train_max < val_min
            assert val_max < test_min
    
            # Convert to Pandas for LightGBM compatibility
            print("Converting Spark DataFrames to Pandas for LightGBM...")
            self.train_pd = self.train.toPandas()
            self.val_pd = self.val.toPandas()
            self.test_pd = self.test.toPandas()
    
            print(f"Train: {self.train_pd.shape}, Val: {self.val_pd.shape}, Test: {self.test_pd.shape}")
            
            return self.train_pd, self.val_pd, self.test_pd
    
    def compute_feature_drift(self, training_df, current_df, threshold=0.1):
        feature_flag = {}
        for feature in MONITORED_FEATURES:
            if feature not in training_df.columns or feature not in current_df.columns:
                continue
            train_vals = training_df[feature].dropna().values
            curr_vals = current_df[feature].dropna().values
            if len(train_vals) == 0 or len(curr_vals) == 0:
                psi = 0.0
            else:
                psi = calculate_psi(train_vals, curr_vals)
            feature_flag[f"{feature}_drift"] = bool(psi > threshold)
        return feature_flag

    def compute_residual_drift(self, df, threshold=0.5):
        if 'actual' not in df.columns or 'prediction' not in df.columns:
            return {"residual_drift": False}
        residual = df['actual'] - df['prediction']
        mean_residual = np.mean(residual)
        return {"residual_drift": abs(mean_residual) > threshold}

    def detect_drift(self, feature_drift_flags, residual_drift_flag):
        feature_detected = any(v for k, v in feature_drift_flags.items() if k != "residual_drift")
        residual_detected = residual_drift_flag.get("residual_drift", False)
        drift_flag = {"feature_drift": feature_detected, "residual_drift": residual_detected}
        if feature_detected:
            drift_flag["features"] = [k for k, v in feature_drift_flags.items() if v and k != "residual_drift"]
        return drift_flag



if __name__=="__main__":
    spark=create_spark_session()
    detector=DriftDetector(spark=spark)
    prev_df=detector.load_data(PATH_PREV_TRAIN_DATA)
    current_df=detector.load_data(PROCESSED_PATH)
    train_prev_df,val_prev_df,test_prev_df=detector.split_data(prev_df)
    train_current_df,val_current_df,test_current_df=detector.split_data(current_df)

    feature_drift_flag=detector.compute_feature_drift(train_current_df,train_prev_df)
    print("===="*60)
    print("feature drift flag")
    print("===="*60)
    print(feature_drift_flag)

    residual_drift_flag=detector.compute_residual_drift(current_df)
    print("===="*60)
    print("residual_drift_flag")
    print("===="*60)
    print(residual_drift_flag)
    
    drift_flag=detector.detect_drift(feature_drift_flag,residual_drift_flag)
    print("===="*60)
    print("drift_flag")
    print("===="*60)
    print(drift_flag)
