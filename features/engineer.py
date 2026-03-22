import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    PUBLIC_HOLIDAYS_2022,
    TRAIN_END_DATE, VAL_END_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS
)






class FeatureEngineer:
    def __init__(self):
        self.zone_stats=None
        self.borough_encoder=None




    def load(self,path:str)->pd.DataFrame:
        print(f"\n[1/6] Loading processed data from: {path}")
        df=pd.read_parquet(path)

        df["hour_timestamp"]=pd.to_datetime(df["hour_timestamp"])
        df["zone_id"]=df["zone_id"].astype(int)
        df["demand"]=df["demand"].fillna(0).astype(float)

        df=df.sort_values(["zone_id","hour_timestamp"]).reset_index(drop=True)
        print(f"  → Loaded {len(df):,} rows | {df['zone_id'].nunique()} zones")
        print(f"  → Date range: {df['hour_timestamp'].min()} to {df['hour_timestamp'].max()}")
        return df




    def add_temporal_features(self,df:pd.DataFrame)->pd.DataFrame:
        print(f"\n[2/6] Adding temporal features...")
        ts=df["hour_timestamp"]

        df["hour_of_day"]=ts.dt.hour
        df["day_of_week"]=ts.dt.dayofweek
        df["month"]=ts.dt.month

        df["is_weekend"]=(df["day_of_week"]>=5).astype(int)

        df["is_rush_am"]=(df["hour_of_day"].between(7,9)).astype(int)
        df["is_rush_pm"]=(df["hour_of_day"].between(17,19)).astype(int)
        df["is_night"]=( (df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)
        ).astype(int)

        holidays=pd.to_datetime(PUBLIC_HOLIDAYS_2022)
        df["is_holiday"]=ts.dt.date.astype(["datetime64[ns]"]).isin(
            holidays
        ).astype(int)
        print(f"  → Added temporal features: hour_of_day, day_of_week, "
              f"month, is_weekend, is_rush_am, is_rush_pm, is_night, is_holiday")
        return df