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



    def add_lag_features(self,df:pd.DataFrame)->pd.DataFrame:
        print(f"\n[3/6] Adding lag features for lags: {LAG_HOURS} hours...")
        grouped=df.groupby("zone_id")["demand"]

        for lag in LAG_HOURS:
            col_name=f"lag_{lag}h"
            df[col_name]=grouped.shift(lag)
        
        nan_count=df[f"lag_{max(LAG_HOURS)}h"].isna().sum()
        print(f"  → Created {len(LAG_HOURS)} lag features")
        print(f"  → {nan_count:,} NaN rows from lag warmup (will be dropped at finalize)")
        return df





    def add_rolling_features(self,df:pd.DataFrame)->pd.DataFrame:
        print(f"\n[4/6] Adding rolling window features for windows: {ROLLING_WINDOWS} hours...")

        grouped=df.groupby("zone_id")["demand"]

        for window in ROLLING_WINDOWS:
            shifted=grouped.shift(1)
            rolled=shifted.groupby(df["zone_id"]).transform(
                lambda x:x.rolling(window,min_periods=1)
            ).mean()
            df[f"roll_mean_{window}h"]=rolled

        for window in [6,24]:
            shifted=grouped.shift(1)
            rolled_std=shifted.groupby(df["zone_id"]).transform(
                lambda x:x.rolling(window,min_periods=2).std()
            )
        df[f"roll_std_{window}h"] = rolled_std.fillna(0)

        print(f"  → Created {len(ROLLING_WINDOWS)} rolling mean + 2 rolling std features")
        return df
    



    def add_zone_features(self,df:pd.DataFrame,train_mask:pd.Series)->pd.DataFrame:
        print(f"\n[5/6] Adding zone/spatial features...")
        train_df=df[train_mask]
        self.zone_stats-(
            train_df.groupby("zone_id")["demand"]
            .agg(
                zone_mean_demand="mean",
                zone_std_demands="std",
                zone_max_demands="max"
            )
            .reset_index()
        )
        self.zone_stats["zone_std_demand"]=self.zone_stats["zone_std_demand"].fillna(0)
        df=df.merge(self.zone_stats,on="zone_id",how="left")


        try:
            zone_lookup=pd.read_csv() 
            zone_lookup=zone_lookup.rename(columns={
                "LocationID": "zone_id",
                "Borough":    "borough"
            })[["zone_id", "borough"]]
            df=df.merge(zone_lookup,on="zone_id",how="left")
            df["borough"]=df["borough"].fillna("Unknown")
        except FileNotFoundError:
            # Fallback: simplified borough assignment
            # Based on TLC official zone ranges (approximate)
            print("  ⚠ taxi_zone_lookup.csv not found — using simplified borough mapping")
            conditions = [
                df["zone_id"].between(1, 69),     # Bronx
                df["zone_id"].between(70, 89),    # Brooklyn
                df["zone_id"].between(90, 168),   # Manhattan
                df["zone_id"].between(169, 220),  # Queens
                df["zone_id"].between(221, 263),  # Staten Island
            ]
            boroughs = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
            df["borough"] = np.select(conditions, boroughs, default="Unknown")

        # Encode borough as integer for the model
        self.borough_encoder = LabelEncoder()
        df["borough_encoded"] = self.borough_encoder.fit_transform(df["borough"])

        print(f"  → Added zone_mean_demand, zone_std_demand, borough_encoded")
        print(f"  → Borough distribution:\n{df['borough'].value_counts()}")
        return df
    def finalise(self,df:pd.DataFrame)->pd.DataFrame:
        print(f"\n[6/6] Finalizing feature matrix...")
        rows_before=len(df)
        
        df=df.dropna(subset=["lag_168h"])

        df["demand"]=df["demand"].clip(lower=0)
        rows_after=len(df)
        print(f"  → Dropped {rows_before - rows_after:,} warmup rows (lag NaNs)")
        missing_features=[c for c in FEATURE_COLUMNS
                        if c not in df.columns]
        if missing_features:
            print(f"  ⚠ Warning: these features are missing: {missing_features}")
            print(f"    Check add_zone_features() — zone lookup file may be missing")


        df.to_parquet(FEATURES_PATH, index=False)
        print(f"  → Saved {len(df):,} rows × {len(df.columns)} columns to {FEATURES_PATH}")
        print(f"  → Feature columns: {list(df.columns)}")
        return df
        



    def run(self)->pd.DataFrame:
        print("=" * 60)
        print("  FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

        # Load
        df = self.load(PROCESSED_PATH)

        # Compute train mask early — needed for zone stats
        train_mask = df["hour_timestamp"] <= pd.Timestamp(TRAIN_END_DATE)
        print(f"\n  Training rows: {train_mask.sum():,} | "
              f"Val+Test rows: {(~train_mask).sum():,}")

        # Build features
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df)
        df = self.add_rolling_features(df)
        df = self.add_zone_features(df, train_mask)

        # Save and return
        df = self.finalize(df)

        print("\n" + "=" * 60)
        print("  FEATURE ENGINEERING COMPLETE")
        print("=" * 60)
        return df

             

if __name__ == "__main__":
    engineer = FeatureEngineer()
    df = engineer.run()

    # Quick sanity check — spot-check a few lag values
    print("\n--- SANITY CHECK: Verify lag_1h for zone 132 ---")
    zone_132 = df[df["zone_id"] == 132][
        ["hour_timestamp", "demand", "lag_1h", "lag_24h", "lag_168h"]
    ].head(10)
    print(zone_132.to_string())
    print("\nFor each row: lag_1h should equal the demand value in the previous row.")
    print("If it doesn't, the groupby-shift logic has a bug.")
