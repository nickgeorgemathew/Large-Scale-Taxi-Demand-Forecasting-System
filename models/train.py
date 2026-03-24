import lightgbm
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import compute_metrics
import time
import  joblib



class ModelTrainer:
    def __init__(self):
        self.best_model={}
    


    def split_data(self,df:pd.DataFrame,val_start,test_start)->pd.DataFrame:
        self.train=df[df.hour_timestamp<val_start]

        self.val=df[(df.hour_timestamp>=val_start)&df.hour_timestamp < test_start]
        self.test=df[df.hour_timestamp >= test_start]
        assert train.hour_timestamp.max() < val.hour_timestamp.min()
        assert val.hour_timestamp.max() < test.hour_timestamp.min()
        
        print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        
        return self.train, self.val, self.test


    def baseline_naive_seasonal(df):
        preds = df['lag_168h']
        return compute_metrics(df['demand'], preds, label='Naive Seasonal Baseline')
    


    def train_model(self,train_df:pd.DataFrame,val_df:pd.DataFrame):
        start=time.perf_counter()
        feature_cols = [
            'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_holiday',
            'is_rush_am', 'is_rush_pm', 'zone_id', 'borough_encoded',
            'lag_1h', 'lag_2h', 'lag_3h', 'lag_6h', 'lag_24h', 'lag_48h', 'lag_168h',
            'roll_mean_3h', 'roll_mean_6h', 'roll_mean_24h',
            'roll_std_6h', 'mean', 'std'   
        ]

        X_train=train_df[feature_cols]
        y_train=train_df['demand']
        X_val=val_df[feature_cols]
        y_val=val_df['demand']


        self.model=LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_leaves=20,
            subsample=0.8,
            colsample_bytree=0.8
        )
        print("="*60)
        print("fitting the model")
        print("="*60)

        self.model.fit(X_train,y_train,eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(100), log_evaluation(100)])
        
        end=(start-time.perf_counter())*1000
        print(f"model fit in time :{end}")
        
        joblib.dump(self.model,"C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/lgbm_demand_v1.pkl")
        
        
        joblib.dump(feature_cols,"C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/feature_cols.json")

        return self.model


    def plot_feature(self):
        lgb.plot(self.model,max_num_features=20)


    def run(self):
        self.train_model()
        self.plot_feature()

    
