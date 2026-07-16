import lightgbm
from collections import Counter

import os
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

from datetime import datetime
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
import json
import matplotlib.pyplot as plt
import optuna
from pathlib import Path
import time
import  joblib
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS,SPARK_APP_NAME, SPARK_SHUFFLE_PARTITIONS, SPARK_DRIVER_MEMORY
)
from models.evaluate import Evaluate






PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


class ModelTrainer:
    def __init__(self,spark,version):
        self.spark=spark
        self.best_model={}
        self.baseline={}
        self.version=version
    


    def load_data(self,file_path):
        df=self.spark.read.parquet(file_path)
        return df


    # def split_data_pandas(self,df:ps.DataFrame)->ps.DataFrame:
        
    #     df=df.pandas_api()
        

    #     self.train=df[df.hour_timestamp<TRAIN_END_DATE]
    #     self.val=df[(df.hour_timestamp>=TRAIN_END_DATE)&(df.hour_timestamp < VAL_END_DATE)]
    #     self.test=df[df.hour_timestamp >= TEST_START_DATE]
        
    #     assert self.train.hour_timestamp.max() < self.val.hour_timestamp.min()
    #     assert self.val.hour_timestamp.max() < self.test.hour_timestamp.min()
        
    #     print(f"Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")
        
    #     return self.train, self.val, self.test 
    
    
    #alternative to split_data_pandas if there occurs an error because of the data being over 10gb
    
        # def split_data(self, df):
        # """Convert to Pandas once, split, and store as Pandas DataFrames."""
        # pdf = df.toPandas()
        # pdf['hour_timestamp'] = pd.to_datetime(pdf['hour_timestamp'])
        
        # self.train_pd = pdf[pdf['hour_timestamp'] < TRAIN_END_DATE]
        # self.val_pd = pdf[(pdf['hour_timestamp'] >= TRAIN_END_DATE) & 
        #                 (pdf['hour_timestamp'] < VAL_END_DATE)]
        # self.test_pd = pdf[pdf['hour_timestamp'] >= TEST_START_DATE]
        
        # # Store Spark versions too (if needed for other methods)
        # self.train = self.spark.createDataFrame(self.train_pd)
        # self.val = self.spark.createDataFrame(self.val_pd)
        # self.test = self.spark.createDataFrame(self.test_pd)
        
        # print(f"Train: {self.train_pd.shape}, Val: {self.val_pd.shape}, Test: {self.test_pd.shape}")
        # return self.train, self.val, self.test

    

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

    
   
   
   
   
    def tune_hyperparameters(self, n_trials=50):
        # Extract features and targets using the Pandas DataFrames
        X_train = self.train_pd[FEATURE_COLUMNS]
        y_train = self.train_pd[TARGET_COLUMN]
        X_val = self.val_pd[FEATURE_COLUMNS]
        y_val = self.val_pd[TARGET_COLUMN]
        def objective(trial):
            params = {
                'n_estimators': 2000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            }
            
            model = LGBMRegressor(**params, random_state=42, verbose=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(100, verbose=False)]
            )
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"\nBest hyperparameters: {study.best_params}")
        print(f"Best RMSE: {study.best_value:.4f}")
        self.best_model.update(study.best_params)
        
        return self.best_model


    def train_model(self):
        start=time.perf_counter()
        

        X_train = self.train_pd[FEATURE_COLUMNS]
        y_train = self.train_pd[TARGET_COLUMN]
        X_val = self.val_pd[FEATURE_COLUMNS]
        y_val = self.val_pd[TARGET_COLUMN]


        self.model=LGBMRegressor(
            **self.best_model
        )
        print("="*60)
        print("fitting the model")
        print("="*60)

        self.model.fit(X_train,y_train,
                       eval_set=[(X_val, y_val)],
        
        callbacks=[early_stopping(100), log_evaluation(100)]
        )
        
        end=(time.perf_counter()-start)*1000
        print(f"model fit in time :{end}")
        model_path = MODEL_DIR / f"model_v{self.version}.pkl"
        joblib.dump(self.model, model_path)

        
        return self.model
    


    def train_quantile_low_model(self):
        start=time.perf_counter()
        

        X_train = self.train_pd[FEATURE_COLUMNS]
        y_train = self.train_pd[TARGET_COLUMN]
        X_val = self.val_pd[FEATURE_COLUMNS]
        y_val = self.val_pd[TARGET_COLUMN]


        self.quantile_low_model=LGBMRegressor(
            **self.best_model,objective ="quantile",alpha = 0.1
        )
        print("="*60)
        print("fitting the model")
        print("="*60)

        self.quantile_low_model.fit(X_train,y_train,
                       eval_set=[(X_val, y_val)],
        
        callbacks=[early_stopping(100), log_evaluation(100)]
        )
        
        end=(time.perf_counter()-start)*1000
        print(f"model fit in time :{end}")

        model_path = MODEL_DIR / f"quantile_low_model_v{self.version}.pkl"
        joblib.dump(self.quantile_low_model, model_path)
        
        return self.quantile_low_model




    def train_quantile_high_model(self):
        start=time.perf_counter()
        

        X_train = self.train_pd[FEATURE_COLUMNS]
        y_train = self.train_pd[TARGET_COLUMN]
        X_val = self.val_pd[FEATURE_COLUMNS]
        y_val = self.val_pd[TARGET_COLUMN]


        self.quantile_high_model=LGBMRegressor(
            **self.best_model,objective ="quantile",alpha = 0.9
        )
        print("="*60)
        print("fitting the model")
        print("="*60)

        self.quantile_high_model.fit(X_train,y_train,
                       eval_set=[(X_val, y_val)],
        
        callbacks=[early_stopping(100), log_evaluation(100)]
        )
        
        end=(time.perf_counter()-start)*1000
        print(f"model fit in time :{end}")

        model_path = MODEL_DIR / f"quantile_high_model_v{self.version}.pkl"
        joblib.dump(self.quantile_high_model, model_path)

        
        return self.quantile_high_model


    
    def best_model_selection(self,models:dict):
        #compare all the metrics across the models and choose the best one
        rmse={}
        mae={}
        r_square={}
        smape={}

        for k,model in models:
            y_true = self.test_pd['demand']
            y_pred = model.predict(self.test_pd[FEATURE_COLUMNS])
            y_pred = np.clip(y_pred,0,None)  
    
            # Global metrics
            MAE  = mean_absolute_error(y_true, y_pred)
            RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            SMAPE=self.smape(y_true,y_pred)
            rmse[k]=RMSE
            mae[k]=MAE
            r_square[k]=r2
            smape[k]=SMAPE
        rmse_min=min(rmse.items())
        mae_min=min(mae.items())
        r_square_min=min(r_square.items())
        smape_max=max(smape.items())
        count=[rmse_min[0],mae_min[0],r_square_min[0],smape_max[0]]
        count=Counter(count)
        count=count.most_common(n=1)
        return count[0][0]
       

    
    
    
    def save_best_model(self,models:list):

        #call  evaluation metrics
        #compare all models then the best model gets saved
        best_model=self.best_model_selection(models)
        print(f"best model is {best_model}")
        
        model_path = MODEL_DIR / f"lgbm_demand_{best_model}_v{self.version}.pkl"
        joblib.dump(models[best_model], model_path)
            
        with open(MODEL_DIR/"feature_cols.json",'w') as f:
            json.dump(FEATURE_COLUMNS,f)
        return models[best_model]
        
        





    def analyze_feature_importance(self,model,top_n=20):
        """Show which features matter most."""
        importance_df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print('='*60)
        print(f"  Top {top_n} Most Important Features")
        print("="*60)
        print(importance_df.head(top_n).to_string(index=False))
        
        # Plot
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'].head(top_n), 
                importance_df['importance'].head(top_n))
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / f'feature_importance_{self.version}.png', dpi=150)
        plt.close()
        
        return importance_df




   


    

    #run the entire pipeline
    def run_complete_pipeline(self):
        """Full training and evaluation pipeline."""
        evaluation=Evaluate()
    
        # 1. Split data
        print("\n" + "="*60)
        print("  STEP 1: SPLITTING DATA")
        print("="*60)
        df=self.load_data(FEATURES_PATH)
        train,val,test=self.split_data(df)
        
        # 2. Baseline
        print("\n" + "="*60)
        print("  STEP 2: BASELINE MODEL")
        print("="*60)
        baseline_metrics_train = evaluation.baseline_naive_seasonal(train,split="train")
        baseline_metrics_val = evaluation.baseline_naive_seasonal(val,split="val")
        baseline_metrics_test = evaluation.baseline_naive_seasonal(test,split="test")
        print("\n" + "="*60)
        print("   BASELINE MODEL METRICS")
        print("\n TRAIN METRICS")
        print(baseline_metrics_train)
        print("\n Test METRICS")
        print(baseline_metrics_test)
        print("\n VALIDATION METRICS")
        print(baseline_metrics_val)
        print("="*60)
        
        # 3. Fine tune hyper parameter of model
        print("\n" + "="*60)
        print("  STEP 3: Fine tuning hyperparameters")
        print("="*60)
        self.tune_hyperparameters()
        


        # 4. Train model
        print("\n" + "="*60)
        print("  STEP 4: TRAINING LIGHTGBM")
        print("="*60)
        model=self.train_model()
        quantile_low_model=self.train_quantile_high_model()
        quantile_high_model=self.train_quantile_low_model()

        # 5. Evaluate
        print("\n" + "="*60)
        print("  STEP 5: EVALUATION")
        print("="*60)
        train_metrics=evaluation.evaluate_model(train, model, 'train')
        val_metrics=evaluation.evaluate_model(val, model, 'val')
        test_metrics = evaluation.evaluate_model(test, model, 'test')
        print("="*60)
        print("EVALUTING QUANTILE LOW MODEL")
        train_metrics_quantile_low=evaluation.evaluate_model(train, quantile_low_model, 'train')
        val_metrics_quantile_low=evaluation.evaluate_model(val, quantile_low_model, 'val')
        test_metrics_quantile_low = evaluation.evaluate_model(test, quantile_low_model, 'test')
        print("="*60)
        print("EVALUTING QUANTILE HIGH MODEL")
        train_metrics_quantile_high=evaluation.evaluate_model(train, quantile_high_model, 'train')
        val_metrics_quantile_high=evaluation.evaluate_model(val, quantile_high_model, 'val')
        test_metrics_quantile_high = evaluation.evaluate_model(test, quantile_high_model, 'test')


        #save best model and features
        print("\n" + "="*60)
        print("  STEP 6:save model and features")
        print("="*60)
        best_model=self.save_best_model(models={"base model":model,"quantile_high_model":quantile_high_model,"quantile_low_model":quantile_low_model})
        
        # 5. Feature importance
        print("\n" + "="*60)
        print("  STEP 7: FEATURE ANALYSIS")
        print("="*60)
        importance_df = self.analyze_feature_importance(best_model)
        
        # 6. Residual analysis
        print("\n" + "="*60)
        print("  STEP 8: ERROR ANALYSIS")
        print("="*60)
        evaluation.analyze_residuals(test,best_model, 'Test')
   
        
        
        print("\n" + "="*60)
        print("  PIPELINE COMPLETE")
        print("="*60)
        
        return {
            'baseline_metrics_train': baseline_metrics_train,
            'baseline_metrics_val':baseline_metrics_val,
            'baseline_metrics_test':baseline_metrics_test,
            'train metrics':train_metrics,
            'validation_metrics':val_metrics,
            'test metrics': test_metrics,
            'train metrics quantile low model':train_metrics_quantile_low,
            'validation_metrics quantile low model':val_metrics_quantile_low,
            'test metrics quanitle low model': test_metrics_quantile_low,
            'train metrics quantile high model':train_metrics_quantile_high,
            'validation_metrics quantile high model':val_metrics_quantile_high,
            'test metrics quanitle high model': test_metrics_quantile_high,
            'feature importance': importance_df
        }
    
if __name__=="__main__":
    print("\n" + "="*60)
    print("  creating new Spark session")
    print("="*60)
    spark=create_spark_session()
    try:
        train=ModelTrainer(spark,version=datetime.today().strftime("%Y%m%d_%H%M%S"))
        train.run_complete_pipeline()
    finally:
        spark.stop()
        print("\n SparkSesssion stopped")
