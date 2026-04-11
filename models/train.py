import lightgbm
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pyspark.sql import SparkSession
from pyspark.sql import Functions as f
import pyspark.pandas as ps
import json
import optuna
from pathlib import Path
import numpy as np
import time
import  joblib
from config.settings import (
    PROCESSED_PATH, FEATURES_PATH,
    LAG_HOURS, ROLLING_WINDOWS,
    TRAIN_END_DATE, VAL_END_DATE,TEST_START_DATE,
    TARGET_COLUMN, FEATURE_COLUMNS,SPARK_APP_NAME, SPARK_SHUFFLE_PARTITIONS, SPARK_DRIVER_MEMORY
)
from evaluate import Evaluate





PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def create_spark_session()->SparkSession:
    spark=(SparkSession.builder.appName(SPARK_APP_NAME).master("local[*]")
           .config("spark.driver.memory",SPARK_DRIVER_MEMORY)
           .config("spark.sql.shuffle.partition",SPARK_SHUFFLE_PARTITIONS)
           .config("spark.sql.adaptive.enabled","true")
           .config("spark.driver.extraJavaOptions", "-Dlog4j.logLevel=WARN")
        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    print(f"SparkSession created | version: {spark.version}")
    return spark


class ModelTrainer:
    def __init__(self,spark):
        self.spark=spark
        self.best_model={}
        self.baseline={}
    
    


    def load_data(self,file_path):
        df=self.spark.read.parquet(file_path)
        return df


    def split_data_pandas(self,df:ps.DataFrame)->ps.DataFrame:
        
        df=df.pandas_api()
        

        self.train=df[df.hour_timestamp<TRAIN_END_DATE]
        self.val=df[(df.hour_timestamp>=TRAIN_END_DATE)&(df.hour_timestamp < VAL_END_DATE)]
        self.test=df[df.hour_timestamp >= TEST_START_DATE]
        
        assert self.train.hour_timestamp.max() < self.val.hour_timestamp.min()
        assert self.val.hour_timestamp.max() < self.test.hour_timestamp.min()
        
        print(f"Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")
        
        return self.train, self.val, self.test 

    

    def split_data(self,df):
        
        self.train=df.filter(f.col("hour_timestamp") < TRAIN_END_DATE)
        self.val=df.filter((f.col("hour_timestamp")>=TRAIN_END_DATE)&(f.col("hour_timestamp") < VAL_END_DATE))
        self.test=df.filter(f.col("hour_timestamp") >= TEST_START_DATE)

        train_max=self.train.select(f.max("hour_timestamp")).collect()[0][0]
        val_min=self.val.select(f.min("hour_timestamp")).collect()[0][0]
        val_max=self.val.select(f.max("hour_timestamp")).collect()[0][0]
        test_min=self.val.select(f.min("hour_timestamp")).collect()[0][0]

        assert train_max < val_min
        assert val_max < test_min

        print(f"Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")
        
        return self.train, self.val, self.test 

    
   
   
   
   
    def tune_hyperparameters(self, n_trials=50):
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
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                callbacks=[early_stopping(100, verbose=False)]
            )
            
            y_pred = model.predict(self.X_val)
            rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"\nBest hyperparameters: {study.best_params}")
        print(f"Best RMSE: {study.best_value:.4f}")
        self.best_model.update(study.best_params)
        
        return self.best_model


    def train_model(self):
        start=time.perf_counter()
        

        X_train=self.train[FEATURE_COLUMNS]
        y_train=self.train[TARGET_COLUMN]
        X_val=self.val[FEATURE_COLUMNS]
        y_val=self.val[TARGET_COLUMN]


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

        
        return self.model



    
        
       

    
    
    
    def save_best_model(self):
         
        model_path = MODEL_DIR / "lgbm_demand_v1.pkl"
        joblib.dump(self.model, model_path)
            
        with open(MODEL_DIR/"feature_cols.json",'w') as f:
            json.dump(FEATURE_COLUMNS,f)
        
        





    def analyze_feature_importance(self, top_n=20):
        """Show which features matter most."""
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print('='*60)
        print(f"  Top {top_n} Most Important Features")
        print("="*60)
        print(importance_df.head(top_n).to_string(index=False))
        
        # Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'].head(top_n), 
                importance_df['importance'].head(top_n))
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(MODEL_DIR / 'feature_importance.png', dpi=150)
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
        self.split_data(PROCESSED_PATH)
        
        # 2. Baseline
        print("\n" + "="*60)
        print("  STEP 2: BASELINE MODEL")
        print("="*60)
        baseline_metrics_train = evaluation.baseline_naive_seasonal(self.train,split="train")
        baseline_metrics_val = evaluation.baseline_naive_seasonal(self.val,split="val")
        baseline_metrics_test = evaluation.baseline_naive_seasonal(self.test,split="test")
        
        # 3. Fine tune hyper parameter of model
        print("\n" + "="*60)
        print("  STEP 3: Fine tuning hyperparameters")
        print("="*60)
        self.tune_hyperparameters()
        


        # 4. Train model
        print("\n" + "="*60)
        print("  STEP 4: TRAINING LIGHTGBM")
        print("="*60)
        self.train_model()

        # 5. Evaluate
        print("\n" + "="*60)
        print("  STEP 5: EVALUATION")
        print("="*60)
        train_metrics=evaluation.evaluate_model(self.train, self.model, 'train')
        val_metrics=evaluation.evaluate_model(self.val, self.model, 'val')
        test_metrics = evaluation.evaluate_model(self.test, self.model, 'test')
        
        # 5. Feature importance
        print("\n" + "="*60)
        print("  STEP 5: FEATURE ANALYSIS")
        print("="*60)
        importance_df = self.analyze_feature_importance()
        
        # 6. Residual analysis
        print("\n" + "="*60)
        print("  STEP 6: ERROR ANALYSIS")
        print("="*60)
        evaluation.analyze_residuals(self.test, 'Test')


         # 6. Residual analysis
        print("\n" + "="*60)
        print("  STEP 7:save model and features")
        print("="*60)
        self.save_best_model()
        
        # 7. Save everything
        self.save_artifacts()
        
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
            'feature importance': importance_df
        }
    
