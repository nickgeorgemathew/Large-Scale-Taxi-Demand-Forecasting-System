import lightgbm
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    TARGET_COLUMN, FEATURE_COLUMNS
)


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class ModelTrainer:
    def __init__(self):
        self.best_model={}
    


    def split_data(self,df:pd.DataFrame)->pd.DataFrame:
        self.train=df[df.hour_timestamp<TRAIN_END_DATE]

        self.val=df[(df.hour_timestamp>=TRAIN_END_DATE)&df.hour_timestamp < VAL_END_DATE]
        self.test=df[df.hour_timestamp >= TEST_START_DATE]
        assert self.train.hour_timestamp.max() < self.val.hour_timestamp.min()
        assert self.val.hour_timestamp.max() < self.test.hour_timestamp.min()
        
        print(f"Train: {self.train.shape}, Val: {self.val.shape}, Test: {self.test.shape}")
        
        return self.train, self.val, self.test
    

    def compute_metrics(self,y_true, y_pred, label=''):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    
    
    def baseline_naive_seasonal(self,df:pd.DataFrame):

        preds = df['lag_168h']
        metrics=self.compute_metrics(df['demand'], preds, label='Naive Seasonal Baseline')
        
        return metrics
        
   
   
   
   
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



    def evaluate_model(self,df:pd.DataFrame,model,split:str=""):
        y_true = df['demand']
        y_pred = self.model.predict(df[FEATURE_COLUMNS])
        y_pred = pd.clip(y_pred, min=0)  
        print(f"evaluation for {split}")
        
        # Global metrics
        MAE  = mean_absolute_error(y_true, y_pred)
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        baseline=self.baseline_naive_seasonal(df)
         # Compare to baseline
        print(f"Model MAE: {MAE:.2f} | Baseline MAE: {baseline['mae']:.2f}")
        print(f"Model RMSE: {RMSE:.2f} | Baseline RMSE: {baseline['rmse']:.2f}")
        
        # Per-zone breakdown (critical for understanding where model fails)
        zone_errors = test_df.copy()
        abs=y_true - y_pred
        zone_errors['abs_error'] = -abs if abs <0 else abs
        per_zone = zone_errors.groupby('zone_id')['abs_error'].mean()
         # Identify: worst 10 zones → likely outlier zones (airports, stadiums)
        # These are good candidates for a separate specialized model
        
        # Per-hour breakdown
        zone_errors['hour'] = test_df['hour_of_day']
        per_hour = zone_errors.groupby('hour')['abs_error'].mean()
        plot per_hour → expect error peaks at 6-7am and 5-6pm (transition times)
        
        # Residuals over time
        plot y_pred vs y_true scatter (should hug diagonal)
        plot residuals (y_true - y_pred) over time (should be stationary, no drift)

    
    
    
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






    def analyze_residuals(self,df:pd.DataFrame,split_name:str):
    
        X = df[FEATURE_COLUMNS]
        y_true = df['demand']
        y_pred = self.model.predict(X)
        
        
        residuals = y_true - y_pred
        
        # By hour
        df['residual'] = residuals
        hourly_error = df.groupby('hour_of_day')['residual'].agg(['mean', 'std'])
        
        print(f"\n{'='*60}")
        print(f"  Error by Hour of Day({split_name})")
        print(f"{'='*60}")
        print(hourly_error)
        
        # By zone (find worst zones)
        zone_error = df.groupby('zone_id').agg({
            'residual': ['mean', 'std', 'count']
        }).round(2)
        zone_error.columns = ['mean_error', 'std_error', 'count']
        zone_error = zone_error.sort_values('mean_error', key=abs, ascending=False)
        
        print(f"\n{'='*60}")
        print(f"  Top 10 Worst Zones (Highest Error)")
        print(f"{'='*60}")
        print(zone_error.head(10))



    #run the entire pipeline
    def run_complete_pipeline(self):
        """Full training and evaluation pipeline."""
        
        # 1. Split data
        print("\n" + "="*60)
        print("  STEP 1: SPLITTING DATA")
        print("="*60)
        self.split_data(PROCESSED_PATH)
        
        # 2. Baseline
        print("\n" + "="*60)
        print("  STEP 2: BASELINE MODEL")
        print("="*60)
        baseline_metrics = self.baseline_naive_seasonal(self.test)
        
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
        train_metrics = self.evaluate_model(self.train, self.model, 'Train')
        val_metrics = self.evaluate_model(self.val, self.model, 'Validation')
        test_metrics = self.evaluate_model(self.test, self.model, 'Test')
        
        # 5. Feature importance
        print("\n" + "="*60)
        print("  STEP 5: FEATURE ANALYSIS")
        print("="*60)
        importance_df = self.analyze_feature_importance()
        
        # 6. Residual analysis
        print("\n" + "="*60)
        print("  STEP 6: ERROR ANALYSIS")
        print("="*60)
        self.analyze_residuals(self.test, 'Test')


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
            'baseline': baseline_metrics,
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'feature_importance': importance_df
        }
    
