import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from pathlib import Path
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





class Evaluate:
    def __init__(self):
        self.metrics={}
        self.baseline={}


    def compute_metrics(self,y_true, y_pred, label:str='',print_metrics:bool=False):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        smape=self.smape(y_true, y_pred)
        if print_metrics is True:
            print(f"\n{'='*60}")
            if label:
                print(f"  {label}")
            print(f"{'='*60}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²:   {r2:.4f}")
            print(f"  SMAPE:   {smape:.4f}")              
        
        return {'label':label,'mae': mae, 'rmse': rmse, 'r2': r2,'smape':smape} if label else {'mae': mae, 'rmse': rmse, 'r2': r2,'smape':smape}
    



    def baseline_naive_seasonal(self,df:pd.DataFrame,split:str=''):

        preds = df['lag_168h']
        metrics=self.compute_metrics(df['demand'], preds, label='Naive Seasonal Baseline')
        self.baseline[split]=metrics
        
        return metrics
    
    
    
    
    def smape(self,y_true, y_pred, epsilon=1e-10): 
        y_true = np.array(y_true) 
        y_pred = np.array(y_pred) 
        numerator = 2 * np.abs(y_true - y_pred) 
        denominator = np.abs(y_true) + np.abs(y_pred) + epsilon 
        smape_value = np.mean(numerator / denominator) * 100 
        return smape_value
    
    def plot_metrics(self,model_name,split:str=""):
        fig,axs=plt.subplots(2, 2, figsize=(10,8))
        axs=axs.flatten()
        baseline_metrics=self.baseline[split]
        model_metrics=self.metrics[split]
        metric_names = ["mae", "rmse", "r2", "smape"]


        for i,val in enumerate(metric_names):
            values=[baseline_metrics[val],model_metrics[val]]
            axs[i].bar(["baseline","model"],values)
            axs[i].set_title(val.upper())
            axs[i].set_ylabel(val.upper())
        plt.title(f"{model_name}_{split}_metric_plot")
        plt.savefig(f"models/artifacts/{model_name}_{split}_metric_plot.png")
        plt.tight_layout()
        



    def evaluate_model(self,df:pd.DataFrame,model,model_name,split:str=''):
        y_true = df['demand']
        y_pred = model.predict(df[FEATURE_COLUMNS])
        y_pred = np.clip(y_pred,0,None)  
        
        print(f"evaluation for {split}")
        
        baseline=self.baseline[split]
        
        # Global metrics
        MAE  = mean_absolute_error(y_true, y_pred)
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        SMAPE=self.smape(y_true,y_pred)
        model_metrics={'mae': MAE, 'rmse': RMSE, 'r2': r2,'smape':SMAPE}
        
        self.metrics[model_name][split]=model_metrics
        self.save_metrics(split,model_name)
        
         # Compare to baseline
        print(f"Model MAE: {MAE:.2f} | Baseline MAE: {baseline['mae']:.2f}")
        print(f"Model RMSE: {RMSE:.2f} | Baseline RMSE: {baseline['rmse']:.2f}")
        print(f"Model r2: {r2:.2f} | Baseline r2: {baseline['r2']:.2f}")
        print(f"Model SMAPE: {SMAPE:.2f} | Baseline SMAPE: {baseline['smape']:.2f}")

        self.plot_metrics(split=split,model_name=model_name)
    

    
    
        
        
        
        
    def save_metrics(self,split:str='',model_name:str=''):
        with open(MODEL_DIR/f"{model_name}model_metrics_{split}.json",'w') as f:
            json.dump(self.metrics[split],f)
        
        
        
        
        
        
        
    def analyze_residuals(self,df,split_name:str,model):
    
        X = df[FEATURE_COLUMNS]
        self.y_true = df['demand']
        self.y_pred = model.predict(X)
        
        
        residuals = self.y_true - self.y_pred
        
        # By hour
        df['residual'] = residuals
        hourly_error = df.groupby('hour_of_day')['residual'].agg(['mean', 'std'])
        
        print(f"\n{'='*60}")
        print(f"  Error by Hour of Day({split_name})")
        print(f"{'='*60}")
        print(hourly_error)
        
        # By zone (find worst zones)
        self.zone_error = df.groupby('zone_id').agg({
            'residual': ['mean', 'std', 'count']
        }).round(2)
        self.zone_error.columns = ['mean_error', 'std_error', 'count']
        self.zone_error = self.zone_error.sort_values('mean_error', key=abs, ascending=False)
        
        print(f"\n{'='*60}")
        print(f"  Top 10 Worst Zones (Highest Error)")
        print(f"{'='*60}")
        print(self.zone_error.head(10))


        
        
        
    

    # def evaluation_plot(self,df):
        
    #     zone_errors=self.zone_error.copy()
    #     zone_errors['abs_error']=np.abs(zone_errors['residual'])
        
    #     hour_error=zone_errors.groupby('hour_of_day')['abs_error'].mean()
    #     hour_error.plot(kind='bar',figsize=(10,6))
    #     plt.title("Average Prediction Error by Hour")
    #     plt.xlabel("Hour of Day")
    #     plt.ylabel("Mean Absolute Error")
    #     plt.show()


    #     plt.scatter(self.y_true, self.y_pred, alpha=0.3)
    #     plt.plot(
    #         [self.y_true.min(), self.y_true.max()],
    #         [self.y_true.min(), self.y_true.max()],
    #         color="red"
    #     )

    #     plt.xlabel("Actual Demand")
    #     plt.ylabel("Predicted Demand")
    #     plt.title("Prediction vs Actual")
    #     plt.show()


        
    #     plt.figure(figsize=(12,4))
    #     plt.plot(zone_errors["timestamp"], residuals)
    #     plt.axhline(0, color="red")
    #     plt.title("Residuals Over Time")
    #     plt.xlabel("Time")
    #     plt.ylabel("Residual")
    #     plt.show()
                
        
        
        
        
        
        
      

    def generate_evaluation_report(self, df, model, split="test"):

        print(f"\n{'='*70}")
        print(f"MODEL EVALUATION REPORT ({split})")
        print(f"{'='*70}")

        X = df[FEATURE_COLUMNS]
        y_true = df["demand"]
        y_pred = model.predict(X)

        y_pred = np.clip(y_pred, 0, None)

        residuals = y_true - y_pred
        abs_error = np.abs(residuals)

        df = df.copy()
        df["prediction"] = y_pred
        df["residual"] = residuals
        df["abs_error"] = abs_error

        # -------------------------------
        # 1. Prediction vs Actual
        # -------------------------------

        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, alpha=0.3)

        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            color="red"
        )

        plt.xlabel("Actual Demand")
        plt.ylabel("Predicted Demand")
        plt.title("Prediction vs Actual")
        plt.show()

        # -------------------------------
        # 2. Residual Distribution
        # -------------------------------

        plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=50)
        plt.title("Residual Distribution")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.show()

        # -------------------------------
        # 3. Residual vs Prediction
        # -------------------------------

        plt.figure(figsize=(6,4))
        plt.scatter(y_pred, residuals, alpha=0.3)

        plt.axhline(0, color="red")

        plt.xlabel("Prediction")
        plt.ylabel("Residual")
        plt.title("Residual vs Prediction")
        plt.show()

        # -------------------------------
        # 4. Error by Hour
        # -------------------------------

        per_hour = df.groupby("hour_of_day")["abs_error"].mean()

        plt.figure(figsize=(10,4))
        per_hour.plot(kind="bar")

        plt.title("Mean Absolute Error by Hour")
        plt.xlabel("Hour of Day")
        plt.ylabel("MAE")
        plt.show()

        # -------------------------------
        # 5. Worst Zones
        # -------------------------------

        per_zone = df.groupby("zone_id")["abs_error"].mean()

        worst_zones = per_zone.sort_values(ascending=False).head(10)

        print("\nWorst Zones (Highest Error):")
        print(worst_zones)

        worst_zones.plot(kind="bar", figsize=(10,4))
        plt.title("Top 10 Worst Zones by Error")
        plt.ylabel("Mean Absolute Error")
        plt.show()

        # -------------------------------
        # 6. Residuals Over Time
        # -------------------------------

        if "timestamp" in df.columns:

            plt.figure(figsize=(12,4))

            plt.plot(df["timestamp"], residuals)

            plt.axhline(0, color="red")

            plt.title("Residuals Over Time")
            plt.xlabel("Time")
            plt.ylabel("Residual")
            plt.show()

        print("\nReport Complete")