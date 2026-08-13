from models.train import ModelTrainer
import joblib

model=joblib.load("C:\\/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/lgbm_demand_base model_v20260717_192254.pkl")
quantile_high_model=joblib.load("models/artifacts/quantile_high_model_v20260715_110044.pkl")
quantile_low_model=joblib.load("models/artifacts/quantile_low_model_v20260715_110044.pkl")

train=ModelTrainer()
chosen_model=train.save_best_model(models={"base_model":model,"quantile_high_model":quantile_high_model,"quantile_low_model":quantile_low_model})
print(chosen_model["model_name"])