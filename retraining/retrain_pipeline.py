import argparse
import logging
import sys
from pathlib import Path
import joblib
import pandas as pd
from config.settings import PROCESSED_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from models.train import ModelTrainer
from pyspark.sql import SparkSession
from models.evaluate import Evaluate
from monitoring.model_registry_manager import ModelRegistry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_retraining_pipeline():
    lock = Path("flags/retrain.lock")
    if not lock.exists():
        logging.error("No lock file found; retraining should be triggered by alert manager.")
        return

    try:
        logging.info("Starting retraining pipeline...")
        # 1. Create Spark session and load data
        
        spark = SparkSession.builder.appName("RetrainPipeline").getOrCreate()
        df = spark.read.parquet(PROCESSED_PATH)

        # 2. Train new model using ModelTrainer
        trainer = ModelTrainer(spark)
        trainer.split_data(df)        # sets train_pd, val_pd, test_pd
        trainer.tune_hyperparameters(n_trials=20)  # quick tuning
        model = trainer.train_model()
        quantile_low = trainer.train_quantile_low_model()
        quantile_high = trainer.train_quantile_high_model()

        # 3. Evaluate on test set
        evaluator = Evaluate()
        test_metrics = evaluator.evaluate_model(trainer.test_pd, model, split="test_retrain")
        logging.info(f"New model test metrics: {test_metrics}")

        # 4. Compare with current production model (optional)
        registry = ModelRegistry()
        current_model_path = registry.get_active_model_path()
        if current_model_path.exists():
            old_model = joblib.load(current_model_path)
            old_metrics = evaluator.evaluate_model(trainer.test_pd, old_model, split="old_test")
            if test_metrics['rmse'] < old_metrics['rmse']:
                registry.promote(f"lgbm_demand_v{trainer.version}.pkl")
                logging.info("New model outperforms old one, promoted to production.")
            else:
                logging.warning("New model not better, skipping promotion.")
        else:
            registry.promote(f"lgbm_demand_v{trainer.version}.pkl")
            logging.info("First model, promoted to production.")

        # 5. Save artifacts
        trainer.save_best_model()
        logging.info("Retraining completed successfully.")

    except Exception as e:
        logging.error(f"Retraining failed: {e}")
        raise
    finally:
        if lock.exists():
            lock.unlink()
            logging.info("Lock removed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", default="unspecified", help="Reason for retraining")
    args = parser.parse_args()
    logging.info(f"Retraining triggered because: {args.reason}")
    run_retraining_pipeline()

if __name__ == "__main__":
    main()