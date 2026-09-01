import pandas as pd
from pathlib import Path

class PredictionLogger:
    "to be used in a pipeline and in the backend"

    @staticmethod
    def save_logs_parquet(logs, path):
        """Append logs (list of dicts or single dict) to a parquet file."""
        path = Path(path)
        df_logs = pd.DataFrame(logs if isinstance(logs, list) else [logs])

        try:
            df_existing = pd.read_parquet(path)
            # simple concatenation – duplicates are responsibility of caller
            df_combined = pd.concat([df_existing, df_logs], ignore_index=True)
            df_combined.to_parquet(path, engine='pyarrow')
        except FileNotFoundError:
            df_logs.to_parquet(path, engine='pyarrow')

        return logs

    @staticmethod
    def log_predictions(features: dict, prediction, model_version, path):
        "save the predictions with metadata to a file"
        timestamp = pd.to_datetime(features["timestamp"])
        zone_id = features["zone_id"]
        log_dict = {
            "timestamp": timestamp,
            "model_version": model_version,
            "zone_id": zone_id,
            "prediction": prediction,
        }
        PredictionLogger.save_logs_parquet(log_dict, path)

    @staticmethod
    def log_predictions_actual(features: dict, prediction, actual, model_version, path):
        timestamp = pd.to_datetime(features["timestamp"])
        zone_id = features["zone_id"]
        residual = (actual - prediction) if actual is not None else None
        log_dict = {
            "timestamp": timestamp,
            "model_version": model_version,
            "residual": residual,
            "zone_id": zone_id,
            "prediction": prediction,
            "actual": actual if actual is not None else None
        }
        PredictionLogger.save_logs_parquet(log_dict, path)

    @staticmethod
    def load_prediction_logs(log_path):
        return pd.read_parquet(log_path)



if __name__=="__main__":
    p=PredictionLogger()
    p.load_prediction_logs()