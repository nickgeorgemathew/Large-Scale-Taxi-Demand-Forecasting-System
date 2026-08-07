import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from models.evaluate import Evaluate
from config.settings import LOG, METRICLOG,BASELINE_METRICS_PATH

class MetricsMonitor:

    def __init__(self):
        self.evaluate = Evaluate()
        self.log_path = LOG
        self.metric_path = METRICLOG.parent  # directory for metrics

    def load_logs(self):
        try:
            self.log_df = pd.read_parquet(self.log_path)
            # ensure required columns exist
            if 'actual' not in self.log_df.columns:
                self.log_df['actual'] = None
            return self.log_df
        except FileNotFoundError:
            raise FileNotFoundError(f"No prediction logs found at {self.log_path}")

    def compute_metrics(self, y_true, y_pred, label=""):
        return self.evaluate.compute_metrics(y_true, y_pred, label)

    def compute_rolling_metrics(self):
        df = self.load_logs()
        df = df.dropna(subset=['prediction'])
        if df.empty:
            return None, None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        now = datetime.now()
        window_24h = now - timedelta(hours=24)
        window_week = now - timedelta(days=7)

        df_24h = df[df['timestamp'] > window_24h].dropna(subset=['actual'])
        df_week = df[df['timestamp'] > window_week].dropna(subset=['actual'])

        metrics_24 = None
        if len(df_24h) >= 10:
            metrics_24 = self.compute_metrics(df_24h['actual'], df_24h['prediction'], label="rolling_24h")
            self._save_metric(metrics_24, "rolling_24h")

        metrics_week = None
        if len(df_week) >= 10:
            metrics_week = self.compute_metrics(df_week['actual'], df_week['prediction'], label="rolling_week")
            self._save_metric(metrics_week, "rolling_week")

        return metrics_week, metrics_24

    def _save_metric(self, metrics, name):
        path = self.metric_path / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def detect_performance_drift(self, filename, threshold=1.3):
        """Compare rolling metrics with baseline test metrics."""
        baseline_path = BASELINE_METRICS_PATH
        if not baseline_path.exists():
            raise FileNotFoundError("Baseline metrics not found. Train model first.")

        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        rolling_path = self.metric_path / filename
        if not rolling_path.exists():
            return {"mae_drift": False, "rmse_drift": False, "r2_drift": False, "smape_drift": False}

        with open(rolling_path, 'r') as f:
            rolling = json.load(f)

        metric_names = ["mae", "rmse", "smape"]  # r2 higher is better, handle separately
        performance_flags = {f"{m}_drift": False for m in metric_names}
        performance_flags["r2_drift"] = False

        for m in metric_names:
            if rolling[m] > baseline[m] * threshold:
                performance_flags[f"{m}_drift"] = True

        # For R2, drift if rolling R2 < baseline R2 * threshold (threshold < 1)
        if rolling.get('r2', 0) < baseline.get('r2', 0) * (2 - threshold):  # e.g., threshold=1.3 -> factor=0.7
            performance_flags["r2_drift"] = True

        return performance_flags




if __name__=="__main__":
    spark=create_spark_session()
    detector=DriftDetector(spark=spark)
    prev_df=detector.load_data(PATH_PREV_TRAIN_DATA)
    current_df=detector.load_data(FEATURES_PATH)
    train_prev_df,val_prev_df,test_prev_df=detector.split_data(prev_df)
    train_current_df,val_current_df,test_current_df=detector.split_data(current_df)

    feature_drift_flag=detector.compute_feature_drift(train_current_df,train_prev_df)
    print("===="*60)
    print("feature drift flag")
    print("===="*60)
    print(feature_drift_flag)

    residual_drift_flag=detector.compute_residual_drift(current_df)
    print("===="*60)
    print("residual_drift_flag")
    print("===="*60)
    print(residual_drift_flag)
    
    drift_flag=detector.detect_drift(feature_drift_flag,residual_drift_flag)
    print("===="*60)
    print("drift_flag")
    print("===="*60)
    print(drift_flag)
