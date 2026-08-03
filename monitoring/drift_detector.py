import numpy as np
import pandas as pd
from spark_create import create_spark_session
from config.settings import MONITORED_FEATURES,TEST_START_DATE,DATA_START_DATE,DATA_END_DATE,VAL_END_DATE,TRAIN_END_DATE,RAW_DATA_PATH

def calculate_psi(expected, actual, n_bins=10):
    """Population Stability Index."""
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)
    expected_percents[expected_percents == 0] = 1e-6
    actual_percents[actual_percents == 0] = 1e-6
    psi_val = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_val)

class DriftDetector:

    def __init__(self, spark=None):
        self.spark = spark  # optional, not used for pandas

    def compute_feature_drift(self, training_df, current_df, threshold=0.1):
        feature_flag = {}
        for feature in MONITORED_FEATURES:
            if feature not in training_df.columns or feature not in current_df.columns:
                continue
            train_vals = training_df[feature].dropna().values
            curr_vals = current_df[feature].dropna().values
            if len(train_vals) == 0 or len(curr_vals) == 0:
                psi = 0.0
            else:
                psi = calculate_psi(train_vals, curr_vals)
            feature_flag[f"{feature}_drift"] = psi > threshold
        return feature_flag

    def compute_residual_drift(self, df, threshold=0.5):
        if 'actual' not in df.columns or 'prediction' not in df.columns:
            return {"residual_drift": False}
        residual = df['actual'] - df['prediction']
        mean_residual = np.mean(residual)
        return {"residual_drift": abs(mean_residual) > threshold}

    def detect_drift(self, feature_drift_flags, residual_drift_flag):
        feature_detected = any(v for k, v in feature_drift_flags.items() if k != "residual_drift")
        residual_detected = residual_drift_flag.get("residual_drift", False)
        drift_flag = {"feature_drift": feature_detected, "residual_drift": residual_detected}
        if feature_detected:
            drift_flag["features"] = [k for k, v in feature_drift_flags.items() if v and k != "residual_drift"]
        return drift_flag



if __name__=="__main__":
    spark=create_spark_session()
    
