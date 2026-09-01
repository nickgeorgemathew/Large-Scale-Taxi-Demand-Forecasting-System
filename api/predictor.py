import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json
from schema import HourlyForecast
from monitoring.prediction_logger import PredictionLogger
from config.settings import LOG, HOTSPOTS, PUBLIC_HOLIDAYS_2022,FEATURE_COLUMNS

class TaxiForecaster:

    def __init__(self, model_path, feature_path, recent_history_path,
                 quantile_low_model_path, quantile_high_model_path):
        self.model = joblib.load(model_path)
        self.quantile_low = joblib.load(quantile_low_model_path)
        self.quantile_high = joblib.load(quantile_high_model_path)
        if feature_path:
            with open(feature_path, "r") as f:
                self.feature_cols = json.load(f)
        else:
            self.feature_cols=FEATURE_COLUMNS
        self.recent_history_df = pd.read_csv(recent_history_path)#add try except if file not found
        self.recent_history_df['timestamp'] = pd.to_datetime(self.recent_history_df['timestamp'])
        self.logger = PredictionLogger()

    def predict(self, zone_id, hours_ahead):
        results = []
        history_buffer = self.recent_history_df[self.recent_history_df["zone_id"] == zone_id].copy()
        history_buffer = history_buffer[["timestamp", "zone_id", "predicted_demand"]].tail(200)

        current_hour = pd.Timestamp.now().floor('h')
        for step in range(1, hours_ahead + 1):
            target_time = current_hour + timedelta(hours=step)
            features_df, features_dict = self.build_feature_row(zone_id, target_time, history_buffer)
            pred = self.model.predict(features_df)[0]
            pred = max(0, pred)
            pred_low = self.quantile_low.predict(features_df)[0]
            pred_high = self.quantile_high.predict(features_df)[0]

            results.append(HourlyForecast(
                timestamp=target_time,
                zone_id=zone_id,
                predicted_demand=pred,
                lower_bound=pred_low,
                upper_bound=pred_high,
                features=features_dict
            ))
            # recursive update
            new_row = pd.DataFrame([{"timestamp": target_time, "zone_id": zone_id, "predicted_demand": pred}])
            history_buffer = pd.concat([history_buffer, new_row], ignore_index=True)

        self.logger.save_logs_parquet([r.dict() for r in results], LOG)
        return results

    def build_feature_rows_all_zones(self, timestamp, valid_zones):
        all_rows = []
        for zone_id in valid_zones:
            hist = self.recent_history_df[self.recent_history_df["zone_id"] == zone_id].copy()
            row_df, _ = self.build_feature_row(zone_id, timestamp, hist)
            all_rows.append(row_df)
        return pd.concat(all_rows, ignore_index=True)

    def predict_hotspots(self, zone_df, timestamp):
        """zone_df: DataFrame with features for each zone at the same timestamp."""
        preds = self.model.predict(zone_df[self.feature_cols])
        preds = np.clip(preds, 0, None)
        low_preds = self.quantile_low.predict(zone_df[self.feature_cols])
        high_preds = self.quantile_high.predict(zone_df[self.feature_cols])
        zone_df["predicted_demand"] = preds
        zone_df["lower_bound"] = low_preds
        zone_df["upper_bound"] = high_preds
        zone_df["timestamp"] = timestamp
        self.logger.save_logs_parquet(zone_df.to_dict(orient="records"), HOTSPOTS)
        return zone_df

    def build_feature_row(self, zone_id, timestamp, history_buffer):
        row = {}
        timestamp = pd.to_datetime(timestamp).floor('h')
        history_buffer['timestamp'] = pd.to_datetime(history_buffer['timestamp']).dt.floor('h')

        row['hour_of_day'] = timestamp.hour
        row['day_of_week'] = timestamp.weekday()
        row['zone_id'] = zone_id
        row['month'] = timestamp.month
        row['is_weekend'] = int(row['day_of_week'] >= 5)
        row['is_rush_am'] = int(7 <= row['hour_of_day'] <= 9)
        row['is_rush_pm'] = int(17 <= row['hour_of_day'] <= 19)
        row['is_night'] = int(row['hour_of_day'] >= 22 or row['hour_of_day'] <= 5)

        holidays = pd.to_datetime(PUBLIC_HOLIDAYS_2022)
        row['is_holiday'] = int(timestamp.date() in holidays.date)

        # lags
        for lag in [1, 2, 3, 6, 24, 48, 168]:
            lookback = timestamp - timedelta(hours=lag)
            match = history_buffer[history_buffer['timestamp'] == lookback]
            row[f'lag_{lag}h'] = float(match['predicted_demand'].iloc[-1]) if not match.empty else 0.0

        # rolling 6h mean
        values = []
        for i in range(1, 7):
            lookback = timestamp - timedelta(hours=i)
            match = history_buffer[history_buffer['timestamp'] == lookback]
            if not match.empty:
                values.append(float(match['predicted_demand'].iloc[-1]))
        row['roll_mean_6h'] = np.mean(values) if values else 0.0

        row_df = pd.DataFrame([row]).reindex(columns=self.feature_cols, fill_value=0)
        return row_df, row