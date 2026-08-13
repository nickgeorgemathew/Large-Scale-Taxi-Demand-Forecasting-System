import pandas as pd
from datetime import datetime
from pathlib import Path
from config.settings import PERFORMANCELOG, METRICLOG,MODELCONDITIONLOG,BEST_MODEL_NAME
from  monitoring.mlops_pipeline import Pipeline as pipeline
from retraining.model_registry_manager import ModelRegistry as model_registry

def save_log_parquet(log, file_path):
    "file path must be a path that leads to the log file and not just a string"
    df_log = pd.DataFrame([log])
    try:
        df_existing = pd.read_parquet(file_path)
        df_combined = pd.concat([df_existing, df_log], ignore_index=True)
        df_combined.to_parquet(file_path, engine='pyarrow')
    except FileNotFoundError:
        df_log.to_parquet(Path(file_path), engine="pyarrow")
    return df_log

class AlertManager:

    def log_state(self, flag):
        "log the current state  with metadata"
        current_time = datetime.now().isoformat()
        log = {"timestamp": current_time, "severity": flag["severity"], "action": flag["action"]}
        save_log_parquet(log, PERFORMANCELOG)

    def model_condition_log(self, flag):
        "log the current state of the model and save it to MODELCONDITIONLOG"
        current_time = datetime.now().isoformat()
        log = {
            "timestamp": current_time,
            "severity": flag["severity"],
            "action triggered": flag["action"],
            "model state": flag["model_state"]
        }
        save_log_parquet(log, MODELCONDITIONLOG)
        return log

    def assess_performance(self, performance_flags):
        return [k for k, val in performance_flags.items() if val]

    def assess_drift(self, drift_flags):
        feature = [k for k, val in drift_flags.items() if val and k != "residual_drift"]
        residual = [k for k, val in drift_flags.items() if val and k == "residual_drift"]
        return feature, residual

    def assess_condition(self, performance_flags, drift_flags):
        performance = self.assess_performance(performance_flags)
        feature, residual = self.assess_drift(drift_flags)

        condition_flags = {"severity": "", "action": ""}

        if performance and feature and residual:
            condition_flags.update(severity="CRITICAL", action="RETRAIN MODEL IMMEDIATELY ! ROLLBACK")
        elif not performance and not feature and not residual:
            condition_flags.update(severity="OK", action="NONE")
        elif performance and feature:
            condition_flags.update(severity="RETRAIN", action="RETRAIN MODEL, PERFORMANCE AND FEATURE DRIFT DETECTED")
        elif feature and residual:
            condition_flags.update(severity="WARNING", action="MONITOR FEATURES AND RESIDUAL")
        elif performance and residual:
            condition_flags.update(severity="CRITICAL", action="INVESTIGATE AND RETRAIN (performance and residual)")
        elif performance:
            condition_flags.update(severity="WARNING", action="INVESTIGATE PERFORMANCE")
        elif feature:
            condition_flags.update(severity="WATCH", action="MONITOR FEATURES")
        elif residual:
            condition_flags.update(severity="WARNING", action="INVESTIGATE BIAS(residual)")

        self.log_state(condition_flags)
        return condition_flags

    def trigger_action(self, condition_flags):
        severity = condition_flags["severity"]
        action = condition_flags["action"]
        timestamp = datetime.now()

        if severity == "CRITICAL":
            
            pipeline.halt_serving(flag=True, reason=severity, timestamp=timestamp)
            pipeline.trigger_retrain(reason=severity)
            model_registry.rollback()
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "ROLLED_BACK","model": BEST_MODEL_NAME},
                
            )
        elif severity == "RETRAIN":
            pipeline.trigger_retrain(reason=severity)
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "RETRAINING","model": BEST_MODEL_NAME},
               
            )
        elif severity == "WARNING":
            pipeline.increase_monitoring_frequency()
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "DEGRADED","model": BEST_MODEL_NAME},
                
            )
        elif severity == "WATCH":
            pipeline.flag_for_review()
            self.model_condition_log(
                {"severity": severity, "action": action, "model_state": "WATCH","model": BEST_MODEL_NAME},
            
            )
        elif severity == "OK":
            self.model_condition_log(
                {"severity": severity, "action": "NONE", "model_state": "HEALTHY"}
            )