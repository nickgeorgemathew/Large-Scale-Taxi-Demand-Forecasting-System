from config.settings import BEST_MODEL_PATH,BEST_MODEL_NAME
from monitoring.drift_detector import DriftDetector,run_drift_pipeline
from monitoring.metrics_monitor import MetricsMonitor,run_monitor_pipeline
from monitoring.alert_manager import AlertManager,run_alert_manager_condition,run_alert_manager_trigger_action
from monitoring.mlops_pipeline import Pipeline,run_mlops_pipeline
from retraining.model_registry_manager import ModelRegistry 
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from config.settings import SERVING_HALTED, MONITORING_CONFIG,BEST_MODEL_NAME



metrics=MetricsMonitor()
alert=AlertManager()
drift=DriftDetector()
mlops=Pipeline()
model_registry=ModelRegistry()
class ModelopsPipeline:


    @staticmethod
    def model_metrics_drift():
        "get the metrics and drift/residuals of the model and return performance_flag,metric_week,metric_24,drift_flag,feature_drift_flag,residual_drift_flag"
        performance_flag,metric_week,metric_24=run_monitor_pipeline()
        drift_flag,feature_drift_flag,residual_drift_flag=run_drift_pipeline()
        return performance_flag,metric_week,metric_24,drift_flag,feature_drift_flag,residual_drift_flag

    @staticmethod
    def model_health(drift_flag,performance_flag):
        "get the health/condition of the model and return confition_flag"
        condition_flag=run_alert_manager_condition(drift_flag,performance_flag)
        return condition_flag


    @staticmethod
    def model_action(condition_flag):
        "trigger mlops actions based on the condition flag and return metadata "
        trigger_action=run_alert_manager_trigger_action(condition_flag)
        return trigger_action

    class ManualMlopsActions:
        "a subclass for more control over choice of  what Mlops action occurs,use to trigger actions manually than through automated pipeline"

            
        @staticmethod
        def halt_serving(flag: bool, reason: str, timestamp: datetime):
            """Stop or resume serving predictions."""
            halted = {
                "serving_halted": flag,
                "reason": reason,
                "halted_at": timestamp.isoformat() if flag else None
            }
            with open(SERVING_HALTED, "w") as f:
                json.dump(halted, f, indent=2)
            model_registry.rollback()
        @staticmethod
        def trigger_retrain( reason: str = "unspecified"):
            """Asynchronously trigger retraining with a lock to prevent concurrent runs."""
            lock = Path("flags/retrain.lock")
            lock.parent.mkdir(exist_ok=True)

            if lock.exists():
                logging.warning("Retrain already in progress, skipping")
                return

            lock.touch()
            try:
                subprocess.Popen(
                    ["python", "pipeline/retrain_pipeline.py", "--reason", reason],
                    stdout=open("logs/retrain.log", "a"),
                    stderr=subprocess.STDOUT
                )
                logging.info("Retrain triggered asynchronously")
            except Exception as e:
                logging.error(f"Failed to start retrain: {e}")
                lock.unlink(missing_ok=True)
        @staticmethod
        def increase_monitoring_frequency():
            """Switch monitoring to high‑frequency mode via config file."""
            config_path = Path(MONITORING_CONFIG)
            if config_path.exists():
                config = json.loads(config_path.read_text())
            else:
                config = {"interval_minutes": 30, "high_alert_mode": False}
            config["interval_minutes"] = 5
            config["high_alert_mode"] = True
            config_path.write_text(json.dumps(config, indent=2))
            logging.info("Monitoring frequency increased to 5 minutes")
        @staticmethod
        def flag_for_review():
            """Add a review flag to the queue for manual inspection."""
            flag = {
                "timestamp": datetime.now().isoformat(),
                "severity": "WATCH",
                "model": BEST_MODEL_NAME,
                "reason": "feature drift detected, performance within threshold",
                "action_required": "manual review",
                "resolved": False
            }
            flags_path = Path("flags/review_queue.json")
            existing = json.loads(flags_path.read_text()) if flags_path.exists() else []
            existing.append(flag)
            flags_path.write_text(json.dumps(existing, indent=2))
            logging.info("Review flag added")
        




    



    
        

       
                
        
