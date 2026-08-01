import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from config.settings import SERVING_HALTED, MONITORING_CONFIG

class Pipeline:
    

    def halt_serving(self, flag: bool, reason: str, timestamp: datetime):
        """Stop or resume serving predictions."""
        halted = {
            "serving_halted": flag,
            "reason": reason,
            "halted_at": timestamp.isoformat() if flag else None
        }
        with open(SERVING_HALTED, "w") as f:
            json.dump(halted, f, indent=2)

    def trigger_retrain(self, reason: str = "unspecified"):
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

    def increase_monitoring_frequency(self):
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

    def flag_for_review(self):
        """Add a review flag to the queue for manual inspection."""
        flag = {
            "timestamp": datetime.now().isoformat(),
            "severity": "WATCH",
            "model": "taxi_demand_xgboost_v2",
            "reason": "feature drift detected, performance within threshold",
            "action_required": "manual review",
            "resolved": False
        }
        flags_path = Path("flags/review_queue.json")
        existing = json.loads(flags_path.read_text()) if flags_path.exists() else []
        existing.append(flag)
        flags_path.write_text(json.dumps(existing, indent=2))
        logging.info("Review flag added")