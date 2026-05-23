import json
import pandas as pd
from config.settings import SERVING_HALTED

class Pipeline:
     


 def halt_serving(self,flag,reason,timestamp):
    halted={"serving_halted":flag,"reason":reason,"halted_at":timestamp}
    
    with open(SERVING_HALTED,"w")as f:
         json.dump(halted,f)
         








def trigger_retrain() :
        trigger_retrain()
    This calls RetrainPipeline.run(). The question is: synchronously or asynchronously?
    Synchronous (blocks alert_manager until retrain finishes):
    pythondef trigger_retrain(self, reason="unspecified"):
        pipeline = RetrainPipeline(config=...)
        pipeline.run()
    Problem: retrain takes 20 minutes. Alert manager is frozen.
    Asynchronous via subprocess (correct approach):
    pythonimport subprocess

    def trigger_retrain(self, reason="unspecified"):
        result = subprocess.Popen(
            ["python", "pipeline/retrain_pipeline.py", "--reason", reason],
            stdout=open("logs/retrain.log", "a"),
            stderr=subprocess.STDOUT
        )
        # Popen returns immediately — retrain runs in background
    With lock file to prevent concurrent retrains:
    pythondef trigger_retrain(self, reason="unspecified"):
        lock = Path("flags/retrain.lock")
        if lock.exists():
            logging.warning("Retrain already in progress, skipping")
            return
        lock.touch()
        try:
            subprocess.Popen(["python", "pipeline/retrain_pipeline.py"])
        finally:
            pass  # lock removed by retrain_pipeline.py at end of run
    retrain_pipeline.py must call Path("flags/retrain.lock").unlink() as its final step.









      # kick off retraining job



increase_monitoring_frequency():
        increase_monitoring_frequency()
    Your monitoring interval lives in your crontab or a config file. Python can't edit crontab live cleanly. The correct pattern: write a config value that the monitoring script reads.
    python# monitoring_config.json
    {"interval_minutes": 30, "high_alert_mode": false}
    pythondef increase_monitoring_frequency(self):
        config_path = Path("config/monitoring_config.json")
        config = json.loads(config_path.read_text())
        config["interval_minutes"] = 5
        config["high_alert_mode"] = True
        config_path.write_text(json.dumps(config, indent=2))
    Then run_monitoring.py reads this config at startup to decide whether to skip a run (if called every 5 min by cron but high_alert_mode is False, skip).
    Alternatively — maintain two crontab entries, one commented:
    bash# normal mode
    */30 * * * * python run_monitoring.py

    # high alert mode — uncomment when needed
    # */5 * * * * python run_monitoring.py
    And switch between them programmatically via crontab subprocess call. But the JSON config approach is cleaner.






flag_for_review():
                flag_for_review()
                This writes a structured record that a human (or downstream process) can inspect.
                python
                def flag_for_review(self):
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
                This is the foundation of a review queue. A dashboard or Slack notifier can read review_queue.json and surface unresolved flags.
