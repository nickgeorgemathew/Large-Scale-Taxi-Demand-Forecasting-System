from config.settings import BEST_MODEL_PATH,BEST_MODEL_NAME
from monitoring.drift_detector import DriftDetector,run_drift_pipeline
from monitoring.metrics_monitor import MetricsMonitor,run_monitor_pipeline
from monitoring.alert_manager import AlertManager



metrics=MetricsMonitor()
alert=AlertManager()
drift=DriftDetector()

class ModelopsPipeline:
    def __init__(self):
        pass

    @staticmethod
    def model_drift_metrics_status():
        performance_flag,metric_week,metric_24=run_monitor_pipeline()
        drift_flag=run_drift_pipeline()
        AlertManager.assess_performance()

       
                
        
