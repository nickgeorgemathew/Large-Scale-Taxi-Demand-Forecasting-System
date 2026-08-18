from config.settings import BEST_MODEL_PATH,BEST_MODEL_NAME
from monitoring.drift_detector import DriftDetector
from monitoring.metrics_monitor import MetricsMonitor
from monitoring.alert_manager import AlertManager



metrics=MetricsMonitor()
alert=AlertManager()
metrics.detect_performance_drift()
class ModelopsPipeline:
    def __init__(self):
        pass

    @staticmethod
    def model_drift_metrics_status():
        performance_flag,metric_week,metric_24=metrics.run_monitor_pipeline()
        AlertManager.assess_performance()

       
                
        
