from config.settings import BEST_MODEL_PATH,BEST_MODEL_NAME
from monitoring.drift_detector import DriftDetector
from monitoring.metrics_monitor import MetricsMonitor
from monitoring.alert_manager import AlertManager

class ModelopsPipeline:
    def __init__(self):
        
    @staticmethod
    def model_condition_metrics():
        metrics=MetricsMonitor()
        alert=AlertManager()
        metrics.detect_performance_drift()
                
        
