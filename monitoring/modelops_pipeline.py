from config.settings import BEST_MODEL_PATH,BEST_MODEL_NAME
from monitoring.drift_detector import DriftDetector,run_drift_pipeline
from monitoring.metrics_monitor import MetricsMonitor,run_monitor_pipeline
from monitoring.alert_manager import AlertManager,run_alert_manager



metrics=MetricsMonitor()
alert=AlertManager()
drift=DriftDetector()

class ModelopsPipeline:


    @staticmethod
    def model_metrics_drift():
        "get the metrics and drift/residuals of the model and return performance_flag,metric_week,metric_24,drift_flag,feature_drift_flag,residual_drift_flag"
        performance_flag,metric_week,metric_24=run_monitor_pipeline()
        drift_flag,feature_drift_flag,residual_drift_flag=run_drift_pipeline()
        return performance_flag,metric_week,metric_24,drift_flag,feature_drift_flag,residual_drift_flag

    @staticmethod
    def model_health(drift_flag,performance_flag):
        "get the health of the model and return confition_flag and trigger_action"
        condition_flag,trigger_action=run_alert_manager(drift_flag,performance_flag)
        return condition_flag,trigger_action
    



    
        

       
                
        
