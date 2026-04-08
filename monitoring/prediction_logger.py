import pandas as pd
from datetime import datetime
import pathlib 
from config.settings import LOG


class prediction_logger:
    def __init__(self):
        
        pass

    def log_predictions(self,features:dict,prediction,actual,model_version):
        
        #add features that are being used for prediction and monitioring
        timestamp=features["timestamp"]
        zone_id=features["zone_id"]
        residual= (actual-prediction) if actual else None
        
        #add features and info that need to be logged
        log_dict={
            "timestamp":timestamp,
            "model_version":model_version,
            "residual": residual,
            "zone_id":zone_id,
            "prediction":prediction,
            "actual_value":actual
        }
        

        df_logs=pd.DataFrame(log_dict)        
        # Load, Combine, and Overwrite
        df_existing = pd.read_parquet(LOG)

        if df_existing:
            df_combined = pd.concat([df_existing, df_logs], ignore_index=True)
            df_combined.to_parquet(LOG, engine='pyarrow')
        else:
            df_logs.to_parquet(LOG,engine="pyarrow") 

        return log_dict


    def load_prediciton_logs(self):
        pass