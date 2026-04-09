import pandas as pd
from datetime import datetime
from pathlib import Path 
from config.settings import LOG


class prediction_logger:
    def __init__(self):
        self.log_path=Path(LOG)
        
        pass

    def log_predictions(self,features:dict,prediction,actual,model_version):
        
        #add features that are being used for prediction and monitioring
        timestamp=pd.to_datetime(features["timestamp"])
        zone_id=features["zone_id"]
        residual= (actual-prediction) if actual else None
        
        #add features and info that need to be logged
        log_dict={
            "timestamp":timestamp,
            "model_version":model_version,
            "residual": residual,
            "zone_id":zone_id,
            "prediction":prediction,
            "actual":actual
        }
        

        df_logs=pd.DataFrame(log_dict)        
        # Load, Combine, and Overwrite
        df_existing = pd.read_parquet(self.log_path)


        if df_existing:
            # perform a 'merge' with an 'indicator' to see which rows are new
            check_merge = pd.merge(
                df_logs, 
                df_existing[['timestamp', 'zone_id']], 
                on=['timestamp', 'zone_id'], 
                how='left', 
                indicator=True
            )
            # Check if any row in df_logs was found in df_existing
            is_duplicate = (check_merge['_merge'] == 'both').any()
            
            if is_duplicate:
                
                return "duplicate predictions"
            else:
                df_combined = pd.concat([df_existing, df_logs], ignore_index=True)
                df_combined.to_parquet(self.log_path, engine='pyarrow')

        else:

            df_logs.to_parquet(self.log_path,engine="pyarrow") 

        return log_dict


    def load_prediciton_logs(self,log_path):
        df=pd.read_parquet(log_path)
        return df