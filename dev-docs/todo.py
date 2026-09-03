
def CONVERT_CONFIGSETTINGS() :
    pass
    #TO A YAML HELPER FILE FOR  PERSISTENT CONFIGS:

   
   
            #     Using a YAML configuration file is the industry-standard way to solve this problem in MLOps. It acts as a single, highly readable source of truth that anyone can open, understand, and tweak without touching Python code.
            # By using the ruamel.yaml library (or pyyaml), your python scripts (like train.py) can programmatically overwrite specific keys in the YAML file. Those changes become permanently persistent on your hard drive, allowing other processes or scripts to instantly see the updated paths.
            # ------------------------------
            # ## 📋 Phase 1: Create your config.yaml
            # Create a file named config.yaml in your root directory. Notice how much cleaner your system parameters look when stripped of raw Python syntax:

            # # File Storage Pathspaths:
            # raw_data_path: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/yellow_tripdata_2022-03.parquet"
            # processed_path: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/data/processed/processed.parquet"
            # features_path: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/features_store/selected_features.parquet"
            # zone_path: ""
            # path_prev_train_data: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/features_store/selected_features.parquet"
            # # Model Registry (Dynamically updated by train.py)model_registry:
            # best_model_path: ""
            # best_model_name: ""
            # best_model_ver: 1
            # production_model_index: 0
            # model_list_json: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/model_list.json"
            # # Logging Frameworklogging:
            # log_parquet: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/log/log.parquet"
            # metric_log: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/metric_log/metric_log.parquet"
            # performance_log: "C:/Users/nikhi/Downloads/Large-Scale-Taxi-Demand-Forecasting-System/performance_log/performance_log.parquet"
            # # Feature Engineering Constraintsfeatures:
            # lag_hours: [1, 2, 3, 6, 24, 48, 168]
            # rolling_windows: [1, 2, 3, 6, 24, 48, 168]
            # target_column: "demand"
            # feature_columns:
            #     - "hour_of_day"
            #     - "day_of_week"
            #     - "month"
            #     - "is_weekend"
            #     - "zone_id"
            #     # ... add the rest of your columns here
            # # Spark Configurationsspark:
            # app_name: "Taxi-Demand-Forecasting"
            # shuffle_partitions: "200"
            # driver_memory: "4g"

            # ------------------------------
            # ## 📋 Phase 2: Create a Persistent Reader & Writer
            # To make sure your comments and formatting inside config.yaml don't get erased when Python updates a value, use ruamel.yaml. Install it via your terminal: pip install ruamel.yaml.
            # Rewrite your config/settings.py file to serve as the unified bridge that loads and saves this file:

            # import osfrom pathlib import Pathfrom ruamel.yaml import YAML
            # # Find the absolute path to config.yaml relative to this settings fileCONFIG_FILE_PATH = Path(__file__).parent.parent / "config.yaml"
            # yaml = YAML()
            # yaml.preserve_quotes = True  # Keeps your string formatting clean
            # def load_config() -> dict:
            #     """Reads the YAML file and returns it as a dictionary."""
            #     if not CONFIG_FILE_PATH.exists():
            #         raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE_PATH}")
            #     with open(CONFIG_FILE_PATH, 'r') as f:
            #         return yaml.load(f)
            # def update_config_value(category: str, key: str, value):
            #     """Updates a single value in the YAML file and permanently saves it."""
            #     config_data = load_config()
                
            #     # Update the value in memory
            #     config_data[category][key] = value
                
            #     # Save it back to the file system persistently
            #     with open(CONFIG_FILE_PATH, 'w') as f:
            #         yaml.dump(config_data, f)

            # ------------------------------
            # ## 📋 Phase 3: Using It in the Rest of Your Project
            # Now, whenever you interact with configuration values across your pipeline, you rely on file persistence instead of volatile Python script memory.
            # ## 1. How train.py updates the settings file persistently:
            # When your training pipeline finishes and selects a winner, it overwrites the YAML configurations directly on the hard drive.

            # from config.settings import update_config_value
            # def complete_training_run():
            #     # Model evaluation chooses a winner...
            #     chosen_path = "C:/Users/nikhi/.../models/artifacts/quantile_high_model"
            #     chosen_name = "LightGBM_Quantile_High"
            #     new_version = 2
                
            #     # Update values permanently in config.yaml
            #     update_config_value("model_registry", "best_model_path", chosen_path)
            #     update_config_value("model_registry", "best_model_name", chosen_name)
            #     update_config_value("model_registry", "best_model_ver", new_version)
                
            #     print("YAML file has been permanently updated for all downstream components!")

            # ## 2. How your Inference/Prediction scripts read the live settings:
            # Because the configuration is read directly from the modified file at runtime, it always displays the exact paths written by train.py.

            # from config.settings import load_config
            # def run_production_inference():
            #     # Load the fresh configuration dictionary
            #     config = load_config()
                
            #     # Extract out what you need
            #     model_path = config["model_registry"]["best_model_path"]
            #     target = config["features"]["target_column"]
            #     spark_mem = config["spark"]["driver_memory"]
                
            #     print(f"Spinning up Spark with {spark_mem} to run inference using {model_path}")
            #     # Load model and run...

            # ## 🌟 Why this approach works best:

            # 1. User-Friendly: Anyone downloading your repository can open config.yaml to change paths, adjustments to rolling windows, or adjust spark cluster allocations instantly.
            # 2. Crash-Proof: If your system reboots, the code doesn't lose track of its parameters—the state is completely safe inside a file structure.
            # 3. No Reference Quirks: You no longer need to worry about Python's immutable variable pointer traps because you are parsing fresh files directly via dictionaries.

            # ------------------------------
            # Would you like help writing a dynamic environment parser into this layout so that it automatically switches between Windows paths (C:/...) and Linux paths if someone else downloads it?

    


def model_registry():
    def metric_improved_logic():
        pass
    def rollback_logic():
        pass
    def update_logic():
        pass