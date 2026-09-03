
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
    #improved logic as shown here,shud update to primary_guardrail logic soon
    def metric_improved_logic():
        pass
        #         Your logic for comparing multiple metrics using a "voting system" is a great way to handle model evaluation! However, the function as written will always return True due to a subtle Python type bug at the very end, and it is missing a crucial import.
        # Here is a breakdown of why it breaks and how to write it correctly.
        # ## 🔍 The Bugs in Your Code

        #    1. The Type Comparison Error (The Main Bug):
        #    Counter.most_common(n=1) does not return a string like "current". It returns a list containing a tuple of the item and its count, looking like this: [('current', 3)].
        #    Because you are checking if a list equals a string (if count == "current"), that condition will always be False, forcing your code to execute the else: block and falsely return True every single time.
        #    2. Missing Import:
        #    You are using Counter but haven't imported it from the collections module.
        #    3. The Tie-Breaker Risk:
        #    If two metrics favor current and two favor new, most_common(1) will just pick whichever one it processed first. In MLOps, if a new model doesn't strictly beat the baseline, you typically want to stick with the current model (return False).

        # ------------------------------
        # ## 🛠️ The Clean and Correct Fix
        # Here is the corrected code. It extracts the winning string key correctly from the Counter object and defaults to safely returning False in the event of a tie:

        # from collections import Counter
        # def metric_improved(current: dict, new: dict) -> bool:
        #     """Compare metrics of current model to new model (or an older model for rollbacks).
        #     Returns True if the new model is overall better based on a majority vote of metrics,
        #     otherwise returns False.
        #     """
        #     # 1. Direct dictionary lookups (no need to instantiate separate dicts for min/max)
        #     rmse_winner = "new" if new["rmse"] < current["rmse"] else "current"
        #     mae_winner = "new" if new["mae"] < current["mae"] else "current"
        #     smape_winner = "new" if new["smape"] < current["smape"] else "current"
        #     r2_winner = "new" if new["r2"] > current["r2"] else "current"  # Higher R2 is better

        #     # 2. Collect votes
        #     votes = [rmse_winner, mae_winner, smape_winner, r2_winner]
        #     vote_counts = Counter(votes)
            
        #     # 3. Safely extract the clear winner
        #     # .most_common(1)[0][0] extracts the string key ('new' or 'current') from [('new', 3)]
        #     overall_winner = vote_counts.most_common(1)[0][0]
            
        #     # 4. If there is a 2-2 tie, most_common picks whichever appeared first. 
        #     # To be safe, let's explicitly handle ties to favor the stable 'current' model.
        #     if vote_counts["new"] > vote_counts["current"]:
        #         return True
                
        #     return False

        # ## 💡 An Alternative MLOps Best Practice: Weighted Thresholds
        # While a voting system is clean, in production taxi demand forecasting, a 0.0001 improvement in $R^2$ shouldn't justify deploying a whole new model if the RMSE degraded significantly.
        # Usually, production teams pick one primary metric (like MAE or RMSE) that must improve by a specific margin, while treating the others as "guardrails." If you ever want to move to that approach, it looks like this:

        # def metric_improved_with_gate(current: dict, new: dict, threshold_pct: float = 0.01) -> bool:
        #     # Require the primary metric (e.g., MAE) to improve by at least 1%
        #     mae_improvement = (current["mae"] - new["mae"]) / current["mae"]
            
        #     # Guardrail: Make sure R2 didn't tank drastically
        #     r2_dropped_severely = new["r2"] < (current["r2"] * 0.95)
            
        #     if mae_improvement >= threshold_pct and not r2_dropped_severely:
        #         return True
        #     return False

        # ------------------------------
        # To make sure this fits into your CI/CD retraining loop seamlessly, tell me:

        # * Would you like help integrating this into your YAML workflow so it automatically reads the current baseline metrics from your model_metrics_test.json file?
        # * Do you want to add validation checks to ensure neither model passes NaN or negative values into this function?
        def primary_and_guardrail_metric_logic():
            pass
            #             Moving to a Primary Metric vs. Guardrail Metrics approach is a fantastic choice for your Taxi Demand Forecasting system. In a production environment, this strategy prevents your pipeline from deploying models that look good on paper but fail under critical conditions.
            # ------------------------------
            # ## 🗺️ Step 1: How to Choose Your Metrics
            # For Taxi Demand Forecasting (regression mapping timestamps and zones to a number of trips), your available metrics solve completely different real-world problems.

            # | Metric Type | Metric Name | What It actually Means for a Taxi Business | Strategic Assignment |
            # |---|---|---|---|
            # | Magnitude | MAE (Mean Absolute Error) | Average absolute trip error. Treating an error of 10 trips off the same as an error of 1 trip off. Reflects the regular day-to-day experience across the city. | 🎯 Primary Metric (or Guardrail) |
            # | Variance / Outliers | RMSE (Root Mean Squared Error) | Penalizes large blunders heavily. If your model predicts 500 cars for a zone that only needs 50, RMSE will skyrocket. | 🎯 Primary Metric (or Guardrail) |
            # | Percentage Error | SMAPE (Symmetric Mean Absolute Percentage Error) | Error relative to volume. Missing by 5 cars in a quiet zone (where demand is 2) is a massive disaster compared to missing by 5 cars in Times Square (where demand is 500). | 🛡️ Guardrail Metric |
            # | Explanatory Power | $R^2$ (R-Squared) | Trend tracking. Does the model capture the rush hour waves and holiday drops, or is it just predicting the overall average flatline? | 🛡️ Guardrail Metric |

            # ## Which should be the Primary Metric?

            # * Choose MAE as your Primary if your primary business goal is scaling supply evenly across the entire city and ensuring drivers get realistic baseline estimates.
            # * Choose RMSE as your Primary if your primary business goal is avoiding devastating scheduling mistakes (e.g., leaving an airport completely abandoned of taxis or flooding a quiet residential zone with cars).

            # ------------------------------
            # ## 🛡️ Step 2: How to Configure Guardrails
            # Guardrail metrics exist to enforce business safety policies. They do not need to show an improvement, but they must not degrade past a specific tolerance window (usually 2% to 5%).

            #    1. SMAPE Guardrail: Protects low-volume zones. If your total city-wide MAE improves, but your SMAPE gets worse, it means your model is getting better at high-demand regions (Manhattan) while completely ignoring low-demand regions (Staten Island).
            #    2. $R^2$ Guardrail: Prevents "flatlining." A model that predicts a constant flat average might score a decent MAE, but its $R^2$ will tank because it has stopped capturing time-series variations (rush hours vs. late nights).

            # ------------------------------
            # ## 💻 Step 3: Implementation Code
            # Here is how you write this clean gating mechanism using your project’s parameters. You can add the thresholds to your config.yaml so they are fully configurable by anyone downloading your code.
            # ## Add this to your config.yaml:

            # model_evaluation:
            #   primary_metric: "mae"
            #   required_improvement_pct: 0.01  # New model must be at least 1% better
            #   guardrails:
            #     rmse_max_degradation_pct: 0.02  # RMSE cannot get worse by more than 2%
            #     smape_max_degradation_pct: 0.03 # SMAPE cannot get worse by more than 3%
            #     r2_min_value: 0.75              # R2 must never drop below a hard floor of 0.75

            # ## Add this to your Python pipeline:

            # def metric_improved_with_gate(current: dict, new: dict, eval_config: dict) -> bool:
            #     """Evaluates a new model using a strict primary metric and safety guardrails.
            #     Returns True if the model passes all checks and is safe to deploy.
            #     """
            #     # 1. Evaluate Primary Metric (Assuming MAE - lower is better)
            #     prim = eval_config["primary_metric"]
            #     req_improvement = eval_config["required_improvement_pct"]
                
            #     # Calculate how much better the new model is
            #     primary_improvement = (current[prim] - new[prim]) / current[prim]
                
            #     if primary_improvement < req_improvement:
            #         print(f"❌ Rejected: New model {prim} improved by {primary_improvement:.4f}, less than required {req_improvement}")
            #         return False

            #     # 2. Guardrail Check: RMSE Degradation (lower is better)
            #     rmse_degradation = (new["rmse"] - current["rmse"]) / current["rmse"]
            #     max_rmse_deg = eval_config["guardrails"]["rmse_max_degradation_pct"]
            #     if rmse_degradation > max_rmse_deg:
            #         print(f"❌ Rejected: RMSE degraded significantly by {rmse_degradation:.4f} (Max allowed: {max_rmse_deg})")
            #         return False

            #     # 3. Guardrail Check: SMAPE Degradation (lower is better)
            #     smape_degradation = (new["smape"] - current["smape"]) / current["smape"]
            #     max_smape_deg = eval_config["guardrails"]["smape_max_degradation_pct"]
            #     if smape_degradation > max_smape_deg:
            #         print(f"❌ Rejected: SMAPE degraded by {smape_degradation:.4f} (Max allowed: {max_smape_deg})")
            #         return False

            #     # 4. Guardrail Check: Hard R2 Floor (higher is better)
            #     min_r2_floor = eval_config["guardrails"]["r2_min_value"]
            #     if new["r2"] < min_r2_floor:
            #         print(f"❌ Rejected: New model R2 dropped to {new['r2']:.4f}, below the safe limit of {min_r2_floor}")
            #         return False

            #     # If it passes the primary threshold and survives all guardrails
            #     print("🚀 Approved! New model is safe and outperforms the current production baseline.")
            #     return True

            # ------------------------------
            # To tweak this safely for your data, could you tell me:

            # * What are the average baseline scores (approximate values for your current MAE and $R^2$) your model is currently hitting on the validation set?
            # * Do you have critical business hours (like rainy Friday nights) where you might want a specialized metric check to make sure error spikes don't cluster on high-revenue time frames?






    def rollback_logic():
        #
        pass
    def update_logic():
        pass