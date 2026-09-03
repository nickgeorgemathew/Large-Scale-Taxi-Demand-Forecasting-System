2-9-26
    fixed json logic in train.save_best_model,adding template and basic.
    layout of model registry.
    Created new config variable
    production_model_index that gives the index of the current production.
    model to be used in model registry for manipulation of model_list and other registry functioning of moving/changing the model.
    rollback base logic and function to check metric improvement.
3-9-26
    added dev-docs folder to store all thought proccess,logic setups,todo etc.
    created new helper function model_registry.get_metrics to help get metrics from the json file created during training and storing,logic->(get the name of the model,then use the f string to get the location of the metrics ,load using json and then return the list ).
    added try and except to catch if there is no model to rollback to when rollback is called(just a base for now,shud add and imrpove logic and working).
    created a simple voting system based logic for model_registry.metric_improved.
    created logic for rollback where it checks for if there is a model to rollback to first,if present it will compare the metrics of the current model to the prev one....if improvements are detected it will roll back...incase the model had some other failure and prev model is required a flag is passed to the rollback function which manually rollback to the previous model without comparing metrics.
    Built a simple logic for checking for feature change in monitoring.drift_detector.asses_feature_change,shud setup rest of the project to align with the downstream effects of feature change,especially in modelops and registry