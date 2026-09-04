#ability to add new model
#rollback to previous model if neccesary
#should be able to see when the model was promoted,trained,its hyper parameters,metrics and other meta data
#should not have race condition because one process chose to promote and another chose to rollback
#if there is no model to rollback and roll back is called it should return the lack of model
#datastructure to show models and wtheir respective metadata + what wud the complexity cost be
#when promote is called to make  a new model the production one, it should pass the benchmark of being better than the current model or else should not be promoted to production


"""Template/basic layout of the registry with template for working and basic logics and scaffolding......to build on top off,upgrade and fix"""
"""production_index determines which model is used in production"""
from config.settings import MODEL_LIST,PROD_MODEL_PATH,PROD_MODEL_NAME,PRODUCTION_MODEL_INDEX,BEST_MODEL_VER
import logging
import json
from collections import Counter

#add class and functions once logic build 
with open(MODEL_LIST,"r+") as f:
    model_list=json.load(f)

model_versions=[k for k,_ in dict.items()]

def get_metrics(model_name):
    try:
        with open(f"Large-Scale-Taxi-Demand-Forecasting-System/models/artifacts/{model_name}_metrics.json","w")as f:
                            model_metrics=json.load(f)
                            return model_metrics
    except FileNotFoundError:
          return("model name is wrong or metric file has been deleted check name and location")


def metric_improved(current,new):
    "compare metrics of current model to new model(could also be older model when doing rollback).If the new model has better metrics return true or else false"
    #note:should update to the primary metric and guardrail setup,dev-docs ->todo.py->metric_logic() has code for this
    rmse="new" if new["rmse"] < current["rmse"] else "current"
    mae= "new" if new["mae"] < current["mae"] else "current"
    smape="new" if new["smape"] < current["smape"] else "current"
    r_square="new" if new["r2"] > current["r2"] else "current"  # Higher R2 is better

    

    votes=[rmse,mae,smape,r_square]
    count=Counter(votes)
    
    if count["new"]>count["current"]:
          return True

    return False


    
def rollback(failure:bool=False):
    if failure:
          try:
              PRODUCTION_MODEL_INDEX -=1
              PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
              PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
              return(f"model rolled back to {model_list[PRODUCTION_MODEL_INDEX["model_name"]]} ,version/index:{PRODUCTION_MODEL_INDEX}")
          except Exception as e:
                return f"error:{e}"
    else:
          
        if f"V_{PRODUCTION_MODEL_INDEX}" == model_versions[0]:
            logging.error(msg="only current model present,no model to rollback to ")
        else:
            current=model_list[model_versions[PRODUCTION_MODEL_INDEX]]
            prev=model_list[model_versions[PRODUCTION_MODEL_INDEX-1]]
            current_name=current["model_name"]
            prev_name=prev["model_name"]
            current_metrics=get_metrics(model_name=current_name)
            prev_metrics=get_metrics(model_name=prev_name)
            improvement= metric_improved(current=current_metrics,new=prev_metrics)
            if improvement:
            
                try:
                    PRODUCTION_MODEL_INDEX -=1
                    PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
                    PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
                    return(f"model rolled back to {model_list[PRODUCTION_MODEL_INDEX["model_name"]]} ,version/index:{PRODUCTION_MODEL_INDEX}")
                except Exception as e:
                    return f"error:{e}"
                  
            

def update(manual:bool=False,):
      #only update manually without checking if the features used to train the model has been changed,this should be done in trigger retraina nd alert manager where it does a feature check with current modela and the feature in the config file,if changed,trigger retrain
      if manual:
           try:
               PRODUCTION_MODEL_INDEX = BEST_MODEL_VER
               PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
               PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
               return(f"model rolled back to {model_list[PRODUCTION_MODEL_INDEX["model_name"]]} ,version/index:{PRODUCTION_MODEL_INDEX}")
           except Exception as e:
                 return f"error:{e}"
      else:
      
         if f"V_{PRODUCTION_MODEL_INDEX}" == model_versions[0]:
             logging.error(msg="only current model present,no model to rollback to ")
         else:
             current=model_list[model_versions[PRODUCTION_MODEL_INDEX]]
             prev=model_list[model_versions[PRODUCTION_MODEL_INDEX-1]]
             current_name=current["model_name"]
             prev_name=prev["model_name"]
             current_metrics=get_metrics(model_name=current_name)
             prev_metrics=get_metrics(model_name=prev_name)
             improvement= metric_improved(current=current_metrics,new=prev_metrics)
             if improvement:
      
                 try:
                     PRODUCTION_MODEL_INDEX = BEST_MODEL_VER
                     PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
                     PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
                     return(f"model rolled back to {model_list[PRODUCTION_MODEL_INDEX["model_name"]]} ,version/index:{PRODUCTION_MODEL_INDEX}")
                 except Exception as e:
                     return f"error:{e}"
      


