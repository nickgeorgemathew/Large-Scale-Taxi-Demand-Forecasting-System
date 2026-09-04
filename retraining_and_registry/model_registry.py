#ability to add new model
#rollback to previous model if neccesary
#should be able to see when the model was promoted,trained,its hyper parameters,metrics and other meta data
#should not have race condition because one process chose to promote and another chose to rollback
#if there is no model to rollback and roll back is called it should return the lack of model
#datastructure to show models and wtheir respective metadata + what wud the complexity cost be
#when promote is called to make  a new model the production one, it should pass the benchmark of being better than the current model or else should not be promoted to production


"""Template/basic layout of the registry with template for working and basic logics and scaffolding......to build on top off,upgrade and fix"""
"""production_index determines which model is used in production"""
from config.settings import MODEL_LIST,PROD_MODEL_PATH,PROD_MODEL_NAME,PRODUCTION_MODEL_INDEX,BEST_MODEL_VER,REGISTRY_PATH,REGISTRY_INDEX
import logging
import json
from datetime import datetime
from collections import Counter
import os

#add class and functions once logic build 
with open(MODEL_LIST,"r+") as f:
    model_list=json.load(f)

model_versions=[k for k,_ in dict.items()]

def get_metrics(model_name):
    """get metrics for a model,pass model name as args"""
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

def get_all_models():
     "return the list of all models saved so far with all their metadata"
     return model_list
     

def ensure_registry_exist():
    """ensures if the Registry json file exist, if it does not, create  new one """
    registry={}
    if not os.path.exists(REGISTRY_PATH):
        REGISTRY_PATH.mkdir(parents=True,exist_ok=True)
        registry[REGISTRY_INDEX]={"Version":PRODUCTION_MODEL_INDEX,"prod":True,"model_name":PROD_MODEL_NAME,"model_path":PROD_MODEL_PATH,"action":"update","date":datetime.today().strftime("%d/%m/%Y, %H:%M:%S"),"prev_model_index":None,"prev_model_name":None,"prev_model_path":None}
        with open(REGISTRY_PATH,"w") as f:
             json.dump(registry,f)
        print("registry created")
        REGISTRY_INDEX+=1
        return(registry)
    else:
         return("registry exists")

#how do i ensure the varaibles of prod in config.settings and the entries in the registry.json is concurrent and right??
def get_prod_model():
     "Return production model name,path and index"
     return ({"Model_name":PROD_MODEL_NAME,"Model_path":PROD_MODEL_PATH,"Model_index":PRODUCTION_MODEL_INDEX})
     
 
#how do i prevent race condition for Registry Index
def update_registry(data):
    "updates the registry"
    try:
        with open(REGISTRY_PATH,"r+") as f:
            registry=json.load(f)
        registry[REGISTRY_INDEX]=data
        with open(REGISTRY_PATH,"w") as f:
            json.dump(registry)
        REGISTRY_INDEX+=1
        return("registry updated")
    except Exception as e:
         return(f"error:{e}")
    

def get_prev_model(registry:bool=True,model_data:bool=False):
        """get all previous models used in prod and saved,pass registry true if only registry data is required,model_data if model data is required"""
        if registry and model_data :
            with open(MODEL_LIST,"r") as f:
                model_list=json.load(f)
            with open(REGISTRY_PATH,"r") as f:
                registry_data=json.load(f)
            return({"model_list":model_list,"registry":registry_data})
        elif registry:
             with open(REGISTRY_PATH,"r") as f:
                 registry_data=json.load(f)
             return({"registry":registry_data})
        elif model_data:
             with open(MODEL_LIST,"r") as f:
                 model_list=json.load(f)
             return({"model_list":model_list})
             

     
     
#the registry json file shud contain the history of the prod models with their names,path adn whether it was a roll back or a promotion with date.each time rollback or update runs it shud update the registry json file with these details

#if the new model index is further then the roll back say rollback index is 3 prev model was 4 and new model is 5 how do i ensure that the index doesnot update to 4 when new model is updated but instead falls into 5

#should build a feature to prevent race condition and deadlock:only one function from this shud run at a time
def rollback(failure:bool=False):
    """rollback the prod model to the previous model,pass failure as true if model needs to be updated to prod without metric checks"""
    if failure:
          try:
            prev_model=PRODUCTION_MODEL_INDEX
            PRODUCTION_MODEL_INDEX -=1
            PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
            PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
            
            data={"Version":PRODUCTION_MODEL_INDEX,"prod":True,"model_name":PROD_MODEL_NAME,"model_path":PROD_MODEL_PATH,"action":"manual rollback","date":datetime.today().strftime("%d/%m/%Y, %H:%M:%S"),"prev_model_index":prev_model,"prev_model_name":model_list[f"V_{prev_model}"["model_name"]],"prev_model_path":model_list[f"V_{prev_model}"["model_path"]]}
            update_registry(data)
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
                    prev_model=PRODUCTION_MODEL_INDEX
                    PRODUCTION_MODEL_INDEX -=1
                    PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
                    PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
                    data={"Version":PRODUCTION_MODEL_INDEX,"prod":True,"model_name":PROD_MODEL_NAME,"model_path":PROD_MODEL_PATH,"action":"automated rollback","date":datetime.today().strftime("%d/%m/%Y, %H:%M:%S"),"prev_model_index":prev_model,"prev_model_name":model_list[f"V_{prev_model}"["model_name"]],"prev_model_path":model_list[f"V_{prev_model}"["model_path"]]}
                    update_registry(data)
                    return(f"model rolled back to {model_list[PRODUCTION_MODEL_INDEX["model_name"]]} ,version/index:{PRODUCTION_MODEL_INDEX}")
                except Exception as e:
                    return f"error:{e}"
                  
            

def update(manual:bool=False):
      """update the prod model to the latest model,pass manual as true if model needs to be updated to prod without metric checks"""
      #only update manually without checking if the features used to train the model has been changed,this should be done in trigger retraina nd alert manager where it does a feature check with current modela and the feature in the config file,if changed,trigger retrain
      if manual:
           try:
               prev_model=PRODUCTION_MODEL_INDEX
               PRODUCTION_MODEL_INDEX = BEST_MODEL_VER
               PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
               PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
               data={"Version":PRODUCTION_MODEL_INDEX,"prod":True,"model_name":PROD_MODEL_NAME,"model_path":PROD_MODEL_PATH,"action":"manual update","date":datetime.today().strftime("%d/%m/%Y, %H:%M:%S"),"prev_model_index":prev_model,"prev_model_name":model_list[f"V_{prev_model}"["model_name"]],"prev_model_path":model_list[f"V_{prev_model}"["model_path"]]}
               update_registry(data)
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
                     prev_model=PRODUCTION_MODEL_INDEX
                     PRODUCTION_MODEL_INDEX = BEST_MODEL_VER
                     PROD_MODEL_NAME=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_name"]]
                     PROD_MODEL_PATH=model_list[f"V_{PRODUCTION_MODEL_INDEX}"["model_path"]]
                     data={"Version":PRODUCTION_MODEL_INDEX,"prod":True,"model_name":PROD_MODEL_NAME,"model_path":PROD_MODEL_PATH,"action":"automated update","date":datetime.today().strftime("%d/%m/%Y, %H:%M:%S"),"prev_model_index":prev_model,"prev_model_name":model_list[f"V_{prev_model}"["model_name"]],"prev_model_path":model_list[f"V_{prev_model}"["model_path"]]}
                     update_registry(data)
                     return(f"model rolled back to {model_list[PRODUCTION_MODEL_INDEX["model_name"]]} ,version/index:{PRODUCTION_MODEL_INDEX}")
                 except Exception as e:
                     return f"error:{e}"
      


