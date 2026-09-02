#ability to add new model
#rollback to previous model if neccesary
#should be able to see when the model was promoted,trained,its hyper parameters,metrics and other meta data
#should not have race condition because one process chose to promote and another chose to rollback
#if there is no model to rollback and roll back is called it should return the lack of model
#datastructure to show models and wtheir respective metadata + what wud the complexity cost be
#when promote is called to make  a new model the production one, it should pass the benchmark of being better than the current model or else should not be promoted to production


"""Template/basic layout of the registry with template for working and basic logics and scaffolding......to build on top off,upgrade and fix"""

from config.settings import MODEL_LIST,BEST_MODEL_PATH,BEST_MODEL_NAME,PRODUCTION_MODEL_INDEX
import logging
import json
with open(MODEL_LIST,"r+") as f:
    model_list=json.load(f)

model_versions=[k for k,_ in dict.items()]

def metric_improved(current,new):
    "compare metrics of current model to new model(could also be older model when doing rollback).If the new model has better metrics return true or else false"
def rollback():
    if PRODUCTION_MODEL_INDEX == model_versions[0]:
        logging.error(msg="only current model present,no model to rollback to ")
    else:
        metric_better=
        if :
        #write a function to compare metrics and return true or false flag
            
            PRODUCTION_MODEL_INDEX -=1
            BEST_MODEL_NAME=model_list[PRODUCTION_MODEL_INDEX["model_name"]]
            BEST_MODEL_PATH=model_list[PRODUCTION_MODEL_INDEX["model_path"]]
            




