#ability to add new model
#rollback to previous model if neccesary
#should be able to see when the model was promoted,trained,its hyper parameters,metrics and other meta data
#should not have race condition because one process chose to promote and another chose to rollback
#if there is no model to rollback and roll back is called it should return the lack of model
#datastructure to show models and wtheir respective metadata + what wud the complexity cost be
#when promote is called to make  a new model the production one, it should pass the benchmark of being better than the current model or else should not be promoted to production


