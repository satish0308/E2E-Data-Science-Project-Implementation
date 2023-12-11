import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
import dill
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_obj(file_path,obj):
    try:
        dire_path=os.path.dirname(file_path)
        os.makedirs(dire_path,exist_ok=True)
        with open(file_path,"wb") as file_Obj:
            dill.dump(obj,file_Obj)

    except Exception as e:
        raise CustomException(sys,e)
    
def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            logging.info(f"the Current model select {para} for the model {model}")

            gs=GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            logging.info(f"The Best Working Model {model}")
            model.fit(x_train,y_train)
            y_train_pred=model.predict(x_train)
            y_test_pred=model.predict(x_test)

            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

        
            report[list(models.keys())[i]] = test_model_score
        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def  load_object(file_path):
    try:
        with open(file_path,'rb') as file_Obj:
            return dill.load(file_Obj)
    except Exception as e:
        raise CustomException(e,sys)