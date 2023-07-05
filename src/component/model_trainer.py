import pandas as pd
import numpy as np
import os 
import sys
from logger import logging
from exception import CustomException
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    tranined_model_file_path = os.path.join("artifact", "model.pkl")

class Modeltrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("splitting training and test input data")
            X_train, y_train, X_test, y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors regressor":KNeighborsRegressor(),
                "XGBregressor":XGBRegressor(),
                "Catboosting Classifier":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            ## to get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ##to get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            

            logging.info("best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.tranined_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_squared = r2_score(y_test, predicted)
            return r2_squared
        except Exception as e:
            raise CustomException(e, sys)