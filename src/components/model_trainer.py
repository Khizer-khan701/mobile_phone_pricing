import os 
import sys
from dataclasses import dataclass  
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass 
class ModelTrainerConfig():
    model_trainer_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and testing input data")
            X_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest Classifier": RandomForestClassifier(),
                "SVC": SVC(),
                "Decision Tree Classifier": DecisionTreeClassifier(),
                "Ada Boost Classifier": AdaBoostClassifier(),
                "Gradient Boosting Classifier": GradientBoostingClassifier(),
                "Cat Boost Classifier": CatBoostClassifier(verbose=0),
                "XGB Classifier": XGBClassifier(verbosity=0)
            }
            params = {
                        "Decision Tree Classifier": {
                            "criterion": ["gini", "entropy", "log_loss"]
                        },
                        "Random Forest Classifier": {
                            "n_estimators": [8, 16, 32, 64, 128, 256]
                        },
                        "Gradient Boosting Classifier": {
                            "learning_rate": [0.1, 0.01, 0.05, 0.001],
                            "subsample": [0.6, 0.7, 0.75, 0.5, 0.85, 0.9],
                            "n_estimators": [8, 16, 32, 64, 128, 256]
                        },
                        "Logistic Regression": [
                            {"penalty": ["l1"], "solver": ["saga"], "max_iter": [1000]},
                            {"penalty": ["l2"], "solver": ["saga"], "max_iter": [1000]},
                            {"penalty": ["elasticnet"], "solver": ["saga"], "l1_ratio": [0.0, 0.5, 1.0], "max_iter": [1000]}
                        ],
                        "SVC": {
                            "kernel": ["linear", "poly", "rbf", "sigmoid"],
                            "C": [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1, 2, 3, 4]
                        },
                        "Ada Boost Classifier": {
                            "n_estimators": [8, 16, 32, 64, 128, 256],
                            "learning_rate": [0.1, 0.2, 0.3, 0.01, 0.05, 0.9, 1]
                        },
                        "Cat Boost Classifier": {
                            "depth": [6, 8, 10],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "iterations": [30, 50, 100]
                        },
                        "XGB Classifier": {
                            "learning_rate": [0.1, 0.01, 0.05, 0.001],
                            "n_estimators": [8, 16, 32, 64, 128, 256]
                        }
                    }


            model_report:dict=evaluate_model(X_train,y_train,x_test,y_test,models,params)

            # to get the best model score from dic
            best_model_score=max(sorted(model_report.values()))

            # best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best Model Found on Training and testing dataset")
            
            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            accuracy=accuracy_score(y_test,predicted)
            Confusion_matrix=confusion_matrix(y_test,predicted)
            return(
                accuracy,
                Confusion_matrix
            )

        except Exception as e:
            raise CustomException(e,sys)
