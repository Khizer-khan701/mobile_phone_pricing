import os 
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd 
import numpy as np
from src.utils import save_object


@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            preprocessor = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            logging.info("Obtaining Preprocessor object")
            preprocessor_obj=self.get_data_transformer_obj()

            target_column="price_range"

            input_features_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_features_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying preprocessing obj on training and testing dataframe")

            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df)

            train_arr=np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)