import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception_handling import CustomException
from src.logging import logging
import os 
import sys

@dataclass
class DataTransformerConfig:
    preprocesser_obj_file_path=os.path.join("artifact","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformer_config=DataTransformerConfig()

    def get_data_transformer_object(self):
        try:
            num_fea=['writing_score','reading_score']
            cat_fea=[
                'gender','race_ethnicity','parental_level_of_education',
                'lunch','test_preparation_score'
            ]    
        
            num_pipelines=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )

            logging.info("Numerical Features standard scaling completed")

            logging.info("categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipelines",num_pipelines,num_fea),
                    ("cat_pipeline",cat_pipeline,cat_fea)
                ]
            )

            return preprocessor
        except Exception as ex:
            raise CustomException(ex,sys)

    def initiate_data_transform(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading the data scuessfully")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj=self.get_data_transformer_object()

            traget_column_name="math_score"
            numeric_features=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[traget_column_name],axis=1)
            traget_feature_train_df=train_df[traget_column_name]

            input_feature_test_df=test_df.drop(columns=[traget_column_name],axis=1)
            traget_feature_test_df=test_df[traget_column_name]

            logging.info("Applying preprocessor object on testing and training dataframe")
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(traget_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(traget_feature_test_df)
            ]
             
            save_object(
                file_path=self.data_transformer_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            ) 

            logging.info("saved preprocessing object")

            return(
                train_arr,
                test_arr,
                self.data_transformer_config.preprocesser_obj_file_path
            )
            
            
        except Exception as e:
            raise CustomException(e,sys)         
            
                        

