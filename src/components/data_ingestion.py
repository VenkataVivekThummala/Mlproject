import os
import sys
from src.exception_handling import CustomException
from src.logging import logging
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.dataclasses import dataclass
from src.components.data_transform import DataTransformerConfig,DataTransformation

#@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifact","train.csv")
    test_data_path=os.path.join("artifact","test.csv")
    raw_data_path=os.path.join("artifact","data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("enter the data ingestion method or component")
        try:
            df=pd.read_csv("/home/venkata/Downloads/firstproj/Mlproject/notebook/data/stud.csv")
            logging.info("Read the Dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,


            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data= obj.initiate_data_ingestion()

    Datatransfer=DataTransformation()
    Datatransfer.initiate_data_transform(train_data,test_data)
