import os
import sys
from exception import CustomException
from logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from component.data_transformation import DataTransformation
from component.data_transformation import DataTransformationConfig
from component.model_trainer import ModelTrainerConfig
from component.model_trainer import Modeltrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact', "train.csv")
    test_data_path: str=os.path.join('artifact', "test.csv")
    raw_data_path: str=os.path.join('artifact', "data.csv")
    # os.path.join() berfungsi untuk menambahkan satu path baru berupa nama folder "artifact" kepada path direktori
    # utama yang di assign padanya, kemudian nama filenya di assing dengan format .csv

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("enter the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            #os.makedirs() berfungsi untuk membuat sebuah direktori baru, argumen pertama adalah path dari direktorinya
            #parameter exist_ok , berfungsi untuk memeriksa apakah direktori dengan nama yang sama sudah pernah  dibuat atau tidak
            #jika argumennya adalah True, maka ketika ada direktori dengan nama yang sama, maka akan dibiarkan.
            #os.path.dirname() memiliki fungsi untuk mengembalikan path direktori yang berbentuk string.
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state =  42)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("ingestion of the  data was completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = Modeltrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
