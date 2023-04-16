import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import datasets

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('data','train.csv')
    val_data_path: str = os.path.join('data','val.csv')
    test_data_path: str = os.path.join('data','test.csv')
    raw_data_path: str = os.path.join('data','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method started')

        try:
            dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech', 'binary')
            df = dataset['train'].to_pandas()
            
            logging.info('loaded dataset')

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            train_set, val_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set, test_set = train_test_split(train_set,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            val_set.to_csv(self.ingestion_config.val_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data Ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.val_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj = DataIngestion()
    train_path, val_path, test_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_dataloader, val_dataloader, test_dataloader = data_transformation.initiate_data_transformation(train_path,val_path, test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_dataloader,val_dataloader,test_dataloader)
    