import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    raw_data_path:str=os.path.join('Artifact','raw.csv')
    train_data_path:str=os.path.join('Artifact','train.csv')
    test_data_path:str=os.path.join('Artifact','test.csv')

class DataIngestion:
    def __init__(self):
        self.Ingestion_Config=DataIngestionConfig()

    def Initiate_data_Ingestion():
        logging.info('Entered Data Ingestion Method')
        try:
            df=pd.read_csv(R'C:\Users\satish.hiremath\Desktop\E2EDatasciecne\notebook\data\stud.csv')
            logging.INFO('data has been read as saved as df data frame')
            os.makedirs(os.path.dirname(self.Ingestion_Config.train_data_path),exist_ok=True)
            df.to_csv(self.Ingestion_Config.raw_data_path,index=False,header=True)

            logging.info('Starting Train and Test Split')
            
        except Exception as e:
            pass