import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import dataingestion
from src.components.data_transformation import datatransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    obj=dataingestion()
    train_data_path, test_data_path=obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)

    obj2=datatransformation()

    train_arr, test_arr, _= obj2.initiate_data_transformation(train_data_path, test_data_path)\
    
    model_trainer=ModelTrainer()
    model_trainer.initate_model_training(train_arr,test_arr)