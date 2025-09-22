import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# intialize the data ingestion configuration
@dataclass
class dataingestionconfig:
    train_data_path=os.path.join('artifacts', 'train.csv')
    test_data_path=os.path.join('artifacts', 'test.csv')
    raw_data_path=os.path.join('artifacts', 'raw.csv')

# create a data ingestion class
class dataingestion:
    def __init__(self):
        self.ingestion_config=dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebook/data', 'gemstone.csv'))
            logging.info('Dataset read as pandas DataFrame.')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True) # make artifacts dir if already not existing.

            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info('Train Test Split')
            train_set, test_set=train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed.')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(f"Error occurred in data ingestion: {e}")