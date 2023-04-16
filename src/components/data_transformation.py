import os
import sys
from dataclasses import dataclass
import torch

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from src.exception import CustomException
from src.logger import logging


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = df['label'].to_list()
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class DataTransformation:
    def __init__(self):
        pass
        # self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self,train_path,val_path, test_path):
        try:
            logging.info('Data Transformation Started')

            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            logging.info('loaded data from path')
            
            cols = ['text','label']
            train_df.rename(columns={'hatespeech':'label'},inplace=True)
            val_df.rename(columns={'hatespeech':'label'},inplace=True)
            test_df.rename(columns={'hatespeech':'label'},inplace=True)

            logging.info('Renamed Column hatespeech to label')
            train, val, test = Dataset(train_df[cols]), Dataset(val_df[cols]), Dataset(test_df[cols])

            logging.info('Converted data to torch.utils.data.Dataset format')

            train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
            test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

            logging.info('Converted data to torch.utils.data.DataLoader format') 
            logging.info('Data Transformation Completed')

            return (
                train_dataloader,
                val_dataloader,
                test_dataloader
            )

        except Exception as e:
            raise CustomException(e, sys)