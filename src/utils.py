import os
import sys

import numpy as np
import pandas as pd
import pickle
import torch

from transformers import BertTokenizer
from src.exception import CustomException

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize(input_text):
    input_tokens = tokenizer(input_text, 
                          padding='max_length',
                          max_length = 512, 
                          truncation=True,
                          return_tensors="pt")
    mask = input_tokens['attention_mask'].to(device)
    input_id = input_tokens['input_ids'].squeeze(1).to(device)
    return mask, input_id

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)