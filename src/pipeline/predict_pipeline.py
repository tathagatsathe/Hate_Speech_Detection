import os
import sys
import torch
from transformers import BertTokenizer

from src.exception import CustomException
from src.components.model_trainer import BertClassifier

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class PredictPipeline:
    def __init__(self):
        pass

    def classify(self,input_text):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_path = os.path.join('models','model.pth')

            model = BertClassifier()
            model.load_state_dict(torch.load(model_path))

            input_tokens = tokenizer(input_text, 
                          padding='max_length',
                          max_length = 512, 
                          truncation=True,
                          return_tensors="pt")
            mask = input_tokens['attention_mask'].to(device)
            input_id = input_tokens['input_ids'].squeeze(1).to(device)

            return model(input_id, mask)

        except Exception as e:
            raise CustomException(e, sys)