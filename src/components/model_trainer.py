import os
import sys
import torch
from dataclasses import dataclass
from torch.optim import Adam
from torch import nn
from tqdm import tqdm
from transformers import BertModel

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('models','model.pth')

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
 
    def initiate_model_trainer(self,train_dataloader,val_dataloader,test_dataloader):
        try:

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            criterion = nn.CrossEntropyLoss()
            model = BertClassifier()
            epochs = 5
            learning_rate = 1e-6
            optimizer = Adam(model.parameters(), lr= learning_rate)

            for epoch_num in range(epochs):

                total_acc_train = 0
                total_loss_train = 0

                for train_input, train_label in tqdm(train_dataloader):

                    train_label = train_label.to(device)
                    mask = train_input['attention_mask'].to(device)
                    input_id = train_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)
                    
                    batch_loss = criterion(output, train_label.long())
                    total_loss_train += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc

                    model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                
                total_acc_val = 0
                total_loss_val = 0

                with torch.no_grad():

                    for val_input, val_label in val_dataloader:

                        val_label = val_label.to(device)
                        mask = val_input['attention_mask'].to(device)
                        input_id = val_input['input_ids'].squeeze(1).to(device)

                        output = model(input_id, mask)

                        batch_loss = criterion(output, val_label.long())
                        total_loss_val += batch_loss.item()
                        
                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc
                
            train_accuracy = round(total_acc_train/len(train_dataloader),2)
            val_accuracy = round(total_loss_val/len(val_dataloader),2)
            test_accuracy = self.evaluate(model, test_dataloader)

            return train_accuracy, val_accuracy,test_accuracy

        except Exception as e:
            raise CustomException(e, sys)
        
    def evaluate(self,model, test_dataloader):

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if use_cuda:
            model = model.cuda()

        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        return round(total_acc_test/len(test_dataloader), 2)
            