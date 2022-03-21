import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from transformers import BertModel
import wandb


class HeadDataset(Dataset):
    def __init__(self, df):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.labels = df['label'].to_numpy()
        self.texts = [self.tokenizer(text, padding='max_length', max_length=32, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, hidden_dim=768, num_classes=3):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, mask):
        _, outputs = self.bert(input_ids=input_ids, attention_mask=mask, return_dict=False)
        outputs = self.dropout(outputs)
        outputs = self.linear(outputs)
        probs = self.softmax(outputs)

        return probs
    
    def save_pretrained(self, path):
        torch.save(self, path)


def train(model, train_dataset, val_dataset, learning_rate=1e-3, epochs=10, batch_size=4):
    wandb.init(project="kogito-relation-matcher", config={"learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size})

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        print("Using CUDA")

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            model.zero_grad()

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_ids, mask)
            
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
        
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_ids, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
                
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc
        
        train_loss = total_loss_train / len(train_data)
        train_acc = total_acc_train / len(train_data)
        val_loss = total_loss_val / len(val_data)
        val_acc = total_acc_val / len(val_data)

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {train_loss: .3f} \
            | Train Accuracy: {train_acc: .3f} \
            | Val Loss: {val_loss: .3f} \
            | Val Accuracy: {val_acc: .3f}')
        
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc})
        model.save_pretrained(f"./models/checkpoint_{epoch_num}.pth")


from relation_modeling_utils import load_data

train_df = load_data("data/atomic2020_data-feb2021/train.tsv")
dev_df = load_data("data/atomic2020_data-feb2021/dev.tsv")
train_data = HeadDataset(train_df)
val_data = HeadDataset(dev_df)
model = BertClassifier()
train(model=model, train_dataset=train_data, val_dataset=val_data, epochs=10, batch_size=4, learning_rate=1e-6)
model.save_pretrained("./models/final_model.pth")