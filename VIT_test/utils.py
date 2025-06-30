import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import MNISTTrainDataset,MNISTValDataset

def get_loaders(train_dir,  batch_size):
    train = pd.read_csv(train_dir)
    train, val = train_test_split(train,  test_size=0.1, random_state=42)

    train_dataset = MNISTTrainDataset(train.iloc[:,1:].values, train.iloc[:,0].values, train.index.values)
    val_dataset = MNISTValDataset(val.iloc[:,1:].values, val.iloc[:,0].values, val.index.values)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader

