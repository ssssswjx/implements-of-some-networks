import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MNISTTrainDataset(Dataset):
    def __init__(self, image, label, indicies):
        self.image = image
        self.label = label
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))
        ])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index].reshape((28, 28)).astype(np.float32)
        label = self.label[index]
        indicies = self.indicies[index]
        image = self.transform(image)
        return {'image': image, 'label': label, 'indicies':indicies}


class MNISTValDataset(Dataset):
    def __init__(self, image, label, indicies):
        self.image = image
        self.label = label
        self.indicies = indicies
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))
        ])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image = self.image[index].reshape((28, 28)).astype(np.float32)
        label = self.label[index]
        indicies = self.indicies[index]
        image = self.transform(image)

        return {'image': image, 'label': label, 'indicies':indicies}
