import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, auto

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root = 'data',train = True, download = True,transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle = True)

test_dataset = datasets.MNIST(root='data', train=False,download=True,transfrom = transform)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle = True)


