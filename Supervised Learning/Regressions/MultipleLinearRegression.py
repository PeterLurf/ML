import torch
import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, auto

#import csv into dataframe
import pandas as pd

df = pd.read_csv("Student_Performance.csv")
print(df.head())