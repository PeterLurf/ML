import torch
import torch.nn as nn
import numpy as np
import pandas as dataloader 
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

