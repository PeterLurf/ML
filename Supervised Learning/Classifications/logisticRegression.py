import numpy as np
import torch 
import torch.nn as nn

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

