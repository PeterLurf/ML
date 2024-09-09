import torch 
from torch import nn

def MSE(y_pred,y_true) -> float:
    #mean squared error is the average of the squares of the differences between the predicted values and the true values
    return torch.mean((y_pred - y_true) ** 2).item()

def MAE(y_pred,y_true) -> float:
    #mean absolute error is the average of the absolute differences between the predicted values and the true values
    return torch.mean(torch.abs(y_pred - y_true)).item()

def RMSE(y_pred,y_true) -> float:
    #root mean squared error is the square root of the mean squared error
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

def BCE_stock(y_pred,y_true) -> float:
    #binary cross entropy is the negative log likelihood of the true class
    return -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred)).item()