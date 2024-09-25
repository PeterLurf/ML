import torch
import torch.nn as nn

def L1Regularization(model, lambda_):
    l1_penalty = 0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
    return lambda_ * l1_penalty

def L2Regularization(model, lambda_):
    l2_penalty = 0
    for param in model.parameters():
        l2_penalty += torch.sum(torch.abs(param))
    return lambda_ * l2_penalty

def ElasticRegularization(model, lambda_, alpha):
    l1_penalty = 0
    l2_penalty = 0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
        l2_penalty += torch.sum(torch.abs(param))
    return lambda_ * (alpha * l1_penalty + (1 - alpha) * l2_penalty)


