#in python context managers are used to manage resources
#they are used to allocate and release resources

#in machine learning, context managers are used to manage the training and testing of models
#example

import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1 = nn.Linear(784,128)
        self.fc2 = nn.Linear(128,10)
        
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = SimpleNN()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

#using the with statement to create a context manager
with torch.no_grad():
    for param in model.parameters():
        print(param.grad)

#the context manager is used to disable gradient calculation

#in newer versions of pytorch, we use torch.Inference_mode() instead of torch.no_grad()
#example

with torch.inference_mode():
    for param in model.parameters():
        print(param.grad)
