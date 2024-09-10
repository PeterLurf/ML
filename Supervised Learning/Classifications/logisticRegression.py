import numpy as np
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#create dataset
np.random.seed(0)
X = np.random.randn(100, 2)
w_true = np.array([0.5, -0.5])
b_true = 0.1
Y = (sigmoid(X @ w_true + b_true) > 0.5).astype(int)

#convert to torch tensors
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

#split dataset
split = 80
X_train = X[:split]
Y_train = Y[:split]
X_test = X[split:]
Y_test = Y[split:]

#Define the model
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, X):
        return torch.sigmoid(self.linear(X))
    
model = LogisticRegression()

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

def accuracy(y_pred, y_true) -> float:
    y_pred_labels = (y_pred > 0.5).float()  # Convert to float for mean calculation
    return torch.mean((y_pred_labels == y_true).float()).item()  # Also convert the equality comparison to float

for epoch in range(100):
    model.train()
    Y_pred = model(X_train)
    
    # Reshape Y_train to match the shape of Y_pred
    Y_train_reshaped = Y_train.unsqueeze(1)
    
    loss = loss_fn(Y_pred, Y_train_reshaped)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_accuracy = accuracy(Y_pred, Y_train_reshaped)
    print(f'Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {train_accuracy}')

# Test the model
model.eval()
Y_pred = model(X_test)
test_accuracy = accuracy(Y_pred, Y_test)
print(f'Test Accuracy: {test_accuracy}')

# Plot the decision boundary
# Convert back to numpy for plotting
X_numpy = X.detach().numpy()
Y_numpy = Y.detach().numpy()

# Plot the decision boundary

