import torch
import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, auto

def ElasticRegularization(model, lambda_, alpha):
    l1_penalty = 0
    l2_penalty = 0
    for param in model.parameters():
        l1_penalty += torch.sum(torch.abs(param))
        l2_penalty += torch.sum(torch.abs(param))
    return lambda_ * (alpha * l1_penalty + (1 - alpha) * l2_penalty)



# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

# Use only the first two features for decision boundary plot
X = X[:, :2]
feature_names = bc.feature_names[:2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

y_train = y_train.view(y_train.shape[0], 1)  # reshaping the y_train tensor to a column vector
y_test = y_test.view(y_test.shape[0], 1)  # reshaping the y_test tensor to a column vector

# Logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(2).to(device)  # Using only 2 features
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 300

train_losses = []
test_losses = []

for epoch in tqdm(range(epochs)):
    # Forward pass
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train) + ElasticRegularization(model=model,lambda_=0.01,alpha=0.5)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    train_losses.append(loss.item())
    
    # Evaluate on test data
    with torch.inference_mode():
        y_pred_test = model(X_test)
        loss_test = loss_fn(y_pred_test, y_test) + ElasticRegularization(model=model,lambda_=0.01,alpha=0.5)
        test_losses.append(loss_test.item())

# Plot train and test losses
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(train_losses, label='train loss')
ax1.plot(test_losses, label='test loss')
ax1.legend()

# Plot decision boundary
# Prepare data for plotting decision boundary
x_min, x_max = X_train[:, 0].min().item() - 1, X_train[:, 0].max().item() + 1
y_min, y_max = X_train[:, 1].min().item() - 1, X_train[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Stack the grid points to create (N, 2) shape input
grid = np.c_[xx.ravel(), yy.ravel()]
grid_torch = torch.from_numpy(grid.astype(np.float32)).to(device)

# Get model predictions for each point in the grid
with torch.no_grad():
    Z = model(grid_torch)
    Z = Z.reshape(xx.shape)
    Z = Z.cpu().numpy()  # Move to CPU and convert to NumPy array

# Plot the contour and training examples
ax2.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
ax2.contour(xx, yy, Z, levels=[0.5], colors='black')  # Decision boundary at probability 0.5
ax2.scatter(X_train[:, 0].cpu(), X_train[:, 1].cpu(), c=y_train[:, 0].cpu(), cmap=plt.cm.RdBu, edgecolors='k')
ax2.set_xlabel(feature_names[0])
ax2.set_ylabel(feature_names[1])
ax2.set_title('Decision Boundary')

plt.tight_layout()

plt.show()
