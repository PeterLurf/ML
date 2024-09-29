import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, auto
#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data
X, y = make_moons(n_samples=1000, noise=0.2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

feature_names = ["X" , "Y"]
#model
class NonLinearClassification(nn.Module):
    def __init__(self):
        super(NonLinearClassification, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(2, 32), # 2 input features, 32 output features
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
model = NonLinearClassification().to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_losses = []
test_losses = []

epochs = 300
for epoch in tqdm(range(epochs)):
    y_pred = model(X_train)
    loss = loss_fn(y_pred.squeeze(dim=1),y_train)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_losses.append(loss.item())

    
#evaluation
with torch.inference_mode():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc.item()}')
    test_loss = loss_fn(y_predicted.squeeze(dim=1), y_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.legend()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')

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
scatter = ax2.scatter(X_train[:, 0].cpu(), X_train[:, 1].cpu(), c=y_train.cpu(), cmap=plt.cm.RdBu, edgecolors='k')
legend1 = ax2.legend(*scatter.legend_elements(), title="Classes")
ax2.add_artist(legend1)
ax2.set_xlabel(feature_names[0])
ax2.set_ylabel(feature_names[1])
ax2.set_title('Decision Boundary')

plt.tight_layout()
plt.show()