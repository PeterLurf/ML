import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import io
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation
X, y = make_moons(n_samples=1000, noise=0.2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = torch.from_numpy(X_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

feature_names = ["X", "Y"]

# Model definition
class NonLinearClassification(nn.Module):
    def __init__(self):
        super(NonLinearClassification, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(2, 32),  # 2 input features, 32 output features
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
images = []  # List to store images for animation

epochs = 300
for epoch in tqdm(range(epochs)):
    # Forward pass and loss computation
    y_pred = model(X_train)
    loss = loss_fn(y_pred.squeeze(dim=1), y_train)
    train_losses.append(loss.item())
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Compute test loss
    with torch.no_grad():
        y_test_pred = model(X_test)
        test_loss = loss_fn(y_test_pred.squeeze(dim=1), y_test)
        test_losses.append(test_loss.item())
    
    # Capture images every epoch
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 6), ncols=2)
    
    # Plot decision boundary
    x_min, x_max = X_train[:, 0].min().item() - 1, X_train[:, 0].max().item() + 1
    y_min, y_max = X_train[:, 1].min().item() - 1, X_train[:, 1].max().item() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_torch = torch.from_numpy(grid.astype(np.float32)).to(device)

    with torch.no_grad():
        Z = model(grid_torch)
        Z = Z.reshape(xx.shape)
        Z = Z.cpu().numpy()

    ax1.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    ax1.contour(xx, yy, Z, levels=[0.5], colors='black')
    scatter = ax1.scatter(X_train[:, 0].cpu(), X_train[:, 1].cpu(), c=y_train.cpu(),
                          cmap=plt.cm.RdBu, edgecolors='k')
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.set_title(f'Decision Boundary at Epoch {epoch + 1}')
    
    # Plot loss curves up to the current epoch
    ax2.plot(range(1, epoch + 2), train_losses, label='Train Loss')
    ax2.plot(range(1, epoch + 2), test_losses, label='Test Loss')
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Test Loss')

    plt.tight_layout()
    
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    images.append(Image.open(buf).convert('RGB'))
    

# Evaluation
with torch.inference_mode():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test.view_as(y_predicted_cls)).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc.item()}')

# Create and save the animation

images[0].save('decision_boundary_animation.gif', save_all=True, append_images=images[1:], duration=30, loop=0)
print(device)