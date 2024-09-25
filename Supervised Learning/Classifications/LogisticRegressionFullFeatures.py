import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
y_train = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1).to(device)  # Shape: (N, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).unsqueeze(1).to(device)    # Shape: (N, 1)

# Logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        logits = self.linear(x)  # No sigmoid here if using BCEWithLogitsLoss
        return logits

model = LogisticRegression(n_features).to(device)

# Loss and optimizer
# Use BCEWithLogitsLoss which combines a sigmoid layer and the BCELoss in one single class.
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 300

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in tqdm(range(epochs)):
    # Forward pass
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    # Compute training accuracy
    with torch.no_grad():
        preds_train = torch.sigmoid(logits)
        preds_train = (preds_train >= 0.5).float()
        train_accuracy = (preds_train == y_train).float().mean().item()
        train_accuracies.append(train_accuracy)
    
    # Evaluate on test data
    with torch.no_grad():
        logits_test = model(X_test)
        loss_test = loss_fn(logits_test, y_test)
        test_losses.append(loss_test.item())
        
        preds_test = torch.sigmoid(logits_test)
        preds_test = (preds_test >= 0.5).float()
        test_accuracy = (preds_test == y_test).float().mean().item()
        test_accuracies.append(test_accuracy)

    # Optionally, print progress every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {loss.item():.4f} "
              f"Test Loss: {loss_test.item():.4f} "
              f"Train Acc: {train_accuracy*100:.2f}% "
              f"Test Acc: {test_accuracy*100:.2f}%")

# Plot train and test losses
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('BCEWithLogitsLoss')
plt.legend()

# Plot train and test accuracies
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
