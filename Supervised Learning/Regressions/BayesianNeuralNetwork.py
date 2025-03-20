import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define a simple regression network with dropout for MC dropout
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super(BayesianNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # Apply dropout to activate stochastic behavior even at test time
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------------
# 1. Generate Synthetic Data
# -------------------------------
# Create a quadratic function with added Gaussian noise:
#   y = x^2 + noise
np.random.seed(0)
x_np = np.linspace(-3, 3, 1000).reshape(-1, 1).astype(np.float32)
y_np = (x_np ** 2).flatten() + np.random.normal(0, 1, 1000).astype(np.float32)

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x_np)
y_tensor = torch.tensor(y_np).unsqueeze(1)

# Create a simple dataset and data loader
dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------------
# 2. Model, Loss, and Optimizer
# -------------------------------
model = BayesianNN(input_dim=1, hidden_dim=64, output_dim=1, dropout_prob=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# -------------------------------
# 3. Train the Model
# -------------------------------
num_epochs = 100
model.train()  # Ensure dropout is active during training

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss/len(dataloader):.4f}")

# -------------------------------
# 4. Prediction with Uncertainty Estimation
# -------------------------------
# To perform MC dropout, we leave dropout active during inference.
# One common trick is to call model.train() even during test time.
model.train()  # Activate dropout

num_samples = 100  # Number of stochastic forward passes
x_test_np = np.linspace(-3, 3, 200).reshape(-1, 1).astype(np.float32)
x_test_tensor = torch.tensor(x_test_np)

# Collect multiple predictions for the same test inputs
predictions_samples = []
with torch.no_grad():
    for _ in range(num_samples):
        preds = model(x_test_tensor)
        predictions_samples.append(preds.numpy())

predictions_samples = np.array(predictions_samples)  # Shape: (num_samples, 200, 1)
pred_mean = predictions_samples.mean(axis=0)
pred_std = predictions_samples.std(axis=0)

# -------------------------------
# 5. Visualization
# -------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(x_np, y_np, alpha=0.2, label="Data")
plt.plot(x_test_np, pred_mean, "r", label="Mean Prediction")
plt.fill_between(
    x_test_np.flatten(),
    (pred_mean.flatten() - 2 * pred_std.flatten()),
    (pred_mean.flatten() + 2 * pred_std.flatten()),
    color="pink", alpha=0.3, label="Uncertainty (2 std)"
)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Bayesian Neural Network using MC Dropout in PyTorch")
plt.legend()
plt.show()
