import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Generate synthetic data
# ----------------------------
# Define a true function (a nonlinear combination)
def true_function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

# Create data points
N = 200
x = np.linspace(-10, 10, N).reshape(-1, 1)
y = true_function(x) + np.random.normal(0, 0.1, size=x.shape)

# Convert data to PyTorch tensors
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# ----------------------------
# Define the HME model
# ----------------------------
class HME(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_experts_per_branch=2):
        super(HME, self).__init__()
        # Top-level (root) gating network to choose between two branches
        self.root_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        # Branch A gating network to choose among experts in branch A
        self.gate_A = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts_per_branch),
            nn.Softmax(dim=1)
        )
        # Branch B gating network to choose among experts in branch B
        self.gate_B = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts_per_branch),
            nn.Softmax(dim=1)
        )
        # Experts for branch A
        self.expert_A1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.expert_A2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Experts for branch B
        self.expert_B1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.expert_B2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Get branch probabilities from the root gating network
        root_weights = self.root_gate(x)  # shape: [batch_size, 2]
        
        # For branch A: obtain gating weights and expert outputs
        gate_A_weights = self.gate_A(x)     # shape: [batch_size, 2]
        out_A1 = self.expert_A1(x)          # shape: [batch_size, 1]
        out_A2 = self.expert_A2(x)          # shape: [batch_size, 1]
        branch_A_output = gate_A_weights[:, 0:1] * out_A1 + gate_A_weights[:, 1:2] * out_A2
        
        # For branch B: obtain gating weights and expert outputs
        gate_B_weights = self.gate_B(x)     # shape: [batch_size, 2]
        out_B1 = self.expert_B1(x)          # shape: [batch_size, 1]
        out_B2 = self.expert_B2(x)          # shape: [batch_size, 1]
        branch_B_output = gate_B_weights[:, 0:1] * out_B1 + gate_B_weights[:, 1:2] * out_B2
        
        # Combine branch outputs with the root gating weights
        final_output = root_weights[:, 0:1] * branch_A_output + root_weights[:, 1:2] * branch_B_output
        return final_output

# Instantiate the model
model = HME()

# ----------------------------
# Training setup
# ----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000
loss_history = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# ----------------------------
# Plot training loss
# ----------------------------
plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss over Epochs")
plt.show()

# ----------------------------
# Evaluate and plot model predictions
# ----------------------------
model.eval()
with torch.no_grad():
    predictions = model(x_tensor).numpy()

plt.figure(figsize=(8, 4))
plt.scatter(x, y, label="Data", color="blue", s=10)
plt.plot(x, true_function(x), label="True Function", color="green", linestyle="--")
plt.plot(x, predictions, label="HME Predictions", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Hierarchical Mixtures of Experts Predictions")
plt.show()
