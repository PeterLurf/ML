import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ----------------------------
# Generate synthetic classification data
# ----------------------------
# Create two-moon dataset
X, y = make_moons(n_samples=400, noise=0.2, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.int64)

# Convert data to PyTorch tensors
x_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# ----------------------------
# Define the HME Classification model
# ----------------------------
class HMEClassification(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, num_experts_per_branch=2, num_classes=2):
        super(HMEClassification, self).__init__()
        self.num_experts_per_branch = num_experts_per_branch
        self.num_classes = num_classes
        
        # Top-level gating network to select between two branches
        self.root_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        # Branch A gating network for expert weights
        self.gate_A = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts_per_branch),
            nn.Softmax(dim=1)
        )
        # Branch B gating network for expert weights
        self.gate_B = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts_per_branch),
            nn.Softmax(dim=1)
        )
        # Experts for branch A: each outputs logits for each class
        self.expert_A1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.expert_A2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        # Experts for branch B
        self.expert_B1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.expert_B2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # Root gating: choose between branch A and B
        root_weights = self.root_gate(x)  # shape: [batch, 2]
        
        # Branch A
        gate_A_weights = self.gate_A(x)   # shape: [batch, num_experts_per_branch]
        logits_A1 = self.expert_A1(x)       # shape: [batch, num_classes]
        logits_A2 = self.expert_A2(x)       # shape: [batch, num_classes]
        # Convert expert outputs to probabilities
        prob_A1 = F.softmax(logits_A1, dim=1)
        prob_A2 = F.softmax(logits_A2, dim=1)
        # Weighted combination of branch A experts
        branch_A_prob = gate_A_weights[:, 0:1] * prob_A1 + gate_A_weights[:, 1:2] * prob_A2
        
        # Branch B
        gate_B_weights = self.gate_B(x)   # shape: [batch, num_experts_per_branch]
        logits_B1 = self.expert_B1(x)       # shape: [batch, num_classes]
        logits_B2 = self.expert_B2(x)       # shape: [batch, num_classes]
        prob_B1 = F.softmax(logits_B1, dim=1)
        prob_B2 = F.softmax(logits_B2, dim=1)
        branch_B_prob = gate_B_weights[:, 0:1] * prob_B1 + gate_B_weights[:, 1:2] * prob_B2
        
        # Final probability as weighted sum of branch outputs
        final_prob = root_weights[:, 0:1] * branch_A_prob + root_weights[:, 1:2] * branch_B_prob
        return final_prob

# Instantiate the model
model = HMEClassification()

# ----------------------------
# Define custom Negative Log Likelihood Loss
# ----------------------------
def nll_loss(prob, target):
    eps = 1e-8
    # Gather probability for the true class for each sample
    true_prob = prob[torch.arange(prob.size(0)), target]
    return -torch.mean(torch.log(true_prob + eps))

# ----------------------------
# Training setup
# ----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000
loss_history = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_tensor)  # final probabilities, shape: [batch, num_classes]
    loss = nll_loss(output, y_tensor)
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
plt.ylabel("NLL Loss")
plt.title("Training Loss over Epochs")
plt.show()

# ----------------------------
# Plot decision boundary
# ----------------------------
# Create a meshgrid for plotting
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.from_numpy(grid.astype(np.float32))

model.eval()
with torch.no_grad():
    probs = model(grid_tensor).numpy()
    # Predicted class is the argmax of the probability vector
    Z = np.argmax(probs, axis=1).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary for HME Classification")
plt.show()
