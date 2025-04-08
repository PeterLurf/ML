import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
from tqdm import tqdm  # For experiment tracking

# ----- Part 1. Generate a 3D Nautilus Shell Dataset -----
a = 0.15         # growth rate of the spiral
c = 0.2          # vertical scaling
r_tube = 0.05    # radius of the shell tube

def nautilus_surface(theta, phi, a=a, c=c, r_tube=r_tube):
    r = np.exp(a * theta)
    center = np.array([r * np.cos(theta), r * np.sin(theta), c * theta])
    
    # Derivative for the centerline to get the tangent vector T
    dx = np.exp(a*theta) * (a * np.cos(theta) - np.sin(theta))
    dy = np.exp(a*theta) * (a * np.sin(theta) + np.cos(theta))
    dz = c
    T = np.array([dx, dy, dz])
    T = T / np.linalg.norm(T)
    
    # Compute a Frenet-like frame
    ref = np.array([0, 0, 1])
    if np.abs(np.dot(T, ref)) > 0.99:
        ref = np.array([0, 1, 0])
    B = np.cross(T, ref)
    B = B / np.linalg.norm(B)
    N = np.cross(B, T)
    N = N / np.linalg.norm(N)
    
    point = center + r_tube * (np.cos(phi) * N + np.sin(phi) * B)
    return point

# Create a meshgrid for the parameter space
n_theta, n_phi = 100, 100
theta_vals = np.linspace(0, 4 * np.pi, n_theta)
phi_vals = np.linspace(0, 2 * np.pi, n_phi)
Theta, Phi = np.meshgrid(theta_vals, phi_vals)
Theta_flat = Theta.flatten()
Phi_flat = Phi.flatten()

# Generate training data points from the Nautilus surface
points = np.array([nautilus_surface(theta, phi) for theta, phi in zip(Theta_flat, Phi_flat)])
inputs = np.stack([Theta_flat, Phi_flat], axis=1).astype(np.float32)
targets = points.astype(np.float32)

# ----- Part 2. Define Fourier Features and Feed Forward Model -----
class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn((mapping_size, input_dim)) * scale, requires_grad=False)
        self.out_features = mapping_size * 2

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B.t()  # (batch, mapping_size)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLP(nn.Module):
    def __init__(self, fourier_mapping, hidden_dim=128, num_hidden_layers=3, output_dim=3):
        super().__init__()
        self.fourier_mapping = fourier_mapping
        in_dim = fourier_mapping.out_features
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.fourier_mapping(x)
        return self.network(x)

input_dim = 2  # (theta, phi)
mapping_size = 32
fourier_mapping = FourierFeatures(input_dim=input_dim, mapping_size=mapping_size, scale=10.0)
model = MLP(fourier_mapping)

# ----- Part 3. Train the Model with tqdm for Experiment Tracking -----
X = torch.from_numpy(inputs)
Y = torch.from_numpy(targets)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

num_epochs = 5000
loss_history = []

# Wrap the training loop with tqdm for a progress bar
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    optimizer.zero_grad()
    pred = model(X)
    loss = criterion(pred, Y)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# ----- Part 4. Visualize the Learned Surface and Training Data -----
n_theta_plot, n_phi_plot = 100, 100
theta_plot = np.linspace(0, 4 * np.pi, n_theta_plot)
phi_plot = np.linspace(0, 2 * np.pi, n_phi_plot)
Theta_plot, Phi_plot = np.meshgrid(theta_plot, phi_plot)
Theta_plot_flat = Theta_plot.flatten()
Phi_plot_flat = Phi_plot.flatten()
plot_inputs = np.stack([Theta_plot_flat, Phi_plot_flat], axis=1).astype(np.float32)
plot_inputs_tensor = torch.from_numpy(plot_inputs)

with torch.no_grad():
    pred_points = model(plot_inputs_tensor).cpu().numpy()
X_pred = pred_points[:, 0].reshape(n_phi_plot, n_theta_plot)
Y_pred = pred_points[:, 1].reshape(n_phi_plot, n_theta_plot)
Z_pred = pred_points[:, 2].reshape(n_phi_plot, n_theta_plot)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_pred, Y_pred, Z_pred, color='lightblue', alpha=0.7, rstride=2, cstride=2, edgecolor='none')
ax.scatter(targets[:, 0], targets[:, 1], targets[:, 2], color='red', s=1, alpha=0.5)
ax.set_title("Learned Nautilus Shell Surface and Training Data")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()
