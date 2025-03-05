import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Set seed for reproducibility
np.random.seed(42)

# Generate toy dataset with a more interesting nonlinear function
n_samples = 3000
x1 = np.random.uniform(-10, 10, n_samples)
x2 = np.random.uniform(-10, 10, n_samples)
X = np.column_stack((x1, x2))
# Define an interesting nonlinear function with noise: y = sin(x1) * cos(x2)
noise = np.random.normal(0, 0.02, n_samples)
y = np.sin(x1) * np.cos(x2) + noise

# Convert to PyTorch tensors
X_features = torch.tensor(X, dtype=torch.float32)         # shape: (n_samples, 2)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # shape: (n_samples, 1)

# Create a design matrix by adding an intercept term (column of ones)
ones = torch.ones(X_features.shape[0], 1)
X_design = torch.cat([ones, X_features], dim=1)  # shape: (n_samples, 3)

# Define the Epanechnikov kernel function.
def epanechnikov(u):
    # u: a tensor of distances normalized by the bandwidth.
    mask = (u.abs() < 1).float()  # weights only within |u| < 1
    return 0.75 * (1 - u**2) * mask

def gaussian(u):
    # u: a tensor of distances normalized by the bandwidth.
    return np.exp(-0.5 * u**2)


# Locally weighted regression function.
# x_query is the raw feature vector (without intercept).
def locally_weighted_regression(X_design, X_features, y, x_query, bandwidth):
    # Compute Euclidean distances from the query point to each training point (using features only)
    distances = torch.norm(X_features - x_query.unsqueeze(0), dim=1)
    # Compute weights using the Epanechnikov kernel
    weights = epanechnikov(distances / bandwidth)
    # Create diagonal weight matrix W
    W = torch.diag(weights)
    # Compute weighted normal equation: theta = (X^T W X)^(-1) (X^T W y)
    theta = torch.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)
    # Augment the query point with the intercept
    x_query_aug = torch.cat([torch.tensor([1.0]), x_query])
    # Return prediction as a scalar
    return (x_query_aug.unsqueeze(0) @ theta).item()

# Bandwidth parameter for the kernel
bandwidth = 1

# Create a grid for plotting the regression surface.
x1_range = np.linspace(X[:, 0].min() - 2, X[:, 0].max() + 2, 100)
x2_range = np.linspace(X[:, 1].min() - 2, X[:, 1].max() + 2, 100)
X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
Y_grid = np.zeros_like(X1_grid)

# Compute prediction via LWR for each grid point.
for i in range(X1_grid.shape[0]):
    for j in range(X1_grid.shape[1]):
        x_query = torch.tensor([X1_grid[i, j], X2_grid[i, j]], dtype=torch.float32)
        Y_grid[i, j] = locally_weighted_regression(X_design, X_features, y_tensor, x_query, bandwidth)

# Plot the regression surface along with the training data.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1_grid, X2_grid, Y_grid, cmap='viridis', alpha=0.7)
ax.scatter(X[:, 0], X[:, 1], y, color='r', label='Training Data')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Target')
ax.set_title('Locally Weighted Regression with Epanechnikov Kernel')
ax.legend()
plt.show()
