import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# -----------------------------
# Generate a 2D dataset
# -----------------------------
np.random.seed(42)
N = 500
# Generate X in R^2 uniformly in [-1, 1]^2
X = np.random.uniform(-1, 1, (N, 2))

# Define the true underlying function (without noise) for visualization
def true_function(X):
    # A nonlinear function combining sinusoids
    return np.sin(2 * np.pi * X[:, 0]) * np.cos(2 * np.pi * X[:, 1]) + 0.5 * np.sin(3 * np.pi * (X[:, 0] + X[:, 1]))

# Generate response with added noise
noise = np.random.normal(0, 0.2, N)
y = true_function(X) + noise

# -----------------------------
# PPR with Smoothing Splines: Single-term Model
# -----------------------------
# We model y = g(w^T x), where w is parameterized as [cos(theta), sin(theta)]

# Objective function: for a given theta, compute w, project X, fit a smoothing spline, and return SSE.
def objective(theta, X, y, s=1.0):
    # Define projection vector from theta
    w = np.array([np.cos(theta), np.sin(theta)])
    # Compute projection z = w^T x
    z = X.dot(w)
    # Fit a smoothing spline to the data (z, y) with a given smoothing factor s
    spline = UnivariateSpline(z, y, s=s)
    y_pred = spline(z)
    error = np.sum((y_pred - y)**2)
    return error

# Optimize over theta in [0, 2pi] using a bounded method
res = minimize_scalar(objective, args=(X, y, 1.0), bounds=(0, 2 * np.pi), method='bounded')
theta_opt = res.x
print("Optimal theta:", theta_opt)

# Compute the optimal projection vector and final projected data
w_opt = np.array([np.cos(theta_opt), np.sin(theta_opt)])
print("Optimal w:", w_opt)
z = X.dot(w_opt)

# Fit the final smoothing spline using the optimal projection
spline_final = UnivariateSpline(z, y, s=1.0)
y_model = spline_final(z)

# -----------------------------
# Visualization: 3D Surfaces
# -----------------------------
# Create a grid in the 2D input space for visualization.
grid_size = 50
x1_grid = np.linspace(-1, 1, grid_size)
x2_grid = np.linspace(-1, 1, grid_size)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
X_grid = np.column_stack([X1.ravel(), X2.ravel()])

# Compute the projected values on the grid
z_grid = X_grid.dot(w_opt)
# Predict using the fitted spline on the grid
y_grid_pred = spline_final(z_grid)
Y_grid_pred = y_grid_pred.reshape(X1.shape)
# Compute the true function (without noise) on the grid
Y_grid_true = true_function(X_grid).reshape(X1.shape)

# 3D Plot: Predicted Surface vs. True Surface, with data points
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the predicted surface (from the spline) with opacity 0.7
pred_surf = ax.plot_surface(X1, X2, Y_grid_pred, cmap='viridis', alpha=0.7, edgecolor='none')
# Plot the true surface with a different colormap and opacity 0.6
true_surf = ax.plot_surface(X1, X2, Y_grid_true, cmap='coolwarm', alpha=0.6, edgecolor='none')

# Scatter the original data points for reference
ax.scatter(X[:, 0], X[:, 1], y, color='black', s=20, label='Data Points')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D Plot: Predicted Surface (Spline) and True Surface')
ax.legend()
plt.show()

# -----------------------------
# Visualization: Fitted Ridge Function g(z)
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(z, y, alpha=0.5, label='Data (projected)')
# For a smooth plot, generate sorted projection values
z_sorted = np.linspace(np.min(z), np.max(z), 200)
plt.plot(z_sorted, spline_final(z_sorted), color='red', linewidth=2, label='Fitted Smoothing Spline')
plt.xlabel('Projection (z = w^T x)')
plt.ylabel('y')
plt.title('Fitted Ridge Function (Smoothing Spline)')
plt.legend()
plt.show()
