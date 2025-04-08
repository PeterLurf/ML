import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# scikit-learn imports
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import SplineTransformer

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# 1. Generate a toy 3D dataset
# ---------------------------
np.random.seed(42)

def generate_spherical_points(n, r_center, noise_scale=0.1):
    # generate points uniformly on a sphere and then scale by a radius with some noise
    X = np.random.randn(n, 3)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)  # normalize to unit sphere
    radii = r_center + np.random.normal(scale=noise_scale, size=(n, 1))
    return X * radii

n_samples = 200
# Inner sphere: class 0 (smaller radius)
X_inner = generate_spherical_points(n_samples, r_center=0.8)
y_inner = np.zeros(n_samples)
# Outer shell: class 1 (larger radius) – surrounds the inner sphere
X_outer = generate_spherical_points(n_samples, r_center=1.5)
y_outer = np.ones(n_samples)

# Combine the data
X = np.vstack([X_inner, X_outer])
y = np.hstack([y_inner, y_outer])

# Plot the raw data in 3D
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_inner[:,0], X_inner[:,1], X_inner[:,2], c='blue', label='Class 0 (Inner)')
ax.scatter(X_outer[:,0], X_outer[:,1], X_outer[:,2], c='red', label='Class 1 (Outer)')
ax.set_title("Toy 3D Dataset")
ax.legend()
plt.show()

# -------------------------------------------
# 2. Fit several SVCs with different basis functions
# -------------------------------------------
# (a) Linear SVM (original space)
model_linear = SVC(kernel='linear', C=1.0).fit(X, y)

# (b) Polynomial kernel SVM (kernel built–in)
model_poly = SVC(kernel='poly', degree=3, coef0=1, C=1.0, gamma='auto').fit(X, y)

# (c) RBF kernel SVM (radial basis function)
model_rbf = SVC(kernel='rbf', gamma='auto', C=1.0).fit(X, y)

# (d) Spline–based classifier: apply a spline basis expansion then a linear SVM.
# Note: SplineTransformer is available in scikit–learn >=1.1.
model_spline = Pipeline([
    ('spline', SplineTransformer(degree=3, n_knots=5, include_bias=False)),
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='linear', C=1.0))
]).fit(X, y)

models = {
    "Linear": model_linear,
    "Polynomial kernel": model_poly,
    "RBF kernel": model_rbf,
    "Spline expansion": model_spline
}

# -------------------------------------------------
# Helper function: Plot 3D decision boundary isosurface
# -------------------------------------------------
def plot_decision_boundary_3d(model, title, grid_size=30, transform_func=None):
    # Determine grid boundaries based on the data range
    pad = 0.5
    xmin, ymin, zmin = X.min(axis=0) - pad
    xmax, ymax, zmax = X.max(axis=0) + pad
    
    # Create a 3D grid
    xx, yy, zz = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size),
        np.linspace(zmin, zmax, grid_size)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    # If a transformation is needed before calling decision_function, apply it
    if transform_func is not None:
        grid_points_transformed = transform_func(grid_points)
    else:
        grid_points_transformed = grid_points
        
    # Some models (like pipelines) support decision_function directly.
    try:
        decision_values = model.decision_function(grid_points_transformed)
    except Exception as e:
        # if decision_function is not available, use predict_proba and take log-odds.
        decision_values = model.predict_proba(grid_points_transformed)[:, 1] - 0.5

    decision_values = decision_values.reshape(xx.shape)

    # Use marching cubes to extract the 0–level isosurface
    # Note: For 3D, the decision boundary is a 2D surface embedded in 3D.
    verts, faces, normals, _ = measure.marching_cubes(decision_values, level=0, spacing=( (xmax-xmin)/grid_size, (ymax-ymin)/grid_size, (zmax-zmin)/grid_size ))
    
    # Shift vertices to the correct coordinate system
    verts += np.array([xmin, ymin, zmin])
    
    # Plot the isosurface and the data points
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    # Plot the extracted surface
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = Poly3DCollection(verts[faces], alpha=0.3, edgecolor='none')
    mesh.set_facecolor("green")
    ax.add_collection3d(mesh)
    
    # Plot training points
    ax.scatter(X_inner[:,0], X_inner[:,1], X_inner[:,2], c='blue', label='Class 0', s=20)
    ax.scatter(X_outer[:,0], X_outer[:,1], X_outer[:,2], c='red', label='Class 1', s=20)
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.legend()
    plt.show()

# Plot decision boundaries for each scikit–learn model
for name, model in models.items():
    plot_decision_boundary_3d(model, f"Decision Boundary: {name}")

# -------------------------------------------------
# 3. PyTorch Implementation: Linear SVM with a fixed polynomial basis (degree=2)
# -------------------------------------------------
# For a 3D input, degree 2 polynomial features include:
# x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3.
def poly_features_torch(x):
    # x: tensor of shape (n_samples, 3)
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    x3 = x[:, 2:3]
    features = torch.cat([x1, x2, x3,
                          x1**2, x2**2, x3**2,
                          x1*x2, x1*x3, x2*x3], dim=1)
    return features

# Prepare data for PyTorch (convert labels to {-1, 1})
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(2*y - 1, dtype=torch.float32).unsqueeze(1)  # convert 0,1 -> -1,1

# Define a simple linear model on the polynomial features
class PolySVM(nn.Module):
    def __init__(self, in_features):
        super(PolySVM, self).__init__()
        self.linear = nn.Linear(in_features, 1)
    def forward(self, x):
        # x is assumed to be raw input (in R^3)
        phi = poly_features_torch(x)
        return self.linear(phi)

model_torch = PolySVM(in_features=9)
optimizer = optim.Adam(model_torch.parameters(), lr=0.01)
n_epochs = 200
loss_history = []

# Hinge loss function
def hinge_loss(outputs, labels):
    # labels expected to be -1 or 1
    return torch.mean(torch.clamp(1 - labels * outputs, min=0))

# Training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model_torch(X_tensor)
    loss = hinge_loss(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

# Plot training loss curve for the PyTorch model
plt.figure(figsize=(6,4))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Hinge Loss")
plt.title("PyTorch SVM Training Loss (Degree 2 Polynomial Basis)")
plt.legend()
plt.show()

# Create a decision function for the PyTorch model
def decision_function_torch(x_np):
    # x_np: numpy array of shape (n_samples, 3)
    x_t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        out = model_torch(x_t)
    return out.numpy().ravel()

# Plot decision boundary for the PyTorch model
# We use the same grid as before.
def plot_decision_boundary_3d_torch(decision_func, title, grid_size=30):
    pad = 0.5
    xmin, ymin, zmin = X.min(axis=0) - pad
    xmax, ymax, zmax = X.max(axis=0) + pad
    xx, yy, zz = np.meshgrid(
        np.linspace(xmin, xmax, grid_size),
        np.linspace(ymin, ymax, grid_size),
        np.linspace(zmin, zmax, grid_size)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    decision_values = decision_func(grid_points)
    decision_values = decision_values.reshape(xx.shape)
    verts, faces, normals, _ = measure.marching_cubes(decision_values, level=0, spacing=((xmax-xmin)/grid_size, (ymax-ymin)/grid_size, (zmax-zmin)/grid_size))
    verts += np.array([xmin, ymin, zmin])
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    mesh = Poly3DCollection(verts[faces], alpha=0.3, edgecolor='none')
    mesh.set_facecolor("purple")
    ax.add_collection3d(mesh)
    ax.scatter(X_inner[:,0], X_inner[:,1], X_inner[:,2], c='blue', label='Class 0', s=20)
    ax.scatter(X_outer[:,0], X_outer[:,1], X_outer[:,2], c='red', label='Class 1', s=20)
    ax.set_title(title)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.legend()
    plt.show()

plot_decision_boundary_3d_torch(decision_function_torch, "Decision Boundary: PyTorch SVM (Poly Basis, Degree 2)")
