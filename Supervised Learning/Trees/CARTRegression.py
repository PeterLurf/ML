import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------
# Data Generation: Toy Regression Dataset in ℝ²
# -------------------------------------
n_samples = 500
# Generate features uniformly in the range [-5, 5]
X = torch.FloatTensor(n_samples, 2).uniform_(-5, 5)
# Define a target function with noise: y = 2*sin(x1) + 0.5*x2 + noise
y = 2 * torch.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.5 * torch.randn(n_samples)

# -------------------------------------
# Regression Tree Implementation using PyTorch
# -------------------------------------
class Node:
    def __init__(self, sse, num_samples, predicted_value, feature_index=None, threshold=None, left=None, right=None):
        self.sse = sse  # Sum of squared errors in this node
        self.num_samples = num_samples  # Number of samples in this node
        self.predicted_value = predicted_value  # Mean value of y in this node
        self.feature_index = feature_index  # Feature used for splitting (if any)
        self.threshold = threshold          # Threshold value for the split
        self.left = left                    # Left child node
        self.right = right                  # Right child node

class RegressionTree:
    def __init__(self, max_depth=6, min_samples_split=20):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _sse(self, y):
        # Sum of squared errors for y relative to its mean
        m = y.numel()
        if m == 0:
            return 0.0
        mean_val = y.mean()
        return ((y - mean_val) ** 2).sum().item()
    
    def _build_tree(self, X, y, depth):
        m = y.numel()
        current_sse = self._sse(y)
        predicted_value = y.mean().item() if m > 0 else 0.0
        node = Node(sse=current_sse, num_samples=m, predicted_value=predicted_value)
        
        # Stop splitting if maximum depth is reached, too few samples, or no variance in y.
        if depth < self.max_depth and m >= self.min_samples_split and current_sse > 0:
            best_sse = float('inf')
            best_idx = None
            best_threshold = None
            n_features = X.shape[1]
            
            # Try splitting on each feature.
            for feature_index in range(n_features):
                feature_values = X[:, feature_index]
                unique_values = torch.unique(feature_values)
                if unique_values.numel() == 1:
                    continue  # Skip if no split is possible
                
                sorted_vals, _ = torch.sort(unique_values)
                # Consider midpoints between adjacent unique values as candidate thresholds.
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
                
                for threshold in thresholds:
                    left_mask = feature_values < threshold
                    right_mask = ~left_mask
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue  # Skip invalid splits
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    sse_left = self._sse(y_left)
                    sse_right = self._sse(y_right)
                    weighted_sse = (y_left.numel() / m) * sse_left + (y_right.numel() / m) * sse_right
                    
                    if weighted_sse < best_sse:
                        best_sse = weighted_sse
                        best_idx = feature_index
                        best_threshold = threshold.item()  # Convert to Python float
            
            # If a valid split is found, recursively build left and right subtrees.
            if best_idx is not None:
                feature_values = X[:, best_idx]
                left_mask = feature_values < best_threshold
                right_mask = ~left_mask
                X_left, y_left = X[left_mask], y[left_mask]
                X_right, y_right = X[right_mask], y[right_mask]
                
                node.feature_index = best_idx
                node.threshold = best_threshold
                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)
        return node
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
    
    def _predict_one(self, x, node):
        # Recursively traverse the tree to get the prediction.
        if node.feature_index is None:
            return node.predicted_value
        if x[node.feature_index] < node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def predict(self, X):
        preds = []
        for i in range(X.shape[0]):
            preds.append(self._predict_one(X[i], self.root))
        return torch.tensor(preds)

# -------------------------------------
# Train the Regression Tree (natively in PyTorch)
# -------------------------------------
reg_tree = RegressionTree(max_depth=6, min_samples_split=20)
reg_tree.fit(X, y)

# Compute training Mean Squared Error (MSE)
preds = reg_tree.predict(X)
mse = ((preds - y) ** 2).mean().item()
print(f"Training Mean Squared Error: {mse:.4f}")

# -------------------------------------
# Regression Surface Visualization in 3D
# -------------------------------------
# Create a grid over the feature space
x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

# Evaluate the regression tree over the grid
with torch.no_grad():
    grid_preds = reg_tree.predict(grid).numpy().reshape(xx.shape)

# 3D Plot: Regression surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, grid_preds, cmap='viridis', alpha=0.7, edgecolor='none')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Predicted y')
ax.set_title('Regression Tree Predicted Surface')

# Overlay the original data points for reference
ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), y.numpy(), color='red', label='Data Points')
plt.legend()
plt.show()
