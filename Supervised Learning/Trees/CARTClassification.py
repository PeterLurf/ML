import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------
# Data Generation: Two Gaussian blobs in ℝ²
# -------------------------------------
n_samples = 500
# Class 0: centered at (-2, -2)
X0 = torch.randn(n_samples, 2) + torch.tensor([-2.0, -2.0])
y0 = torch.zeros(n_samples, dtype=torch.long)
# Class 1: centered at (2, 2)
X1 = torch.randn(n_samples, 2) + torch.tensor([2.0, 2.0])
y1 = torch.ones(n_samples, dtype=torch.long)

# Combine the datasets
X = torch.cat((X0, X1), dim=0)
y = torch.cat((y0, y1), dim=0)

# -------------------------------------
# CART Algorithm Implementation using PyTorch
# -------------------------------------
class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_prob,
                 feature_index=None, threshold=None, left=None, right=None):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class  # (count_class0, count_class1)
        self.predicted_prob = predicted_prob  # probability of class 1 in this node
        self.feature_index = feature_index  # index of feature used for split
        self.threshold = threshold          # threshold value for the split
        self.left = left                    # left child node
        self.right = right                  # right child node

class CART:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _gini(self, y):
        # y is a 1D tensor of labels (0 or 1)
        m = y.numel()
        if m == 0:
            return 0.0
        count1 = (y == 1).sum().item()
        count0 = m - count1
        p0 = count0 / m
        p1 = count1 / m
        return 1.0 - (p0**2 + p1**2)
    
    def _build_tree(self, X, y, depth):
        m = y.numel()
        # Count samples for each class
        count1 = (y == 1).sum().item()
        count0 = m - count1
        predicted_prob = count1 / m if m > 0 else 0
        current_gini = self._gini(y)
        node = Node(gini=current_gini, num_samples=m,
                    num_samples_per_class=(count0, count1),
                    predicted_prob=predicted_prob)
        
        # Stop splitting if maximum depth reached, too few samples, or pure node
        if depth < self.max_depth and m >= self.min_samples_split and current_gini > 0:
            best_gini = float('inf')
            best_idx = None
            best_threshold = None
            n_features = X.shape[1]
            
            # Iterate over features
            for feature_index in range(n_features):
                feature_values = X[:, feature_index]
                # Get sorted unique values for current feature
                unique_values = torch.unique(feature_values)
                if unique_values.numel() == 1:
                    continue  # no split possible if all values are equal
                
                sorted_vals, _ = torch.sort(unique_values)
                # Consider midpoints between adjacent unique values as candidate thresholds
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
                
                for threshold in thresholds:
                    # Create binary split based on threshold
                    left_mask = feature_values < threshold
                    right_mask = ~left_mask
                    
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
                    
                    y_left = y[left_mask]
                    y_right = y[right_mask]
                    
                    gini_left = self._gini(y_left)
                    gini_right = self._gini(y_right)
                    
                    weighted_gini = (y_left.numel() / m) * gini_left + (y_right.numel() / m) * gini_right
                    
                    if weighted_gini < best_gini:
                        best_gini = weighted_gini
                        best_idx = feature_index
                        best_threshold = threshold.item()  # convert to a Python float
            
            # If a valid split is found, recursively build the tree
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
        # Recursively traverse the tree to get prediction probability for class 1
        if node.feature_index is None:
            return node.predicted_prob
        if x[node.feature_index] < node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
    
    def predict(self, X):
        # Returns predicted probability for class 1 for each sample
        preds = []
        for i in range(X.shape[0]):
            prob = self._predict_one(X[i], self.root)
            preds.append(prob)
        return torch.tensor(preds)
    
    def predict_class(self, X, threshold=0.5):
        # Convert probabilities to binary class predictions
        probs = self.predict(X)
        return (probs >= threshold).long()

# -------------------------------------
# Train the CART Model (natively in PyTorch)
# -------------------------------------
cart = CART(max_depth=5, min_samples_split=10)
cart.fit(X, y)

# Compute training accuracy
predicted_classes = cart.predict_class(X)
accuracy = (predicted_classes == y).sum().item() / y.numel()
print(f"Training Accuracy: {accuracy*100:.2f}%")

# -------------------------------------
# Decision Boundary Visualization in 3D
# -------------------------------------
# Create a grid over the feature space
x_min, x_max = X[:,0].min().item() - 1, X[:,0].max().item() + 1
y_min, y_max = X[:,1].min().item() - 1, X[:,1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

# Evaluate the CART model over the grid to get class 1 probabilities
with torch.no_grad():
    grid_probs = cart.predict(grid).numpy().reshape(xx.shape)

# 3D Plot: Decision surface (probability for class 1)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, grid_probs, cmap='viridis', alpha=0.7, edgecolor='none')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('P(Class 1)')
ax.set_title('CART Decision Surface (Probability for Class 1)')

# Overlay the original data points (using z=0 for class 0 and z=1 for class 1)
ax.scatter(X0[:,0].numpy(), X0[:,1].numpy(), 0*np.ones_like(X0[:,0].numpy()),
           color='red', label='Class 0')
ax.scatter(X1[:,0].numpy(), X1[:,1].numpy(), 1*np.ones_like(X1[:,0].numpy()),
           color='blue', label='Class 1')

plt.legend()
plt.show()
