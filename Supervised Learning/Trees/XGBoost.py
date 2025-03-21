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
# XGBoost Tree Implementation
# -------------------------------------
# In XGBoost, at each node we use the following:
#   - For squared error loss: gradient = prediction - y and Hessian = 1.
#   - The optimal leaf weight is given by: -G / (H + lambda)
#   - The gain from a split is:
#       gain = 0.5 * (G_left^2/(H_left+lambda) + G_right^2/(H_right+lambda) - G_total^2/(H_total+lambda)) - gamma
# where lambda is a regularization term and gamma is the minimum required gain.

class XGBNode:
    def __init__(self, predicted_value, feature_index=None, threshold=None, left=None, right=None, gain=0.0):
        self.predicted_value = predicted_value  # Leaf prediction value
        self.feature_index = feature_index      # Feature used for splitting (if any)
        self.threshold = threshold              # Threshold for the split
        self.left = left                        # Left child node
        self.right = right                      # Right child node
        self.gain = gain                        # Gain achieved at this split

class XGBoostTree:
    def __init__(self, max_depth=3, min_samples_split=10, lambda_=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ = lambda_  # Regularization term on leaf weights
        self.gamma = gamma      # Minimum gain required to perform a split
        self.root = None
    
    def _build_tree(self, X, grad, hess, depth):
        n = grad.shape[0]
        # Sum of gradients and hessians in current node
        G = grad.sum().item()
        H = hess.sum().item()  # For squared error loss, this equals the number of samples.
        # Compute optimal leaf value for current node:
        leaf_value = -G / (H + self.lambda_)
        # Create a leaf node by default:
        node = XGBNode(predicted_value=leaf_value)
        
        # If we haven't reached maximum depth and we have enough samples, try to split
        if depth < self.max_depth and n >= self.min_samples_split:
            best_gain = -float('inf')
            best_feature = None
            best_threshold = None
            best_left_idx = None
            best_right_idx = None
            
            n_features = X.shape[1]
            for feature_index in range(n_features):
                feature_values = X[:, feature_index]
                unique_vals = torch.unique(feature_values)
                if unique_vals.numel() == 1:
                    continue  # Cannot split if all values are the same
                sorted_vals, _ = torch.sort(unique_vals)
                # Consider candidate thresholds as midpoints between unique values
                thresholds = (sorted_vals[:-1] + sorted_vals[1:]) / 2.0
                for threshold in thresholds:
                    left_idx = (feature_values < threshold)
                    right_idx = ~left_idx
                    if left_idx.sum() == 0 or right_idx.sum() == 0:
                        continue  # Skip invalid splits
                    G_left = grad[left_idx].sum().item()
                    H_left = hess[left_idx].sum().item()
                    G_right = grad[right_idx].sum().item()
                    H_right = hess[right_idx].sum().item()
                    gain = 0.5 * ((G_left**2)/(H_left + self.lambda_) +
                                  (G_right**2)/(H_right + self.lambda_) -
                                  (G**2)/(H + self.lambda_)) - self.gamma
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_index
                        best_threshold = threshold.item()
                        best_left_idx = left_idx
                        best_right_idx = right_idx
            
            # Only split if the best gain is positive
            if best_gain > 0:
                node.feature_index = best_feature
                node.threshold = best_threshold
                node.gain = best_gain
                X_left = X[best_left_idx]
                X_right = X[best_right_idx]
                grad_left = grad[best_left_idx]
                grad_right = grad[best_right_idx]
                hess_left = hess[best_left_idx]
                hess_right = hess[best_right_idx]
                node.left = self._build_tree(X_left, grad_left, hess_left, depth + 1)
                node.right = self._build_tree(X_right, grad_right, hess_right, depth + 1)
        return node
    
    def fit(self, X, grad, hess):
        self.root = self._build_tree(X, grad, hess, depth=0)
    
    def _predict_one(self, x, node):
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
# XGBoost Ensemble Implementation
# -------------------------------------
# We build an ensemble of trees in a stagewise fashion.
# Starting with an initial prediction (mean of y), at each iteration we:
#  1. Compute the gradient and Hessian for each sample (for squared error: grad = pred - y, hess = 1)
#  2. Fit a tree to these gradients/Hessians using our XGBoostTree class.
#  3. Update the prediction: pred = pred + learning_rate * tree_prediction

class XGBoostEnsemble:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=10, lambda_=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_ = lambda_
        self.gamma = gamma
        self.init_prediction = None
    
    def fit(self, X, y):
        n = y.shape[0]
        # Initialize predictions with the mean of y
        self.init_prediction = y.mean().item()
        pred = torch.full((n,), self.init_prediction)
        
        for i in range(self.n_estimators):
            # For squared error loss:
            grad = pred - y
            hess = torch.ones_like(y)
            tree = XGBoostTree(max_depth=self.max_depth, 
                               min_samples_split=self.min_samples_split, 
                               lambda_=self.lambda_, 
                               gamma=self.gamma)
            tree.fit(X, grad, hess)
            update = tree.predict(X)
            # Update predictions using the learning rate and tree output
            pred = pred + self.learning_rate * update
            self.trees.append(tree)
            mse = ((pred - y)**2).mean().item()
            print(f"Iteration {i+1}/{self.n_estimators}, Training MSE: {mse:.4f}")
    
    def predict(self, X):
        n = X.shape[0]
        pred = torch.full((n,), self.init_prediction)
        for tree in self.trees:
            pred = pred + self.learning_rate * tree.predict(X)
        return pred

# -------------------------------------
# Train XGBoost Ensemble on the Toy Regression Dataset
# -------------------------------------
xgb_model = XGBoostEnsemble(n_estimators=20, learning_rate=0.1, max_depth=3, 
                            min_samples_split=10, lambda_=1.0, gamma=0.1)
xgb_model.fit(X, y)

# Compute training Mean Squared Error (MSE)
preds = xgb_model.predict(X)
mse = ((preds - y)**2).mean().item()
print(f"\nFinal Training MSE: {mse:.4f}")

# -------------------------------------
# Regression Surface Visualization in 3D
# -------------------------------------
x1_min, x1_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
x2_min, x2_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

with torch.no_grad():
    grid_preds = xgb_model.predict(grid).numpy().reshape(xx.shape)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, grid_preds, cmap='viridis', alpha=0.7, edgecolor='none')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Predicted y')
ax.set_title('XGBoost Regression Surface')
# Overlay the original data points
ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), y.numpy(), color='red', label='Data Points')
plt.legend()
plt.show()
