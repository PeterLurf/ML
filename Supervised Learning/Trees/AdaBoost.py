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
# Generate Class -1: centered at (-2, -2)
X0 = torch.randn(n_samples, 2) + torch.tensor([-2.0, -2.0])
y0 = -torch.ones(n_samples)  # labels: -1
# Generate Class +1: centered at (2, 2)
X1 = torch.randn(n_samples, 2) + torch.tensor([2.0, 2.0])
y1 = torch.ones(n_samples)   # labels: +1

# Combine the datasets
X = torch.cat((X0, X1), dim=0)
y = torch.cat((y0, y1), dim=0)

# -------------------------------------
# Decision Stump Implementation as Weak Learner
# -------------------------------------
class DecisionStump:
    def __init__(self):
        self.feature_index = None  # feature to split on
        self.threshold = None      # threshold value for splitting
        self.polarity = 1          # polarity: determines which side is positive

    def predict(self, X):
        # X is expected to be a tensor of shape (n_samples, n_features)
        n_samples = X.shape[0]
        feature_values = X[:, self.feature_index]
        # Initialize predictions with +1
        preds = torch.ones(n_samples)
        if self.polarity == 1:
            preds[feature_values < self.threshold] = -1
        else:
            preds[feature_values < self.threshold] = 1
        return preds

# -------------------------------------
# AdaBoost Algorithm Implementation
# -------------------------------------
def adaboost_train(X, y, T=20):
    """
    Trains an AdaBoost ensemble with T decision stumps.
    Returns a list of weak classifiers and their corresponding weights (alphas).
    """
    n_samples, n_features = X.shape
    # Initialize weights uniformly
    weights = torch.ones(n_samples) / n_samples

    classifiers = []
    alphas = []
    eps = 1e-10  # small value to avoid division by zero

    for t in range(T):
        stump = DecisionStump()
        best_error = float('inf')
        best_feature = None
        best_threshold = None
        best_polarity = 1

        # Iterate over all features and candidate thresholds (using unique values)
        for feature in range(n_features):
            feature_values = X[:, feature]
            thresholds = torch.unique(feature_values)
            for threshold in thresholds:
                for polarity in [1, -1]:
                    # Generate predictions for this candidate stump
                    preds = torch.ones(n_samples)
                    if polarity == 1:
                        preds[feature_values < threshold] = -1
                    else:
                        preds[feature_values < threshold] = 1

                    # Compute weighted error: sum of weights where prediction != actual label
                    misclassified = (preds != y).float()
                    error = torch.sum(weights * misclassified).item()

                    if error < best_error:
                        best_error = error
                        best_feature = feature
                        best_threshold = threshold.item()
                        best_polarity = polarity

        # Compute classifier weight (alpha)
        alpha = 0.5 * torch.log(torch.tensor((1 - best_error + eps) / (best_error + eps)))

        # Set best stump parameters
        stump.feature_index = best_feature
        stump.threshold = best_threshold
        stump.polarity = best_polarity

        classifiers.append(stump)
        alphas.append(alpha)

        # Update sample weights: w_i <- w_i * exp(-alpha * y_i * h(x_i))
        preds = stump.predict(X)
        weights = weights * torch.exp(-alpha * y * preds)
        weights = weights / weights.sum()  # normalize

        print(f"Iteration {t+1}/{T}, Error: {best_error:.4f}, Alpha: {alpha:.4f}")

    return classifiers, alphas

# Train AdaBoost ensemble
T = 20  # number of weak learners
classifiers, alphas = adaboost_train(X, y, T)

# -------------------------------------
# Ensemble Prediction
# -------------------------------------
def adaboost_predict(X, classifiers, alphas):
    """
    Computes the AdaBoost ensemble prediction for input X.
    Returns the sign of the weighted sum of predictions.
    Also returns the raw decision value for visualization.
    """
    n_samples = X.shape[0]
    # Sum predictions weighted by alpha
    ensemble_preds = torch.zeros(n_samples)
    for stump, alpha in zip(classifiers, alphas):
        ensemble_preds += alpha * stump.predict(X)
    # Final prediction: sign of the ensemble sum
    final_preds = torch.sign(ensemble_preds)
    return final_preds, ensemble_preds

# Compute training accuracy
final_preds, _ = adaboost_predict(X, classifiers, alphas)
accuracy = (final_preds == y).float().mean().item()
print(f"\nTraining Accuracy: {accuracy * 100:.2f}%")

# -------------------------------------
# 3D Visualization of Decision Surface
# -------------------------------------
# Create a grid over the feature space
x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()

# Evaluate ensemble decision value on grid
with torch.no_grad():
    _, grid_decision = adaboost_predict(grid, classifiers, alphas)
    grid_decision = grid_decision.numpy().reshape(xx.shape)

# 3D Plot: Decision surface (raw ensemble output)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(xx, yy, grid_decision, cmap='viridis', alpha=0.7, edgecolor='none')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Ensemble Output')
ax.set_title('AdaBoost Decision Surface')

# Overlay the training data points at z = -2 (for -1) and z = 2 (for +1) for visualization
ax.scatter(X0[:, 0].numpy(), X0[:, 1].numpy(), -2*np.ones(X0.shape[0]), color='red', label='Class -1')
ax.scatter(X1[:, 0].numpy(), X1[:, 1].numpy(), 2*np.ones(X1.shape[0]), color='blue', label='Class +1')

plt.legend()
plt.show()
