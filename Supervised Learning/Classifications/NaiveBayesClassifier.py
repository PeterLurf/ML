import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity

# ---------------------------
# Custom Gaussian Naive Bayes using PyTorch
# ---------------------------
class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = {}
        self.vars = {}
        self.priors = {}
    
    def fit(self, X, y):
        """
        Fit the model given data X and labels y.
        """
        self.classes = torch.unique(y)
        n_samples = X.shape[0]
        for c in self.classes:
            X_c = X[y == c]
            # Compute mean and variance for each feature
            self.means[int(c.item())] = X_c.mean(dim=0)
            self.vars[int(c.item())] = X_c.var(dim=0, unbiased=False)
            self.priors[int(c.item())] = X_c.shape[0] / n_samples

    def predict_proba(self, X):
        """
        Compute class probabilities for each sample in X.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        log_probs = torch.zeros(n_samples, n_classes)
        for idx, c in enumerate(self.classes):
            mean = self.means[int(c.item())]
            var = self.vars[int(c.item())]
            prior = self.priors[int(c.item())]
            # Compute the log-likelihood for each sample:
            log_likelihood = -0.5 * torch.sum(torch.log(2 * np.pi * var)) \
                             - torch.sum((X - mean)**2 / (2 * var), dim=1)
            log_prior = torch.log(torch.tensor(prior))
            log_probs[:, idx] = log_likelihood + log_prior
        
        # Convert log probabilities to probabilities
        max_log = torch.max(log_probs, dim=1, keepdim=True)[0]
        probs_exp = torch.exp(log_probs - max_log)
        probs = probs_exp / torch.sum(probs_exp, dim=1, keepdim=True)
        return probs

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1)

# ---------------------------
# Create a toy dataset using scikit-learn's make_blobs
# ---------------------------
X, y = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42, cluster_std=1.5)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ---------------------------
# Train the custom Gaussian Naive Bayes classifier
# ---------------------------
gnb = GaussianNaiveBayes()
gnb.fit(X_tensor, y_tensor)
y_pred = gnb.predict(X_tensor)
accuracy = (y_pred == y_tensor).float().mean().item()
print("Training Accuracy:", accuracy)

# ---------------------------
# Plot all figures at once in a single figure with multiple subplots
# ---------------------------
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))
plt.tight_layout(pad=4.0)

# Plot 1: Decision Boundaries with Data Scatter (Axes 0)
ax = axes[0]
# Create a meshgrid covering the data range
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid, dtype=torch.float32)
grid_preds = gnb.predict(grid_tensor).numpy()
Z = grid_preds.reshape(xx.shape)

# Plot decision boundaries and data points
ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(-0.5, len(gnb.classes)+0.5, 1), cmap='viridis')
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
ax.set_title("Decision Boundaries and Data")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

# Plot 2 & 3: Feature Distributions with Kernel Density Estimates
features = ["Feature 1", "Feature 2"]
bandwidth = 0.5  # Bandwidth for the kernel density estimator

for i in range(2):
    ax = axes[i+1]
    # Define x values for KDE plot for this feature
    x_vals = np.linspace(X[:, i].min() - 1, X[:, i].max() + 1, 200)
    for c in np.unique(y):
        # Get data for class c and feature i
        X_c = X[y == c, i][:, np.newaxis]  # make 2D array for KernelDensity
        # Plot histogram of the feature (normalized to density)
        ax.hist(X_c.ravel(), bins=20, density=True, alpha=0.5, label=f"Class {c} Histogram")
        # Fit a Kernel Density Estimator
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(X_c)
        log_dens = kde.score_samples(x_vals[:, np.newaxis])
        dens = np.exp(log_dens)
        ax.plot(x_vals, dens, label=f"Class {c} KDE")
    ax.set_title(f"Feature Distribution: {features[i]}")
    ax.set_xlabel(features[i])
    ax.set_ylabel("Density")
    ax.legend()

plt.show()
