import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate toy data (3 clusters in 2D)
N = 300
centers = [[0, 0], [5, 5], [0, 5]]
X_np, y_true = make_blobs(n_samples=N, centers=centers, cluster_std=0.8)
X = torch.tensor(X_np, dtype=torch.float32)

# Number of clusters (assumed known) and data dimensions
K = 3
d = X.shape[1]
N_data = X.shape[0]

# ---- Initialize GMM Parameters ----
# Mixing coefficients (initialized uniformly)
pi = torch.ones(K) / K  # shape (K,)

# Initialize means by randomly choosing K data points
indices = np.random.choice(N_data, K, replace=False)
mu = X[indices]  # shape (K, d)

# Initialize covariance matrices as identity matrices for each component
Sigma = torch.stack([torch.eye(d) for _ in range(K)])  # shape (K, d, d)

# ---- EM Algorithm ----
num_iters = 100
log_likelihoods = []

for iteration in range(num_iters):
    # E-step: compute responsibilities r (N_data x K)
    r = torch.zeros(N_data, K)
    for k in range(K):
        # Create a multivariate normal distribution for component k
        dist = torch.distributions.MultivariateNormal(mu[k], covariance_matrix=Sigma[k])
        # Compute the weighted probability for each data point
        r[:, k] = pi[k] * torch.exp(dist.log_prob(X))
    # Normalize responsibilities so they sum to 1 for each data point
    r = r / r.sum(dim=1, keepdim=True)
    
    # M-step: update the parameters based on the current responsibilities
    N_k = r.sum(dim=0)  # Effective number of points assigned to each component (shape: (K,))
    
    # Update mixing coefficients
    pi = N_k / N_data
    
    for k in range(K):
        # Update means: weighted average of the data points
        mu[k] = (r[:, k].unsqueeze(1) * X).sum(dim=0) / N_k[k]
        
        # Update covariance matrices
        diff = X - mu[k]  # shape: (N_data, d)
        Sigma_k = torch.zeros(d, d)
        for n in range(N_data):
            # Outer product of the difference vector (d x d) weighted by responsibility
            Sigma_k += r[n, k] * torch.outer(diff[n], diff[n])
        Sigma[k] = Sigma_k / N_k[k]
    
    # Compute the log-likelihood for monitoring convergence
    ll = 0.0
    for n in range(N_data):
        temp = 0.0
        for k in range(K):
            dist = torch.distributions.MultivariateNormal(mu[k], covariance_matrix=Sigma[k])
            temp += pi[k] * torch.exp(dist.log_prob(X[n]))
        ll += torch.log(temp + 1e-10)  # add a small value for numerical stability
    log_likelihoods.append(ll.item())
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, log likelihood: {ll.item()}")

# Final cluster assignments: assign each data point to the component with the highest responsibility
cluster_assignments = torch.argmax(r, dim=1).numpy()

# ---- Plotting the Results ----

# Plot the clustered data points
plt.figure(figsize=(8, 6))
plt.scatter(X_np[:, 0], X_np[:, 1], c=cluster_assignments, cmap='viridis', s=30)
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Plot the log likelihood over iterations to check convergence
plt.figure(figsize=(8, 6))
plt.plot(log_likelihoods, marker='o')
plt.title("Log Likelihood over EM Iterations")
plt.xlabel("Iteration")
plt.ylabel("Log Likelihood")
plt.show()
