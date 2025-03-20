import torch
import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)

# -------------------------------
# Generate synthetic data
# -------------------------------
N = 200
# Two predictors uniformly distributed in [0, 1]
x1 = torch.rand(N)
x2 = torch.rand(N)

# True nonlinear functions:
# f1(x1) = sin(2πx1)
# f2(x2) = (x2 - 0.5)^2
f1_true = torch.sin(2 * np.pi * x1)
f2_true = (x2 - 0.5)**2

# True intercept and additive model:
alpha_true = 2.0
noise = 0.1 * torch.randn(N)
y = alpha_true + f1_true + f2_true + noise

# -------------------------------
# Define a smoothing spline function (penalized cubic fit)
# -------------------------------
def fit_smoothing_spline(x, target, lam=0.1):
    """
    Fit a cubic function to (x, target) with a penalty on the second derivative.
    For simplicity, we form a design matrix with [1, x, x^2, x^3] and add a penalty
    on the quadratic and cubic coefficients (which are proportional to the second derivative).
    """
    # Create design matrix: each row = [1, x, x^2, x^3]
    X_design = torch.stack([torch.ones_like(x), x, x**2, x**3], dim=1)
    
    # Penalty matrix: only penalize the 2nd and 3rd coefficients (associated with curvature)
    P = torch.diag(torch.tensor([0.0, 0.0, 4.0 * lam, 36.0 * lam]))
    
    # Solve (X'X + P) beta = X'target in closed form
    A = X_design.T @ X_design + P
    b = X_design.T @ target
    beta = torch.linalg.solve(A, b)
    
    # Return fitted values: f(x) = X_design @ beta
    return X_design @ beta

# -------------------------------
# Backfitting Algorithm Initialization
# -------------------------------
# Initialize intercept as the average of y and each function as zero
alpha_hat = y.mean()
f1_hat = torch.zeros(N)
f2_hat = torch.zeros(N)

# -------------------------------
# Backfitting Iterations
# -------------------------------
max_iter = 20
tol = 1e-4

print("Starting backfitting iterations:")
for iter in range(max_iter):
    # Save previous estimates for convergence check
    f1_old = f1_hat.clone()
    f2_old = f2_hat.clone()
    
    # Update f1: fit on residuals after removing current f2
    residual1 = y - alpha_hat - f2_hat
    f1_hat = fit_smoothing_spline(x1, residual1, lam=0.1)
    # Enforce identifiability: set average of f1 to zero
    f1_hat = f1_hat - f1_hat.mean()
    
    # Update f2: fit on residuals after removing updated f1
    residual2 = y - alpha_hat - f1_hat
    f2_hat = fit_smoothing_spline(x2, residual2, lam=0.1)
    f2_hat = f2_hat - f2_hat.mean()
    
    # Check convergence by looking at the change in f1 and f2
    diff = ((f1_hat - f1_old).pow(2).mean() + (f2_hat - f2_old).pow(2).mean()).sqrt()
    print(f"Iteration {iter+1}, change = {diff.item():.6f}")
    if diff < tol:
        break

# Compute final fitted values
y_hat = alpha_hat + f1_hat + f2_hat

# -------------------------------
# Visualization
# -------------------------------
# Plot f1: True function vs estimated function
x1_np = x1.numpy()
f1_true_np = f1_true.numpy()
f1_hat_np = f1_hat.numpy()
sort_idx1 = np.argsort(x1_np)

plt.figure(figsize=(8, 4))
plt.plot(x1_np[sort_idx1], f1_true_np[sort_idx1], label='True f1')
plt.plot(x1_np[sort_idx1], f1_hat_np[sort_idx1], label='Estimated f1', linestyle='--')
plt.xlabel("x1")
plt.ylabel("f1")
plt.legend()
plt.title("Function f1: True vs. Estimated")
plt.show()

# Plot f2: True function vs estimated function
x2_np = x2.numpy()
f2_true_np = f2_true.numpy()
f2_hat_np = f2_hat.numpy()
sort_idx2 = np.argsort(x2_np)

plt.figure(figsize=(8, 4))
plt.plot(x2_np[sort_idx2], f2_true_np[sort_idx2], label='True f2')
plt.plot(x2_np[sort_idx2], f2_hat_np[sort_idx2], label='Estimated f2', linestyle='--')
plt.xlabel("x2")
plt.ylabel("f2")
plt.legend()
plt.title("Function f2: True vs. Estimated")
plt.show()

# Scatter plot: True y vs. Fitted y
plt.figure(figsize=(6, 6))
plt.scatter(y.numpy(), y_hat.numpy(), alpha=0.6)
plt.xlabel("True y")
plt.ylabel("Fitted y")
plt.title("True y vs. Fitted y")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red")  # 45-degree line
plt.show()
