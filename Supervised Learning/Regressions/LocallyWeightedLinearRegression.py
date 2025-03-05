import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LinearRegression

# Set random seed for reproducibility
np.random.seed(0)

# Generate toy dataset: x in [-3, 3] and y = sin(x) with added noise
N = 50
x = np.linspace(-3, 3, N)
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

# Create the design matrix with a bias term: first column ones, second column x.
# We'll use this augmented matrix for both PyTorch and scikit-learn.
X_aug = np.vstack([np.ones_like(x), x]).T

# Bandwidth parameter for the Gaussian kernel (controls locality)
tau = 0.5

def lwr_torch(x_q, X_aug, y, tau):
    """
    Locally weighted linear regression using PyTorch.
    
    Args:
      x_q (float): Query point.
      X_aug (ndarray): Training design matrix (n_samples x 2).
      y (ndarray): Training targets (n_samples,).
      tau (float): Bandwidth parameter for the Gaussian kernel.
      
    Returns:
      float: Prediction at query point.
    """
    # Extract the training x values from the augmented design matrix (column index 1)
    x_train = X_aug[:, 1]
    # Compute Gaussian weights for each training example relative to x_q
    weights = np.exp(-((x_train - x_q) ** 2) / (2 * tau ** 2))
    
    # Convert arrays to torch tensors
    X_tensor = torch.tensor(X_aug, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    W_tensor = torch.diag(torch.tensor(weights, dtype=torch.float32))
    
    # Compute the weighted closed-form solution: theta = (X^T W X)^{-1} X^T W y
    Xt_W = X_tensor.t().mm(W_tensor)
    theta = torch.inverse(Xt_W.mm(X_tensor)).mm(Xt_W).mm(y_tensor)
    
    # Create the augmented query point vector [1, x_q] and predict
    x_q_aug = torch.tensor([1, x_q], dtype=torch.float32).view(1, -1)
    y_pred = x_q_aug.mm(theta)
    return y_pred.item()

def lwr_sklearn(x_q, X_aug, y, tau):
    """
    Locally weighted linear regression using scikit-learn.
    
    Args:
      x_q (float): Query point.
      X_aug (ndarray): Training design matrix (n_samples x 2).
      y (ndarray): Training targets (n_samples,).
      tau (float): Bandwidth parameter for the Gaussian kernel.
      
    Returns:
      float: Prediction at query point.
    """
    # Extract the training x values (column index 1)
    x_train = X_aug[:, 1]
    # Compute Gaussian weights
    weights = np.exp(-((x_train - x_q) ** 2) / (2 * tau ** 2))
    
    # Create and fit a weighted linear regression model.
    # We set fit_intercept=False since X_aug already includes a bias column.
    model = LinearRegression(fit_intercept=False)
    model.fit(X_aug, y, sample_weight=weights)
    
    # Predict at the query point (must be 2D: [1, x_q])
    return model.predict(np.array([[1, x_q]]))[0]

# Define a set of query points for prediction (more finely spaced)
x_query = np.linspace(-3, 3, 100)
preds_torch = []
preds_sklearn = []

# Compute predictions for each query point using both methods
for x_q in x_query:
    preds_torch.append(lwr_torch(x_q, X_aug, y, tau))
    preds_sklearn.append(lwr_sklearn(x_q, X_aug, y, tau))

# Plot the training data and both sets of predictions
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Training data')
plt.plot(x_query, preds_torch, label='LWR Prediction (PyTorch)', linewidth=2)
plt.plot(x_query, preds_sklearn, '--', label='LWR Prediction (scikit-learn)', linewidth=2)
plt.title('Locally Weighted Linear Regression Demo')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
