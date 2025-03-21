import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
from matplotlib.patches import Patch  # for legend proxies

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate a more complicated dataset in R^2:
N = 500
X = np.random.uniform(-1, 1, (N, 2))
# True function (without noise) for visualization:
def true_function(x):
    return np.sin(2 * np.pi * x[:, 0]) * np.cos(2 * np.pi * x[:, 1]) + np.sin(3 * np.pi * (x[:, 0] + x[:, 1]))

# Generate noisy response for training/testing
y = true_function(X) + np.random.normal(0, 0.2, N)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features and target
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Define the PPR model using PyTorch
class PPRegressor(nn.Module):
    def __init__(self, input_dim, num_terms=3, hidden_size=20):
        """
        input_dim: Dimension of input (here, 2).
        num_terms: Number of projection+ridge function terms.
        hidden_size: Number of neurons in the hidden layer for each g_m.
        """
        super(PPRegressor, self).__init__()
        self.num_terms = num_terms
        
        # Learnable projection vectors for each term (shape: num_terms x input_dim)
        self.projections = nn.Parameter(torch.randn(num_terms, input_dim))
        
        # Ridge functions implemented as small neural networks.
        # Each network takes a scalar input (the projection) and outputs a scalar.
        self.g_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            ) for _ in range(num_terms)
        ])
        
        # Learnable intercept term
        self.beta0 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, input_dim)
        Returns: Tensor of shape (batch_size, 1) with predictions.
        """
        y_pred = self.beta0.expand(x.size(0), 1)
        for m in range(self.num_terms):
            # Compute projection: w_m^T x
            z = torch.matmul(x, self.projections[m].unsqueeze(1)).squeeze(1)
            z = z.unsqueeze(1)  # reshape to (batch_size, 1)
            g_out = self.g_nets[m](z)
            y_pred = y_pred + g_out
        return y_pred

# Hyperparameters for the model
input_dim = 2
num_terms = 3      # Number of projection+ridge function components
hidden_size = 20   # Size of hidden layer in each ridge function
learning_rate = 0.01
epochs = 1000

# Instantiate the model, loss function, and optimizer
model = PPRegressor(input_dim, num_terms, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
loss_history = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate on test data
model.eval()
with torch.no_grad():
    test_preds = model(X_test_tensor)
    test_loss = criterion(test_preds, y_test_tensor)
    print(f"\nTest MSE Loss: {test_loss.item():.4f}")

# Create a grid for visualization in the 2D input space
grid_size = 100
x1_grid = np.linspace(-2, 2, grid_size)
x2_grid = np.linspace(-2, 2, grid_size)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
X_grid = np.column_stack([X1.ravel(), X2.ravel()])

# Standardize the grid using the same scaler as training data
X_grid_scaled = scaler_X.transform(X_grid)
X_grid_tensor = torch.tensor(X_grid_scaled, dtype=torch.float32)

# Predict on the grid and inverse-transform the predictions to the original scale
with torch.no_grad():
    y_grid_pred = model(X_grid_tensor).numpy().flatten()
y_grid_pred = scaler_y.inverse_transform(y_grid_pred.reshape(-1, 1)).flatten()
Y_pred_grid = y_grid_pred.reshape(X1.shape)

# Compute the true surface on the grid (without noise)
Y_true_grid = true_function(X_grid).reshape(X1.shape)

# Inverse-transform test data for plotting
X_test_orig = scaler_X.inverse_transform(X_test_scaled)
y_test_orig = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

# --- 3D Plot: Predicted Surface, True Surface, and True Test Data ---
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the predicted surface
pred_surf = ax.plot_surface(X1, X2, Y_pred_grid, cmap='viridis', alpha=0.7, edgecolor='none')

# Plot the true surface
true_surf = ax.plot_surface(X1, X2, Y_true_grid, cmap='coolwarm', alpha=0.6, edgecolor='none')

# Scatter plot for the true test data
ax.scatter(X_test_orig[:, 0], X_test_orig[:, 1], y_test_orig, color='black', s=40, label='True Test Data')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('3D Plot: Predicted Surface and True Surface with Test Data')

# Create custom legend proxies for the surfaces
proxy_pred = Patch(facecolor='lightblue', edgecolor='lightblue', label='Predicted Surface')
proxy_true = Patch(facecolor='lightcoral', edgecolor='lightcoral', label='True Surface')
ax.legend(handles=[proxy_pred, proxy_true], loc='upper left')

plt.show()
