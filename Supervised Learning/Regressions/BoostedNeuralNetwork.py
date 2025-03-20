import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate a synthetic binary classification dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                           n_informative=2, n_clusters_per_class=1, 
                           flip_y=0.1, random_state=42)
# Convert to torch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)

# Define a simple neural network as a weak learner
class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=10, output_dim=2):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Boosting parameters
T = 5  # Number of boosting rounds
# Initialize sample weights uniformly
sample_weights = torch.ones(len(X_tensor), device=device) / len(X_tensor)

models = []
model_weights = []

# Boosting loop: Train T weak learners sequentially
for t in range(T):
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train the model for a fixed number of epochs
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        # Compute per-sample loss (no reduction so we can weight each sample)
        loss_values = nn.CrossEntropyLoss(reduction='none')(outputs, y_tensor)
        weighted_loss = (sample_weights * loss_values).mean()
        weighted_loss.backward()
        optimizer.step()
    
    # Evaluate the model on the training data
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = outputs.argmax(dim=1)
        incorrect = (predictions != y_tensor).float()  # 1 for misclassified, 0 otherwise
    
    # Compute the weighted error
    weighted_error = (sample_weights * incorrect).sum().item()
    
    # If the error is too high, skip this round
    if weighted_error > 0.5:
        print(f"Round {t}: error too high ({weighted_error:.2f}), skipping this round.")
        continue
    
    # Avoid division by zero
    if weighted_error == 0:
        weighted_error = 1e-10
    
    # Compute the model weight (alpha) using AdaBoost update rule
    alpha = 0.5 * np.log((1 - weighted_error) / weighted_error)
    models.append(model)
    model_weights.append(alpha)
    
    # Update sample weights: correctly classified samples get a factor of exp(-alpha)
    # and misclassified samples get exp(alpha)
    sample_weights = sample_weights * torch.exp(alpha * (incorrect * 2 - 1))
    # Normalize sample weights
    sample_weights = sample_weights / sample_weights.sum()
    
    print(f"Round {t}: weighted error = {weighted_error:.2f}, alpha = {alpha:.2f}")

# Function to combine predictions from all models in the ensemble
def boosted_predict(X_data):
    X_data = torch.FloatTensor(X_data).to(device)
    ensemble_output = torch.zeros(X_data.shape[0], device=device)
    # Each model votes: we map predictions from {0,1} to {-1, 1} and weight them by alpha
    for model, alpha in zip(models, model_weights):
        outputs = model(X_data)
        preds = outputs.argmax(dim=1).float() * 2 - 1  # Convert 0 -> -1 and 1 -> 1
        ensemble_output += alpha * preds
    # Final prediction is the sign of the ensemble output, mapping back to {0,1}
    final_pred = (torch.sign(ensemble_output) + 1) // 2
    return final_pred.cpu().numpy()

# Function to plot the decision boundary
def plot_decision_boundary(X, y, predict_func):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_func(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, levels=[-0.5, 0.5, 1.5], cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title("Boosted Neural Network Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X, y, boosted_predict)
