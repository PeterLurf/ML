import torch as torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a simple dataset
X_data = np.random.rand(100, 1)
Y_data = 2 * X_data + 1 + 0.1 * np.random.randn(100, 1)

# Convert the data to PyTorch tensors
X = torch.tensor(X_data, dtype=torch.float32)
Y = torch.tensor(Y_data, dtype=torch.float32)

# Split dataset
split = 80
X_train = X[:split]
Y_train = Y[:split]
X_test = X[split:]
Y_test = Y[split:]

# Define the model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.A = torch.nn.Parameter(torch.randn(1,1))
        self.b = torch.nn.Parameter(torch.randn(1))

    def forward(self, X):
        return self.A * X + self.b
    
model = LinearRegression()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def accuracy(y_pred, y_true) -> float:
    return torch.mean(torch.abs(y_pred - y_true) / y_true).item()

# Train the model
train_accuracy_list = []
test_accuracy_list = []
a_values = []
b_values = []

def trainingLoop(epochs):
    for epoch in range(epochs):
        model.train()
        Y_pred = model(X_train)
        loss = loss_fn(Y_pred, Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_accuracy = accuracy(Y_pred, Y_train)
        train_accuracy_list.append(train_accuracy)
        a_values.append(model.A.item())
        b_values.append(model.b.item())

        with torch.inference_mode():
            model.eval()
            Y_test_pred = model(X_test)
            loss_test = loss_fn(Y_test_pred, Y_test)
            acc = accuracy(Y_test_pred, Y_test)
            test_accuracy_list.append(acc)

trainingLoop(100)

# Create the combined figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Accuracy plots
ax1.plot(train_accuracy_list, label='Train Accuracy')
ax1.plot(test_accuracy_list, label='Test Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Animation plot
ax2.scatter(X_data, Y_data, label='Data points')
line, = ax2.plot([], [], color='red', label='Regression line')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend()

# Initialization function for the animation
def init():
    """Initialize the background of the plot."""
    line.set_data([], [])
    return line,

# Update function for the animation with reset behavior
def update(frame):
    """Update the regression line for each frame and reset when finished."""
    # Reset the frame index if it exceeds the length of a_values
    if frame >= len(a_values):
        frame = 0  # Reset to the first frame

    a = a_values[frame]
    b = b_values[frame]
    y_line = a * X_data + b
    line.set_data(X_data, y_line)
    return line,

# Create the animation with increased FPS and looping
ani = FuncAnimation(fig, update, frames=len(a_values), init_func=init, blit=True, interval=30, repeat=True)

# Adjust layout and spacing
plt.tight_layout()

# Show the combined figure with both the static accuracy plot and animated regression line
plt.show()
