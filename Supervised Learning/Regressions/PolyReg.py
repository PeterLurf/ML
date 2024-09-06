import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create synthetic data for polynomial regression
x = np.linspace(-1, 1, 100)
y = 0.8 * x**3 - 0.5 * x**2 + 0.1 * x + np.random.normal(0, 0.02, size=x.shape)

# Convert to torch tensors
x_train = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple polynomial regression model (degree 3)
class PolyRegression(nn.Module):
    def __init__(self):
        super(PolyRegression, self).__init__()
        self.poly = nn.Linear(4, 1)

    def forward(self, x):
        # [1, x, x^2, x^3] as input for the linear model
        x_poly = torch.cat([x**i for i in range(4)], dim=1)
        return self.poly(x_poly)

# Initialize model, loss function, and optimizer
model = PolyRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 600
history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    y_pred = model(x_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    # Save model predictions for plotting
    with torch.no_grad():
        y_pred_np = y_pred.numpy().flatten()
        history.append(y_pred_np)

# Setup the plot for animation
fig, ax = plt.subplots()
line, = ax.plot(x, history[0], color='blue')
ax.scatter(x, y, color='red', s=10)
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([min(y) - 0.1, max(y) + 0.1])

# Add a text box to display the epoch number
epoch_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))

# Update function for animation with epoch number displayed
def update(epoch):
    line.set_ydata(history[epoch])
    epoch_text.set_text(f'Epoch: {epoch + 1}/{epochs}')  # Update epoch number in the text box
    return line, epoch_text

# Animate with interval for smoothness
ani = FuncAnimation(fig, update, frames=range(epochs), blit=True, interval=30)

# Save the animation as a GIF
ani.save('PolyRegGif.gif', writer='pillow')

plt.show()
