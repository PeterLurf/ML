# %%
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


# %%

# Device configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# %%
# Load datasets
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=100)


# %%
# Define the model
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

# %%

model = FullyConnectedNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Statistics tracking
train_losses = []
test_losses = []

# %%
epochs = 10
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        pred_labels = model(images)
        loss = loss_fn(pred_labels, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_test_loss = 0
    with torch.inference_mode():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            pred_labels = model(images)
            loss = loss_fn(pred_labels, labels)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_losses.append(avg_test_loss)
    
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")


# %%
# Plotting the average losses per epoch
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


