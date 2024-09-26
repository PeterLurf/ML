import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm, auto

# Device agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Import CSV into dataframe
import pandas as pd
csv_path = "/Users/child/Documents/Programming Main copy/ML/ML/Supervised Learning/Regressions/Student_Performance.csv"
df = pd.read_csv(csv_path)
mapping = {"Yes": 1, "No": 0}
# Map activities
df['Activities_Encoded'] = df['Extracurricular Activities'].map(mapping)

# Convert to torch tensors
X = torch.tensor(df[['Hours Studied', 'Sleep Hours', 'Activities_Encoded', 'Previous Scores', 'Sample Question Papers Practiced']].values, dtype=torch.float32)
y = torch.tensor(df['Performance Index'].values, dtype=torch.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Standardize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Convert back to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# List of columns to exclude
exclude_columns = ['Performance Index', 'Extracurricular Activities']

# Identify feature columns by excluding specified columns
feature_columns = [col for col in df.columns if col not in exclude_columns]

# Count the number of features
num_features = len(feature_columns)

# Model
class MultipleLinearRegression(nn.Module):
    def __init__(self):
        super(MultipleLinearRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)
    def forward(self, x):
        return self.linear(x)
    
model = MultipleLinearRegression().to(device)

Loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 300
train_accuracy_list = []
test_accuracy_list = []
train_losses = []
test_losses = []

for epoch in tqdm(range(epochs)):
    model.train()
    y_pred = model(X_train).squeeze()
    loss = Loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Compute R² score on training data
    with torch.no_grad():
        ss_res = torch.sum((y_train - y_pred) ** 2)
        ss_tot = torch.sum((y_train - torch.mean(y_train)) ** 2)
        train_r2 = 1 - ss_res / ss_tot
        train_accuracy_list.append(train_r2.item())

    model.eval()
    with torch.inference_mode():
        y_pred_test = model(X_test).squeeze()
        loss_test = Loss_fn(y_pred_test, y_test)
        test_losses.append(loss_test.item())

        # Compute R² score on test data
        ss_res_test = torch.sum((y_test - y_pred_test) ** 2)
        ss_tot_test = torch.sum((y_test - torch.mean(y_test)) ** 2)
        test_r2 = 1 - ss_res_test / ss_tot_test
        test_accuracy_list.append(test_r2.item())

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(train_accuracy_list, label='Training R² Score')
ax1.plot(test_accuracy_list, label='Test R² Score')
ax1.set_title('R² Score')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('R² Score')
ax1.legend()

ax2.plot(train_losses, label="Train Loss")
ax2.plot(test_losses, label="Test Loss")
ax2.set_title('Training Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
plt.show()
