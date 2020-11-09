#!/usr/bin/env python3

''' 
Fully connected three layer neural network using PyTorch
This is an older implementation that was subsequently replace
by ids_nn.py
'''

# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler

from ids_utils import *

# 3 layer fully connected NN
class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(num_ids_features, num_ids_features)
        self.relu1 = F.relu
        self.linear2 = nn.Linear(num_ids_features, num_ids_classes)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        # x is a tensor with num_ids_classes number of entries
        return x

# Define a utility function to train the model
def train(train_dl, num_epochs, model, loss_fn, opt, sch):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)

            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
        print('Training loss: ', loss_fn(model(train_inputs), train_targets))
        sch.step()


# Set seed for reproducability
torch.manual_seed(42)

# Input data
df = ids_load_df_from_csv (outdir, balanced_data)
X_train, X_val, X_test, y_train, y_val, y_test = ids_split(df)

# Convert from numpy arrays to torch tensors
train_inputs = torch.from_numpy(X_train).float()
# cross_entropy() loss function insists on long
train_targets = torch.from_numpy(y_train).long()

# Convert from torch tensors to torch dataset
train_ds = TensorDataset(train_inputs, train_targets)

# Define data loader
batch_size = 64
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Initialize model
model = SimpleNet()

# Initialize optimizer - to run gradient descent
learning_rate = 0.004
opt = torch.optim.Adam(model.parameters(), learning_rate)

# Initialize scheduler - every step size, lr = gamma * lr
sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.50)

# Define loss function
loss_fn = F.cross_entropy

# Record initial value of loss
# The model returns num_ids_classes number of entries per input
# The loss function has to be such that it can use a single
# target value to calculate loss even though the model returns
# multiple values
loss = loss_fn(model(train_inputs), train_targets)
print ("Initial Loss:  ", loss)

# Train the model for num_epochs; default 30
num_epochs = 1
train(train_dl, num_epochs, model, loss_fn, opt, sch)

# Generate predictions for the training set (no backpropagation)
# Turn-off gradient tracking (small optimization) for forward propagation
with torch.no_grad():
    val_inputs = torch.from_numpy(X_val).float()
    val_preds = model(val_inputs)

# Since the model returns values for all num_ids_classes
# The ids_class with the maximim value is picked as the label
y_pred = val_preds.argmax(dim=1)

print ("Epochs: ", num_epochs, "Learning Rate: ", learning_rate, "Batch Size: ", batch_size)

# A numpy array is needed to evaluate metrics
y_pred = y_pred.detach().to('cpu').numpy()
ids_metrics(y_val, y_pred)
