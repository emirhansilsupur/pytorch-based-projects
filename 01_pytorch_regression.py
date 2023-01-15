# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from sklearn.model_selection import train_test_split

tips = sns.load_dataset("tips")

print(f"Shape of data: {tips.shape}")  # --> (244, 7)

# Use the pandas get_dummies function to perform one-hot-encoding on the categorical variables of the tips dataset
tips_enc = pd.get_dummies(tips)

# Assign the features of the dataset
X = tips_enc.drop(columns="tip", axis=1)
# Assign the target variable
y = tips_enc["tip"]


# Convert the input and output data from numpy arrays to torch tensors
X = torch.from_numpy(X.values).type(torch.float)
y = torch.from_numpy(y.values).type(torch.float)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape  # --> (torch.Size([195, 12])
y_train.shape  # --> torch.Size([195])

# Set the device to be used for training If a GPU is available, use it. Otherwise, use the CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

### BUILDING A PYTORCH LINEAR MODEL ###


class LinearModel(nn.Module):
    """
    Args:
    input_feature (int): The number of input features for the model.
    output_feature (int): The number of output features for the model.

    Returns:
    None
    """

    def __init__(self, input_feature, output_feature):
        super().__init__()
        self.linear_ = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=64),
            nn.Linear(in_features=64, out_features=32),
            nn.Linear(in_features=32, out_features=16),
            nn.Linear(in_features=16, out_features=output_feature),
        )

    def forward(self, x):
        """
        Performs a forward pass on the model.

        Args:
        x (torch.Tensor): The input data for the model.

        Returns:
        torch.Tensor: The output of the model after a forward pass.
        """
        return self.linear_(x)


# Create an instance of the model and send it to target device
model = LinearModel(input_feature=12, output_feature=1).to(device)


# Define the loss function as L1(MSE) Loss
loss_fn = nn.L1Loss()
# Define the optimizer as Stochastic Gradient Descent (SGD)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005)

# Move the training and test data to the specified device (e.g. GPU or CPU)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

### TRAINING LOOP ###

# Set the random seed for reproducibility
torch.manual_seed(42)
# Define the number of training epochs
epochs = 100
# Loop over the number of epochs
for epoch in range(epochs):
    # Set the model to train mode
    model.train()
    # Perform a forward pass on the training data
    y_pred = model(X_train).squeeze()
    # Calculate the loss on the training data
    loss = loss_fn(y_pred, y_train)
    # Zero the gradients
    optimizer.zero_grad()
    # Perform a backward pass to calculate gradients
    loss.backward()
    # Update the model parameters
    optimizer.step()
    ## Test ##
    # Set the model to evaluation mode
    model.eval()
    # Perform a forward pass on the test data
    with torch.inference_mode():
        test_pred = model(X_test).squeeze()
        # Calculate the loss on the test data
        test_loss = loss_fn(test_pred, y_test)
    # Print the loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | train loss: {loss} | test loss: {test_loss}")

### OUTPUT ###

# Epoch: 0  | train loss: 0.7548857927322388 | test loss: 0.6271986365318298
# Epoch: 10 | train loss: 0.7548364996910095 | test loss: 0.6277750730514526
# Epoch: 20 | train loss: 0.7547748684883118 | test loss: 0.6264582276344299
# Epoch: 30 | train loss: 0.7547512054443359 | test loss: 0.627041757106781
# Epoch: 40 | train loss: 0.7546840906143188 | test loss: 0.6276319622993469
# Epoch: 50 | train loss: 0.7546396851539612 | test loss: 0.6263181567192078
# Epoch: 60 | train loss: 0.7546056509017944 | test loss: 0.6269152164459229
# Epoch: 70 | train loss: 0.7545284628868103 | test loss: 0.6275190114974976
# Epoch: 80 | train loss: 0.7545038461685181 | test loss: 0.6262075304985046
# Epoch: 90 | train loss: 0.7544481754302979 | test loss: 0.6268187165260315
