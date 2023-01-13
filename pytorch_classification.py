# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_gaussian_quantiles
import torch
from torch import nn
from torchmetrics import Accuracy


# Construct dataset
NUM_CLASS = 3
NUM_FEATURES = 2
RANDOM_SEED = 42
X, y = make_gaussian_quantiles(
    cov=3.0,
    n_samples=10000,
    n_features=NUM_FEATURES,
    n_classes=NUM_CLASS,
    random_state=RANDOM_SEED,
)

# Convert the input and output data from numpy arrays to torch tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
# Create device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Put data to target device
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

### BUILDING MULTI-CLASS PYTORCH MODEL ###


class MultiClassModel(nn.Module):
    """
    A PyTorch module for a multi-class classification model.

    Args:

    input_feature (int): The number of input features for the model.
    output_feature (int): The number of output features for the model.
    """

    def __init__(self, input_feature, output_feature):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=input_feature, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=output_feature),
        )

    def forward(self, x):
        return self.linear_layer(x)


model = MultiClassModel(input_feature=NUM_FEATURES, output_feature=NUM_CLASS).to(device)

# Create loss, optimizer and accuracy(from torchmetrics)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
accuracy_fn = Accuracy(task="multiclass", num_classes=NUM_CLASS).to(device)

### TRAINING LOOP ###

torch.manual_seed(RANDOM_SEED)
epochs = 100

for epoch in range(epochs):
    # Set the model to train mode
    model.train()
    # Passes the training data X_train through the model to get the output logits
    y_logits = model(X_train)
    # Applies softmax function on the logits and finds the index of the maximum value for each input
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    # Calculates the loss by passing the logits and true labels
    loss = loss_fn(y_logits, y_train)
    # Compute the train accuracy by passing the predicted classes and true labels
    train_acc = accuracy_fn(y_preds, y_train) * 100
    # Clears the gradients of the optimizer
    optimizer.zero_grad()
    # Computes the gradients of the loss function with respect to the model parameters
    loss.backward()
    # Updates the model parameters using the optimizer
    optimizer.step()
    # Sets the model to evaluation mode
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_y_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits, y_test)
        test_acc_fn = accuracy_fn(test_y_preds, y_test) * 100
    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} | train loss: {loss:.3f} | train acc: {train_acc:.2f}% | test loss: {test_loss:.3f} | test acc: {test_acc_fn:.2f}%"
        )

### OUTPUT ###

# Epoch: 0 | train loss: 1.099 | train acc: 33.20% | test loss: 1.421 | test acc: 34.10%
# Epoch: 10 | train loss: 0.248 | train acc: 95.04% | test loss: 0.208 | test acc: 93.55%
# Epoch: 20 | train loss: 0.086 | train acc: 96.39% | test loss: 0.082 | test acc: 96.50%
# Epoch: 30 | train loss: 0.071 | train acc: 97.12% | test loss: 0.075 | test acc: 96.60%
# Epoch: 40 | train loss: 0.042 | train acc: 98.44% | test loss: 0.043 | test acc: 98.35%
# Epoch: 50 | train loss: 0.038 | train acc: 98.60% | test loss: 0.034 | test acc: 98.50%
# Epoch: 60 | train loss: 0.034 | train acc: 98.79% | test loss: 0.031 | test acc: 98.80%
# Epoch: 70 | train loss: 0.066 | train acc: 96.72% | test loss: 0.104 | test acc: 95.40%
# Epoch: 80 | train loss: 0.055 | train acc: 97.36% | test loss: 0.042 | test acc: 97.95%
# Epoch: 90 | train loss: 0.031 | train acc: 98.75% | test loss: 0.039 | test acc: 98.40%
