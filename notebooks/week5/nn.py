# 1
import matplotlib.pyplot as plt
import numpy as np

# 2
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# 3
from torch import nn
import torch.nn.functional as F

# 4
from copy import deepcopy
import torch.optim as optim

CHECKPOINT   = "checkpoint.pth" 
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

#################################################################
################# HELPER FUNCTIONS ##############################
#################################################################
# Convert the polar coordinates to Cartesian coordinates.
def encode_coords_lr(r_list, theta_list):
    coords_lr = np.column_stack(
    [
        r_list * np.cos(np.pi / 4 - theta_list),
        r_list * np.sin(np.pi / 4 - theta_list),
    ]
    )
    # The shape should be (n_samples, 2).
    assert coords_lr.shape == (len(images), 2)
    coords_lr = coords_lr.astype(np.float32)

    return coords_lr

def decode_coords_lr(coords_lr):
    """Convert the rotated left-right encoding back to polar coordinates."""
    coords_complex = coords_lr @ np.array([1.0, -1.0j]) * np.exp(1.0j * np.pi / 4)
    theta = np.angle(coords_complex)
    radius = np.abs(coords_complex)
    return radius, theta



#################################################################
################# # 1 : Loading the Dataset #####################
#################################################################
with np.load("assets/data.npz") as data:
    images = data["images"]
    r_list = data["r_list"]
    theta_list = data["theta_list"]

coords_lr = encode_coords_lr(r_list, theta_list)

print(f"images.shape = {images.shape}")
print(f"r_list.shape = {r_list.shape}")
print(f"theta_list.shape = {theta_list.shape}")
print(f"coords_lr.shape = {coords_lr.shape}")

"""# Display the first 16 paired eye images.
fig, axes = plt.subplots(4, 4, figsize=(10, 4))
separator_bar = np.ones((images.shape[-2], 2))
for idx, ax in enumerate(axes.ravel()):
    ax.imshow(
        np.concatenate([images[idx, 0], separator_bar, images[idx, 1]], axis=-1),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"r={r_list[idx]:.2f}, theta={theta_list[idx]:.2f}")
    ax.axis("off")"""


#################################################################
######### 2 : Creating the Dataset splits and DataLoaders #######
#################################################################

dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(coords_lr))
split_names = ["train", "val", "test"]
split_generator = torch.Generator().manual_seed(0)
dataset_splits = dict(
    zip(split_names, random_split(dataset, [0.8, 0.1, 0.1], generator=split_generator))
)

batch_size = 32
loaders = {
    split_name: DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=split_name == "train",
    )
    for split_name, split_dataset in dataset_splits.items()
}



#################################################################
############### 3 : Constructing the Neural Network #############
#################################################################
class Model(nn.Module):
    """Convolutional regressor for target position prediction."""

    def __init__(self):
        super().__init__()
        
        # Defining the layers
        self.conv1 = nn.Conv2d(2, 8, 3, groups=2)
        self.conv2 = nn.Conv2d(8, 8, 3, groups=2)
        self.conv3 = nn.Conv2d(8, 8, 3, groups=2)
        self.fc1 = nn.LazyLinear(16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = F.tanh(self.conv3(x))
        x = x.max(-2)[0]
        x = x.flatten(1)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    



#################################################################
##################### 4 : Training the Model ####################
#################################################################


def evaluate_loss(model, data_loader, criterion):
    """Compute the mean loss over a data loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            predictions = model(inputs)
            total_loss += criterion(predictions, labels).item()
    return total_loss / len(data_loader)


model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
best_model_state = deepcopy(model.state_dict())
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0

    for inputs, labels in loaders["train"]:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(loaders["train"])
    val_loss = evaluate_loss(model, loaders["val"], criterion)
    print(f"epoch={epoch:03d}, train_loss={train_loss:0.4f}, val_loss={val_loss:0.4f}")

    if val_loss < best_val_loss:
        best_model_state = deepcopy(model.state_dict())
        best_val_loss = val_loss

model.load_state_dict(best_model_state)



#################################################################
##################### 5 : Evaluating the Model ####################
#################################################################
test_indices = dataset_splits["test"].indices
coords_lr_test = coords_lr[test_indices]
with torch.no_grad():
    coords_lr_pred = model(torch.from_numpy(images[test_indices])).numpy()

r_pred, theta_pred = decode_coords_lr(coords_lr_pred)


fig, axs = plt.subplots(1, 4, figsize=(12, 3), tight_layout=True)
for i in range(2):
    axs[i].scatter(coords_lr_test[:, i], coords_lr_pred[:, i], alpha=0.1)
    axs[i].set_xlabel("Ground truth")
    axs[i].set_ylabel("Prediction")

axs[2].scatter(theta_list[test_indices], theta_pred, alpha=0.1)
axs[2].set_xlabel("Ground truth")
axs[3].scatter(r_list[test_indices], r_pred, alpha=0.1)
axs[3].set_xlabel("Ground truth")

for ax, title in zip(axs, ["$x_L$", "$x_R$", "$\\theta$", "$r$"]):
    ax.set_title(title)

