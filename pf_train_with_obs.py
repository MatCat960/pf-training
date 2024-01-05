import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import numpy as np
import math
import os

from pathlib import Path
from copy import deepcopy as dc

# custom imports
from models import *
from train_utils import *

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

path = Path().resolve()
files = path / ('logs/pf_dataset_with_obs/').glob('**/*')

# len(files)

ROBOTS_NUM = 5
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0
GRAPHICS_ON = False

data = []
sizes = []
for file in files:
  with open(file) as f:
    lines = f.readlines()
    sizes.append(len(lines))

  for l in lines:
    data.append(l)

  print(data[0])

poses = np.zeros([len(data), 8], dtype="float32")

for i in range(len(data)):
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')[:-1]))

print("Sizes: {}".format(sizes))
poses.shape

"""## Remove elements with null covariance matrix"""

# poses_new = np.zeros((6))
# print(poses[0].shape)
# for i in range(len(poses)):
#   if poses[i, 3] != 0.0:
#     poses_new = np.append(poses_new, poses[i, :], axis=0)

# poses_new.shape

"""## Convert numpy to torch.Tensor"""

y = torch.from_numpy(poses).to(device)

y.shape

"""## Generate input tensors

- Input tensor  : $X = [x_t, Σ_{t-1}]$
- Output tensor : $y = [x_t, Σ_t]$
"""

X = torch.zeros_like(y)
X[:, :2] = y[:, :2]

for i in range(1, y.shape[0]):
  X[i, 2:] = y[i-1, 2:]

X = X[1:, :]
y = y[1:, :-2]

X.shape, y.shape

"""## Create train and test split"""



train_size = int(0.8 * X.shape[0])
test_size = X.shape[0] - train_size

X, y = X.to(device), y.to(device)

# Transpose the tensors to have the channel dimensions first
# X = Zf.permute(2, 0, 1).unsqueeze(1)
# Xf = Zf.permute(2, 0, 1).unsqueeze(1)
# y = Zreal.permute(2, 0, 1).unsqueeze(1)

X.shape, y.shape



X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.2)

y_train = y_train.squeeze(1)
y_test = y_test.squeeze(1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# create TensorDatasets for training and testing sets
train_dataset = TensorDataset(X[:train_size], y[:train_size])
test_dataset = TensorDataset(X[train_size:], y[train_size:])

# create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for input, target in train_loader:
  print(input.shape, target.shape)
  break



"""## Training"""

model = PFModelWithObs(8, 6, 512)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

RUN_BATCHED = False


if RUN_BATCHED:

  epochs = 100

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
      outputs = model(inputs)
      # outputs = outputs.squeeze(1)
      print(f"Output shape: {outputs.shape}")
      print(f"Targets shape: {targets.shape}")
      outputs = outputs.unsqueeze(1)
      loss = loss_fn(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print(f"Epoch: {epoch+1} | Loss: {running_loss/len(train_loader)}")

    ### Testing
    model.eval()
    with torch.no_grad():
      running_test_loss = 0.0
      for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        test_outputs = test_outputs.unsqueeze(1)
        # test_outputs = test_outputs.view(-1, GRID_STEPS, GRID_STEPS)
        # test_outputs = test_outputs.unsqueeze(1)
        test_loss = loss_fn(test_outputs, test_targets)
        running_test_loss += loss.item()

      print(f"Epoch: {epoch+1} | Test Loss: {running_test_loss/len(test_loader)}")

"""## Train on unbatched data"""

if not RUN_BATCHED:
  epochs = 10000
  epsilon = 0.01

  for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    if epoch == 0:
      print(f"output shape: {y_pred.shape}")
      print(f"Target shape: {y_train.shape}")
    loss = loss_fn(y_pred[:, 2:], y_train[:, 2:])
    # print(f"Matrix before cholesky: {y_train[:, 2:]}")
    # cholesky_output = y_pred[:, 2:].view(-1, 2, 2)
    # loss_pos = loss_fn(y_pred[:, :2], y_train[:, :2])
    # chol = torch.matmul(cholesky_output, cholesky_output.transpose(1,2)).view(-1, 4)
    # print(f"Cholesky matrix: {chol}")
    # print(f"Chol shape: {chol.shape}")
    # print(f"y_train cov shape: {y_train[:, 2:].shape}")
    # loss_cholesky = loss_fn(chol, y_train[:, 2:])
    # loss = loss_pos + loss_cholesky
    # print(f"Position loss: {loss_pos.item()} | Covariance loss: {loss_cholesky.item()}")
    # reg_term = torch.mean(torch.abs(y_pred[2:]))
    # loss = loss + reg_term
    # loss_covariance = loss_fn(y_pred[2:], y_train[2:])
    # alpha = 0.75
    # loss = alpha * loss_position + (1-alpha) * loss_covariance
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Testing
    model.eval()
    with torch.no_grad():
      test_pred = model(X_test)
      test_loss = loss_fn(test_pred[:, 2:], y_test[:, 2:])
      # test_reg = torch.mean(torch.abs(test_pred[2:]))
      # test_loss = test_loss + test_reg

    if epoch % 100 == 0:
      print(f"Epoch: {epoch} | Loss: {loss.item()} | Test loss: {test_loss.item()}")

# for name, param in model.named_parameters():
#   if param.requires_grad:
#     print(name, param)

# SAVE TRAINED MODEL
dir_path = os.getcwd()
dir_path = os.path.join(dir_path, "models")
SAVE_MODEL_PATH = os.path.join(dir_path, "pf-model-with-obs.pth")
torch.save(model.state_dict(), SAVE_MODEL_PATH)