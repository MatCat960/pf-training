import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
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
# logpath = (path / 'logs/dynamic_coverage_vel3/').glob('**/*')
logpath = (path / 'logs/pycov/gmm_coverage').glob('**/*')
print("Logpath: {}".format(logpath))
files = [x for x in logpath if x.is_file()]

len(files)

ROBOTS_NUM = 20
lookback = 7
AREA_W = 40.0
GRID_STEPS = 100
ROBOT_RANGE = 10.0
ROBOT_FOV = 120.0


data = []
sizes = []
grid_list = []
grids = np.zeros((1, GRID_STEPS, GRID_STEPS))
idx = np.zeros((1))
c = 0
count = 0
for file in files:
  with open(file) as f:
    lines = f.readlines()
    sizes.append(len(lines))


    # Save grid first
    cov = lines[0][:-2]
    cov = cov.replace('\n', '')
    cov = cov.replace(' ', ',')
    cov = tuple(map(float, cov.split(',')))
    cov = np.array(cov)
    cov = cov.reshape(1, GRID_STEPS, GRID_STEPS)
    grids = np.append(grids, cov, axis=0)

    # Save positions and velocities
    for l in lines[1:]:
      data.append(l)
      # grid_list.append(cov)
      idx = np.append(idx, c)

    c += 1



# print(f"Grids shape: {grids.shape}")

poses = np.zeros([len(data), 2], dtype="float32")
for i in range(len(data)):
  # print(i)
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')))

print(f"Poses shape: {poses.shape}")
grids = grids[1:, :, :]
idx = idx[1:]
print(f"Grids shape: {grids.shape}")
print(f"Idx shape: {idx.shape}")

pos = np.zeros((int(len(data)/2), 2), dtype="float32")
vel = np.zeros_like(pos)

print(poses[0])
print(f"Original length: {len(poses)}")
print(f"Pos length: {len(pos)}")

for i in range(0, len(pos)):
  pos[i] = poses[2*i]
  vel[i] = poses[2*i+1]

pos[pos == 100.0] = 0.0
vel[vel == 99.9] = 0.0

"""## Convert numpy to torch.Tensor"""

X = torch.from_numpy(pos).to(device)
y = torch.from_numpy(vel).to(device)
# Z = torch.tensor(grid_list).to(device)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
# print(f"Z shape: {Z.shape}")

# X[:ROBOTS_NUM, :]
# X.shape, y.shape

# # X, y = X[:-4], y[:-4]
# s = int(X.shape[0]/ROBOTS_NUM)

# X = X[:s*ROBOTS_NUM]
# y = y[:s*ROBOTS_NUM]

# X = X.view(-1, 2*ROBOTS_NUM)
# y = y.view(-1, 2*ROBOTS_NUM)

# X.shape, y.shape

# """## Scale values"""

# from sklearn.preprocessing import MinMaxScaler

# # scaler = MinMaxScaler(feature_range=(-1,1))
# # X_scaled = scaler.fit_transform(X.cpu())
# # # y_scaled = scaler.fit_transform(y.cpu())
# # X_new = np.float32(X_scaled)
# # # y_new = y_scaled

# # X = torch.from_numpy(X_new)
# # print(type(X))
# # y = torch.from_numpy(y_new)

# """## Create train and test split"""

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                    y,
#                                                    test_size=0.2)

# y_train = y_train.squeeze(1)
# y_test = y_test.squeeze(1)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# from torch.utils.data import TensorDataset, DataLoader

# # create TensorDatasets for training and testing sets
# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)

# # create DataLoaders
# batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# for input, target in train_loader:
#   print(input.shape, target.shape)
#   break


# """## Training"""

# model = DropoutCoverageModel(2 * ROBOTS_NUM, 2 * ROBOTS_NUM, device)
# model = model.to(device)
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# RUN_BATCHED = False

# """## Train on unbatched data"""

# if not RUN_BATCHED:
#   epochs = 5000
#   epsilon = 0.01

#   for epoch in range(epochs):
#     model.train()
#     y_pred = model(X_train)
#     if epoch == 0:
#       print(f"output shape: {y_pred.shape}")
#       print(f"Target shape: {y_train.shape}")
#     loss = loss_fn(y_pred, y_train)
#     torch.autograd.set_detect_anomaly(True)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     ### Testing
#     model.eval()
#     with torch.inference_mode():
#       test_pred = model(X_test)
#       test_loss = loss_fn(test_pred, y_test)

#     if epoch % 100 == 0:
#       print(f"Epoch: {epoch} | Loss: {loss.item()} | Test loss: {test_loss.item()}")


# """## Train on batched data"""
# if RUN_BATCHED:
#   epochs = 100

#   for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for inputs, targets in train_loader:
#       outputs = model(inputs)
#       # if epoch == 0:
#         # print(f"Output shape: {outputs.shape}")
#         # print(f"Targets shape: {targets.shape}")

#       loss = loss_fn(outputs, targets)
#       optimizer.zero_grad()
#       loss.backward()
#       optimizer.step()
#       running_loss += loss.item()

#     # print(f"Epoch: {epoch+1} | Loss: {running_loss/len(train_loader)}")

#     ### Testing
#     model.eval()
#     with torch.no_grad():
#       running_test_loss = 0.0
#       for test_inputs, test_targets in test_loader:
#         test_outputs = model(test_inputs)
#         test_loss = loss_fn(test_outputs, test_targets)
#         running_test_loss += test_loss.item()

#     # if epoch % 100 == 0.0:
#       print(f"Epoch: {epoch+1} | Loss: {running_loss/len(train_loader)} | Test Loss: {running_test_loss/len(test_loader)}")


# # for name, param in model.named_parameters():
# #   if param.requires_grad:
# #     print(name, param)

# # SAVE TRAINED MODEL
# dir_path = os.getcwd()
# dir_path = os.path.join(dir_path, "models")
# SAVE_MODEL_PATH = os.path.join(dir_path, "pycov_model2.pth")
# torch.save(model.state_dict(), SAVE_MODEL_PATH)