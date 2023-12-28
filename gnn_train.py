import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import random

from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

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
logpath = (path / 'logs/pycov/30/log.txt')
print("Logpath: {}".format(logpath))
# files = [x for x in logpath if x.is_file()]
file = logpath # if logpath.is_file()

# len(files)

ROBOTS_NUM = 30
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0



data = []
sizes = []
# for file in files:
with open(file) as f:
  lines = f.readlines()
  sizes.append(len(lines))

s = int(len(lines)/ROBOTS_NUM) * ROBOTS_NUM
lines = lines[:s]

for l in lines:
  data.append(l)

print(data[0])

poses = np.zeros([len(data), 2], dtype="float32")

for i in range(len(data)):
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')))

len(data)/ROBOTS_NUM

"""## Split poses and velocities"""

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

pos[0]

"""## Convert numpy to torch.Tensor"""

X = torch.from_numpy(pos).to(device)
y = torch.from_numpy(vel).to(device)

print(X[:ROBOTS_NUM, :])
X.shape, y.shape



# X, y = X[:-4], y[:-4]
s = int(X.shape[0]/ROBOTS_NUM)

print(f"Original shape: {X.shape}")
X = X[:s*ROBOTS_NUM]
y = y[:s*ROBOTS_NUM]
print(f"Final shape: {X.shape}")

X[:ROBOTS_NUM, :]

X = X.view(-1, ROBOTS_NUM, 2)
Y = y.view(-1, ROBOTS_NUM, 2)

X.shape, Y.shape


rnd = random.randint(0, int(X.shape[0]/ROBOTS_NUM))
print(rnd)
# X[rnd, :]

dataset = []

for i in range(X.shape[0]):
  x = X[i, :, :]#.view(-1)
  y = Y[i, :, :]#.view(-1)
  edge_index = torch.combinations(torch.arange(ROBOTS_NUM), 2).t().contiguous()
  edge_index = edge_index.to(device)
  data = Data(x=x, edge_index=edge_index, y=y)
  dataset.append(data)

x.shape, y.shape, edge_index.shape

"""## Create train and test split"""


s = len(dataset)
train_size = int(0.8 * s)
test_size = s - train_size

train_dataset = dataset[:s]
test_dataset = dataset[s:]

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset[0]





# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                    Y,
#                                                    test_size=0.2)

# y_train = y_train.squeeze(1)
# y_test = y_test.squeeze(1)
# X_train.shape, X_test.shape, y_train.shape, y_test.shape

# X_train = X_train.view(-1, 2*ROBOTS_NUM)
# X_test = X_test.view(-1, 2*ROBOTS_NUM)
# y_train = y_train.view(-1, 2*ROBOTS_NUM)
# y_test = y_test.view(-1, 2*ROBOTS_NUM)
# X_train.shape


# train_data = Data(x=X_train, edge_index=edge_index, y=y_train)
# test_data = Data(x=X_test, edge_index=edge_index, y=y_test)
# train_data, test_data


# # create TensorDatasets for training and testing sets
# # train_dataset = TensorDataset(X_train, y_train)
# # test_dataset = TensorDataset(X_test, y_test)

# # create DataLoaders
# batch_size = 16
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# for batch in train_loader:
#   print(batch)
#   break


"""## Training"""

model = GNNCoverageModel(2, 64, ROBOTS_NUM, 2)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

TRAIN_BATCHED = False

epochs = 10

if TRAIN_BATCHED:
  for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
      print(batch.x.shape, batch.y.shape)
      output = model(batch)
      loss = loss_fn(output, batch.y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * len(batch)

    avg_loss = total_loss / len(train_loader.dataset)

    ### Testing
    model.eval()
    test_loss_tot = 0.0

    with torch.no_grad():
      for batch in test_loader:
        test_output = model(batch)
        test_loss = loss_fn(test_output, batch.y)
        test_loss_tot += test_loss.item() * len(batch)

    avg_test_loss = test_loss_tot / len(test_loader.dataset)

    print(f"Epoch: {epoch} | Loss: {avg_loss} | Test Loss: {avg_test_loss}")


# SAVE TRAINED MODEL
dir_path = os.getcwd()
dir_path = os.path.join(dir_path, "models")
SAVE_MODEL_PATH = os.path.join(dir_path, "gnn_model.pth")
torch.save(model.state_dict(), SAVE_MODEL_PATH)


"""## Train on unbatched data"""

epochs = 10
LOG = True
if not TRAIN_BATCHED:
  for epoch in range(epochs):
    model.train()
    for data in train_dataset:
      # print(data.x.shape)
      vel_pred = model(data)
      if LOG:
        print(f"output shape: {vel_pred.shape}")
        print(f"Target shape: {data.y.shape}")
        LOG = False
      loss = loss_fn(vel_pred.squeeze(0), data.y)
      torch.autograd.set_detect_anomaly(True)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
      for test_data in test_dataset:
        test_pred = model(test_data)
        test_loss = loss_fn(test_pred, test_data.y)

    # if epoch % 100 == 0:
    print(f"Epoch: {epoch} | Loss: {loss.item()}")

# vel_pred.shape

# for name, param in model.named_parameters():
#   if param.requires_grad:
#     print(name, param)

"""## Test on simulated robots"""

import random
N_ROBOTS = 12
robots = np.zeros((N_ROBOTS, 2), dtype="float32")
for i in range(N_ROBOTS):
  robots[i, :] = -40.0 + 30.0 * np.random.rand(1, 2)

robots_dummy = np.zeros((ROBOTS_NUM, 2), dtype="float32")
robots_dummy[:N_ROBOTS, :] = robots
# robots = np.array(([-4.0, 4.0],
#                   [-4.0, -4.0],
#                   [4.0, -4.0],
#                   [4.0, 4.0],
#                   [6.0, 0.0],
#                   [-6.0, 0.0]),
#                   dtype="float32")

# robots = robots - 8.0
plt.scatter(robots[:, 0], robots[:, 1])
Xt = torch.from_numpy(robots_dummy)
# Xt = Xt.view(-1, ROBOTS_NUM*2)
Xt = Xt.to(device)

robots_dummy[:ROBOTS_NUM, :]

Xt.unsqueeze(0).shape

"""## Forecast next steps"""

NUM_STEPS = 500
dt = 0.2

X_hist = [Xt]
# v_hist = []

r_hist = []

for i in range(ROBOTS_NUM):
  r = []
  r_hist.append(r)

robots_hist = torch.Tensor(NUM_STEPS, ROBOTS_NUM, 2)
print(robots_hist.shape)

for i in range(NUM_STEPS):
  val_data = Data(x=Xt.unsqueeze(0), edge_index=edge_index)
  # get velocity
  v_pred = model(val_data)
  if i % 100 == 0.0:
    print(f"Vpred : {v_pred}")

  # move robots
  # v = v_pred.view(ROBOTS_NUM, 2)

  # for j in range(2*ROBOTS_NUM):
  # Xt[0, :] = Xt[0, :] + v_pred[0, :] * dt
  if i % 100 == 0.0:
    print(f"vpred shape: {v_pred.shape}")
    print(f"Xt shape: {Xt.shape}")
  Xt = Xt + v_pred * dt
  # print(f"Actual Xt: {Xt}")

  xp = Xt.view(ROBOTS_NUM, 2)
  for j in range(ROBOTS_NUM):
    robots_hist[i, j, :] = xp[j, :]

  X_hist.append(Xt)

robots_hist[:, 0, :]

for i in range(N_ROBOTS):
  plt.plot(robots_hist[:, i, 0].cpu().detach().numpy(), robots_hist[:, i, 1].cpu().detach().numpy())

  # for i in range(ROBOTS_NUM):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')

"""## Plot final position"""

for i in range(N_ROBOTS):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')