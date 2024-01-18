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

print(f"Number of files : {len(files)}")

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
    # grids = np.append(grids, cov, axis=0)

    # Save positions and velocities
    for l in lines[1:]:
      data.append(l)
      grid_list.append(cov)
      # idx = np.append(idx, c)

    c += 1



# print(f"Grids shape: {grids.shape}")
grids = np.array(grid_list)
# print(f"Grids shape: {grids.shape}")

poses = np.zeros([len(data), 2], dtype="float32")
for i in range(len(data)):
  # print(i)
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')))

print(f"Poses shape: {poses.shape}")
# grids = grids[1:, :, :]
# idx = idx[1:]
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
Z = torch.from_numpy(grids).to(device)
Z = Z.to(torch.float32)
# Z = torch.tensor(grid_list).to(device)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Z shape: {Z.shape}")

X = X.view(-1, 2*ROBOTS_NUM)
y = y.view(-1, 2*ROBOTS_NUM)
Z = Z.view(X.shape[0], -1, GRID_STEPS, GRID_STEPS)
Z = Z[:, 0, :, :]
print("After view:")
print(f"X shape: {X.shape}")
print(f"Z shape: {Z.shape}")

train_size = int(0.8*X.shape[0])
X_train, y_train, Z_train = X[:train_size, :], y[:train_size, :], Z[:train_size]
X_test, y_test, Z_test = X[train_size:, :], y[train_size:, :], Z[train_size:]
print(X_train.shape, y_train.shape, Z_train.shape)
print(X_test.shape, y_test.shape, Z_test.shape)


from torch.utils.data import TensorDataset, DataLoader

# create TensorDatasets for training and testing sets
train_dataset = TensorDataset(X_train, y_train, Z_train)
test_dataset = TensorDataset(X_test, y_test, Z_test)

# create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
for input, target, grid in train_loader:
  print(input.shape, target.shape, grid.shape)
  break


model = GridCoverageModel(2 * ROBOTS_NUM, 2 * ROBOTS_NUM, 512)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

RUN_BATCHED = True


if RUN_BATCHED:
  epochs = 10

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets, grid in train_loader:
      outputs = model(inputs, grid)
      # if epoch == 0:
        # print(f"Output shape: {outputs.shape}")
        # print(f"Targets shape: {targets.shape}")

      loss = loss_fn(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    # print(f"Epoch: {epoch+1} | Loss: {running_loss/len(train_loader)}")

    ### Testing
    model.eval()
    with torch.no_grad():
      running_test_loss = 0.0
      for test_inputs, test_targets, test_ids in test_loader:
        test_outputs = model(test_inputs, test_ids)
        test_loss = loss_fn(test_outputs, test_targets)
        running_test_loss += test_loss.item()

    # if epoch % 100 == 0.0:
    print(f"Epoch: {epoch} | Loss: {running_loss/len(train_loader)} | Test Loss: {running_test_loss/len(test_loader)}")


import random
N_ROBOTS = 20
robots = np.zeros((N_ROBOTS, 2), dtype="float32")
for i in range(N_ROBOTS):
  robots[i, :] = -0.5*AREA_W + AREA_W * np.random.rand(1, 2)

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
# plt.scatter(robots[:, 0], robots[:, 1])
Xt = torch.from_numpy(robots_dummy)
Xt = Xt.view(-1, ROBOTS_NUM*2)
Xt = Xt.to(device)

robots_dummy[:ROBOTS_NUM, :]

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

###############################################################################################################

def gauss_pdf(x, y, mean, covariance):

  points = np.column_stack([x.flatten(), y.flatten()])
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob

def gmm_pdf(x, y, means, covariances, weights):
  prob = 0.0
  s = len(means)
  for i in range(s):
    prob += weights[i] * gauss_pdf(x, y, means[i], covariances[i])

  return prob
###############################################################################################################


## DEFINE GMM
TARGETS_NUM = 8
STD_DEV = 2.0
PARTICLES_NUM = 500
COMPONENTS_NUM = 4
targets = np.zeros((TARGETS_NUM, 1, 2))
for i in range(TARGETS_NUM):
  targets[i, 0, 0] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)
  targets[i, 0, 1] = -0.5*(AREA_W-1) + (AREA_W-1) * np.random.rand(1,1)

samples = np.zeros((TARGETS_NUM, PARTICLES_NUM, 2))
for k in range(TARGETS_NUM):
  for i in range(PARTICLES_NUM):
    samples[k, i, :] = targets[k, 0, :] + STD_DEV * np.random.randn(1, 2)

# Fit GMM
from sklearn.mixture import GaussianMixture

samples = samples.reshape((TARGETS_NUM*PARTICLES_NUM, 2))
print(samples.shape)
gmm = GaussianMixture(n_components=COMPONENTS_NUM, covariance_type='full', max_iter=1000)
gmm.fit(samples)

means = gmm.means_
covariances = gmm.covariances_
mix = gmm.weights_

print(f"Means: {means}")
print(f"Covs: {covariances}")
print(f"Mix: {mix}")

xg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
yg = np.linspace(-0.5*AREA_W, 0.5*AREA_W, GRID_STEPS)
Xg, Yg = np.meshgrid(xg, yg)
Xg.shape

Zt = gmm_pdf(Xg, Yg, means, covariances, mix)
Zt = Zt.reshape(GRID_STEPS, GRID_STEPS)

Zmax = np.max(Zt)
Zt = Zt / Zmax
Zt = torch.from_numpy(Zt).to(device)
Zt = Zt.to(torch.float32)
print(f"Zt shape: {Zt.shape}")

for i in range(NUM_STEPS):
  # get velocity
  v_pred = model(Xt, Zt)
  if i % 100 == 0.0:
    print(f"Vpred : {v_pred}")

  # move robots
  # v = v_pred.view(ROBOTS_NUM, 2)

  # for j in range(2*ROBOTS_NUM):
  Xt[0, :] = Xt[0, :] + v_pred[0, :] * dt
  # print(f"Actual Xt: {Xt}")

  xp = Xt.view(ROBOTS_NUM, 2)
  for j in range(ROBOTS_NUM):
    robots_hist[i, j, :] = xp[j, :]

  X_hist.append(Xt)


# # SAVE TRAINED MODEL
# dir_path = os.getcwd()
# dir_path = os.path.join(dir_path, "models")
# SAVE_MODEL_PATH = os.path.join(dir_path, "pycov_model2.pth")
# torch.save(model.state_dict(), SAVE_MODEL_PATH)