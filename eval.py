import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math
import random

from pathlib import Path
from copy import deepcopy as dc

# custom imports
from models import *
from train_utils import *

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

path = Path().resolve()
img1_path = path / 'pics/img7.png'
img2_path = path / 'pics/img8.png'
model_path = path / 'models/model3.pth'

# len(files)

ROBOTS_NUM = 5
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0

# start in (5, 0) with low uncertainty (inside FoV)
Xtest = torch.Tensor([5.0, 0.0, 0.011218145581487034, 0.020915139555660062, 0.020915139555660062, 0.042239186289305224]).to(device)

# Load model
model = myCNN2(6,6).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

# axs[i].set_xlim([-10, 10])
# axs[i].set_ylim([-10, 10])
# ax.set_title(f"t-{i+1}")

## Forecast next steps
### Robot starting inside the FoV, then moving outside and finally coming back inside.
NUM_STEPS = 5
dt = 0.5
vel = 5.0

xp = []; yp = []
x = Xtest[0]
y = Xtest[1]
xp.append(x.cpu().detach().numpy())
yp.append(y.cpu().detach().numpy())
for _ in range(NUM_STEPS):
  x = x + dt * vel
  y = y + dt * vel
  xp.append(x.cpu().detach().numpy())
  yp.append(y.cpu().detach().numpy())

xp = np.array(xp)
yp = np.array(yp)
robot = np.vstack((xp, yp))
robot_back = np.fliplr(robot)
print(robot.shape, robot_back.shape)
robot = np.hstack((robot, robot_back[:, 1:]))
print(robot.shape)

# print(xp)

# fig, ax = plt.subplots(1, 1, figsize=(10,10))

# plot_fov(ROBOT_FOV, ROBOT_RANGE, ax)
# plt.plot(xp, yp)

NUM_STEPS = 5
dt = 0.2
vel = 2.0
fig, axs = plt.subplots(2, NUM_STEPS, figsize=(18,5))
X_i = Xtest
print(f"X test: {Xtest}")
for i in range(1, 2*NUM_STEPS+1):
  row = 0
  if i > 5:
    row = 1
  x_i = torch.from_numpy(robot[:, i]).to(device)
  cov_i = Xtest[2:]
  X_i = torch.cat((x_i, cov_i))
  # print(Xt.shape)
  y_pred = model(X_i.unsqueeze(0)).squeeze(0)
  # print(x_i)
  # print(y_pred[:2])

  # ctr = y_pred[:2]
  ctr = x_i
  cov_matrix = y_pred[2:]
  # print(f"Center: {ctr}")
  # print(f"Cov matrix: {cov_matrix}")
  cov_matrix = cov_matrix.view(2,2)
  plot_ellipse(ctr.cpu().detach().numpy(), cov_matrix.cpu().detach().numpy(), axs[row, i-1 - 5*row])
  plot_fov(ROBOT_FOV, ROBOT_RANGE, axs[row, i-1-5*row])
  axs[row, i-1-5*row].scatter(robot[0, i], robot[1, i])
  axs[row, i-1-5*row].set_xticks([]); axs[row, i-1-5*row].set_yticks([])
  axs[row, i-1-5*row].set_title(f"t+{i}")

  # update tensors for next iteration
  Xtest = y_pred
  # print(f"X_test shape: {Xtest.shape}")
  # ytest = y_test[rnd+1]

# save image
plt.savefig(str(img1_path))

"""### Robot starting inside, then moving outside and staying outside"""

NUM_STEPS = 5
dt = 0.5
vel = 5.0
Xtest = torch.Tensor([5.0, 0.0, 0.011218145581487034, 0.020915139555660062, 0.020915139555660062, 0.042239186289305224]).to(device)
xp = []; yp = []
x = Xtest[0]
y = Xtest[1]
xp.append(x.cpu().detach().numpy())
yp.append(y.cpu().detach().numpy())
for _ in range(NUM_STEPS):
  x = x + dt * vel
  y = y + dt * vel
  xp.append(x.cpu().detach().numpy())
  yp.append(y.cpu().detach().numpy())

xp = np.array(xp)
yp = np.array(yp)
robot = np.vstack((xp, yp))
x_last = x.cpu().detach().numpy() * np.ones((1, NUM_STEPS), dtype="float32")
y_last = y.cpu().detach().numpy() * np.ones((1, NUM_STEPS), dtype="float32")
robot_back = np.vstack((x_last, y_last))
print(robot.shape, robot_back.shape)
robot = np.hstack((robot, robot_back))
print(robot.shape)

NUM_STEPS = 5
dt = 0.2
vel = 2.0
fig, axs = plt.subplots(2, NUM_STEPS, figsize=(18,5))
X_i = Xtest
for i in range(1, 2*NUM_STEPS+1):
  row = 0
  if i > 5:
    row = 1
  x_i = torch.from_numpy(robot[:, i]).to(device)
  cov_i = Xtest[2:]
  X_i = torch.cat((x_i, cov_i))
  # print(Xt.shape)
  y_pred = model(X_i.unsqueeze(0)).squeeze(0)
  # print(x_i)
  # print(y_pred[:2])

  # ctr = y_pred[:2]
  ctr = x_i
  cov_matrix = y_pred[2:]
  # print(f"Center: {ctr}")
  # print(f"Cov matrix: {cov_matrix}")
  cov_matrix = cov_matrix.view(2,2)
  plot_ellipse(ctr.cpu().detach().numpy(), cov_matrix.cpu().detach().numpy(), axs[row, i-1 - 5*row])
  plot_fov(ROBOT_FOV, ROBOT_RANGE, axs[row, i-1-5*row])
  axs[row, i-1-5*row].scatter(robot[0, i], robot[1, i])
  axs[row, i-1-5*row].set_xticks([]); axs[row, i-1-5*row].set_yticks([])
  axs[row, i-1-5*row].set_title(f"t+{i}")

  # update tensors for next iteration
  Xtest = y_pred

plt.savefig(str(img2_path))

