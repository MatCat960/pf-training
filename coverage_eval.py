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
img1_path = path / 'pics/coverage_img1.png'
img2_path = path / 'pics/coverage_img2.png'
model_path = path / 'models/coverage_model.pth'

# len(files)

ROBOTS_NUM = 6
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0


# Load model
model = CoverageModel2(2*ROBOTS_NUM,2*ROBOTS_NUM).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

# axs[i].set_xlim([-10, 10])
# axs[i].set_ylim([-10, 10])
# ax.set_title(f"t-{i+1}")

"""## Test on simulated robots"""

import random

robots = np.zeros((ROBOTS_NUM, 2), dtype="float32")
for i in range(ROBOTS_NUM):
  robots[i, :] = -10.0 + 20.0 * np.random.rand(1, 2)

# robots = np.array(([-4.0, 4.0],
#                   [-4.0, -4.0],
#                   [4.0, -4.0],
#                   [4.0, 4.0],
#                   [6.0, 0.0],
#                   [-6.0, 0.0]),
#                   dtype="float32")

# robots = robots - 8.0
plt.scatter(robots[:, 0], robots[:, 1])
Xt = torch.from_numpy(robots)
Xt = Xt.view(-1, ROBOTS_NUM*2)
Xt = Xt.to(device)

Xt[0, 0]

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
  # get velocity
  v_pred = model(Xt)
  # print(f"Vpred : {v_pred}")

  # move robots
  # v = v_pred.view(ROBOTS_NUM, 2)

  # for j in range(2*ROBOTS_NUM):
  Xt[0, :] = Xt[0, :] + v_pred[0, :] * dt
  # print(f"Actual Xt: {Xt}")

  xp = Xt.view(ROBOTS_NUM, 2)
  for j in range(ROBOTS_NUM):
    robots_hist[i, j, :] = xp[j, :]

  X_hist.append(Xt)

robots_hist[:, 0, :]
fig = plt.figure(figsize=(10,10))
for i in range(ROBOTS_NUM):
  plt.plot(robots_hist[:, i, 0].cpu().detach().numpy(), robots_hist[:, i, 1].cpu().detach().numpy())

  # for i in range(ROBOTS_NUM):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')
plt.savefig(str(img1_path))

"""## Plot final position"""
fig = plt.figure(figsize=(10,10))
for i in range(ROBOTS_NUM):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')
plt.savefig(str(img2_path))