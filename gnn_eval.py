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
img1_path = path / 'pics/gnn_coverage_img1.png'
img2_path = path / 'pics/gnn_coverage_img2.png'
model_path = path / 'models/gnn_model.pth'

# len(files)

ROBOTS_MAX = 30
ROBOTS_NUM = 12
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0
AREA_W = 40.0


# Load model
model = GNNCoverageModel(2*ROBOTS_MAX,2*ROBOTS_MAX, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

# axs[i].set_xlim([-10, 10])
# axs[i].set_ylim([-10, 10])
# ax.set_title(f"t-{i+1}")

"""## Test on simulated robots"""

import random
N_ROBOTS = 12
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
plt.savefig(str(img1_path))


"""## Plot final position"""
fig = plt.figure(figsize=(10,10))
for i in range(ROBOTS_NUM):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')
plt.savefig(str(img2_path))