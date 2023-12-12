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
img1_path = path / 'pics/rnn_coverage_traj.png'
img2_path = path / 'pics/rnn_coverage_final.png'
model_path = path / 'models/rnn_coverage_model.pth'

ROBOTS_MAX = 20
ROBOTS_NUM = 12
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0
lookback = 7


# Load model
model = CoverageRNN(input_size = 2 * ROBOTS_NUM,
                    output_size = 2 * ROBOTS_NUM,
                    hidden_size = 128,
                    num_stacked_layers= 4)
model = model.to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


"""## Test on simulated robots"""

import random
N_ROBOTS = 12
robots = np.zeros((N_ROBOTS, 2), dtype="float32")
for i in range(N_ROBOTS):
  robots[i, :] = -40.0 + 40.0 * np.random.rand(1, 2)

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
Xt = Xt.view(-1, ROBOTS_NUM*2)
Xt = Xt.to(device)

robots_dummy[:ROBOTS_NUM, :]

Xt[0, 0]

"""## Simulate first steps

Simulate first `lookback` steps as a linear motion towards `(0,0)`.
"""

Xt_seq = torch.zeros((1, lookback, 2*ROBOTS_NUM)).to(device)
print(f"Xt_seq shape: {Xt_seq.shape}")
Xt_seq[:, -1, :] = dc(Xt)
vmax = 1.0
# vel_tensor = torch.Tensor([vel, vel]).to(device)
K = 0.8
dt = 0.2
# j = 0
# xi = Xt[:, 2*j:2*(j+1)]
for i in range(lookback):
  x_n = torch.zeros((1, 2*ROBOTS_NUM)).to(device)
  xi = Xt_seq[:, lookback-i-1, :]
  # print(f"xi shape: {xi.shape}")
  # print(f"xn shape: {x_n.shape}")
  for j in range(ROBOTS_NUM):
    xj = xi[0, 2*j:2*(j+1)]
    # print(f"xj shape: {xj.shape}")
    dx = -K * xi[0, 2*j]
    dy = -K * xi[0, 2*j+1]
    vx = max(-vmax, min(vmax, dx))
    vy = max(-vmax, min(vmax, dy))
    x_n[0, 2*j] = xj[0] + dt * vx
    x_n[0, 2*j+1] = xj[1] + dt * vy
  # print(x_n)

  Xt_seq[:, lookback-i-2, :] = x_n


print(f"Final shape: {Xt_seq.shape}")

"""## Plot simulated steps"""

Xt_seq = Xt_seq.view((lookback, -1, 2*ROBOTS_NUM))
for i in range(N_ROBOTS):
  plt.plot(Xt_seq[:, 0, 2*i].cpu().detach().numpy(), Xt_seq[:, 0, 2*i+1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')

"""## Forecast next steps"""

NUM_STEPS = 2000
dt = 0.2

X_hist = [Xt]
# v_hist = []

r_hist = []

Xt_seq = Xt_seq.view((-1, lookback, 2*ROBOTS_NUM))

for i in range(ROBOTS_NUM):
  r = []
  r_hist.append(r)

robots_hist = torch.Tensor(NUM_STEPS, ROBOTS_NUM, 2)
print(robots_hist.shape)

for i in range(NUM_STEPS):
  # get velocity
  v_pred = model(Xt_seq)
  if i % 100 == 0.0:
    print(f"Vpred : {v_pred}")

  # move robots
  # v = v_pred.view(ROBOTS_NUM, 2)

  # for j in range(2*ROBOTS_NUM):
  Xt_new = Xt_seq[0, 0, :] + v_pred[0, :] * dt
  Xt_new = Xt_new.unsqueeze(0)
  Xt_new = Xt_new.unsqueeze(0)
  # print(f"Xt-new shape: {Xt_new.shape}")
  Xt_seq = torch.cat((Xt_new, Xt_seq[:, :-1, :]), dim=1)
  # print(f"Actual Xt_seq shape: {Xt_seq.shape}")

  xp = Xt_new.view(ROBOTS_NUM, 2)
  for j in range(ROBOTS_NUM):
    robots_hist[i, j, :] = xp[j, :]

  # X_hist.append(Xt)

robots_hist[:, 0, :]

for i in range(N_ROBOTS):
  plt.plot(robots_hist[:, i, 0].cpu().detach().numpy(), robots_hist[:, i, 1].cpu().detach().numpy())

  # for i in range(ROBOTS_NUM):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')
plt.savefig(str(img1_path))

"""## Plot final position"""

for i in range(N_ROBOTS):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')
plt.savefig(str(img2_path))