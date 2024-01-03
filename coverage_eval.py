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
img1_path = path / 'pics/dropout_coverage_img1.png'
img2_path = path / 'pics/dropout_coverage_img2.png'
model_path = path / 'models/pycov_model2.pth'

# len(files)

ROBOTS_NUM = 20
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0


# Load model
model = DropoutCoverageModel(2*ROBOTS_NUM,2*ROBOTS_NUM, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

def mirror(points):
    mirrored_points = []

    # Define the corners of the square
    square_corners = [(-0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, -0.5*AREA_W), (0.5*AREA_W, 0.5*AREA_W), (-0.5*AREA_W, 0.5*AREA_W)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points

# axs[i].set_xlim([-10, 10])
# axs[i].set_ylim([-10, 10])
# ax.set_title(f"t-{i+1}")

"""## Test on simulated robots"""

import random

robots = np.zeros((ROBOTS_NUM, 2), dtype="float32")
for i in range(ROBOTS_NUM):
  robots[i, :] = -0.5*AREA_W + AREA_W * np.random.rand(1, 2)

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

NUM_STEPS = 200
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

"""## Plot final position and Voronoi partitioning"""
# fig = plt.figure(figsize=(10,10))
# for i in range(ROBOTS_NUM):
#   plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

# plt.plot(0.0, 0.0, '*')
# plt.savefig(str(img2_path))
from shapely import Polygon, Point, intersection
from scipy.spatial import Voronoi, voronoi_plot_2d

fig = plt.figure(figsize=(10,10))
# Voronoi partitioning
pts = robots_hist[-1].cpu().detach().numpy()
# mirror points across each edge of the env
dummy_points = np.zeros((5*ROBOTS_NUM, 2))
dummy_points[:ROBOTS_NUM, :] = pts
mirrored_points = mirror(pts)
mir_pts = np.array(mirrored_points)
dummy_points[ROBOTS_NUM:, :] = mir_pts
vor = Voronoi(dummy_points)

conv = True
lim_regions = []

Area_tot = 0.0

for i in range(ROBOTS_NUM):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

  region = vor.point_region[i]
  poly_vert = []
  for vert in vor.regions[region]:
    v = vor.vertices[vert]
    poly_vert.append(v)
    # plt.scatter(v[0], v[1], c='tab:red')

  poly = Polygon(poly_vert)
  x,y = poly.exterior.xy
  # plt.plot(x, y, c='tab:orange')
  # robot = np.array([-18.0, -12.0])
  robot = vor.points[i]


  # Intersect with robot range
  ROBOT_RANGE = 5.0
  step = 0.5
  range_pts = []
  for th in np.arange(0.0, 2*np.pi, step):
    xi = robot[0] + ROBOT_RANGE * np.cos(th)
    yi = robot[1] + ROBOT_RANGE * np.sin(th)
    pt = Point(xi, yi)
    range_pts.append(pt)
    # plt.plot(xi, yi, c='tab:blue')

  range_poly = Polygon(range_pts)
  xc, yc = range_poly.exterior.xy

  lim_region = intersection(poly, range_poly)
  Area_tot += lim_region.area
  # lim_regions.append(lim_region)

  xl,yl = lim_region.exterior.xy
  plt.plot(xl, yl)


plt.plot(0.0, 0.0, '*')
plt.savefig(str(img2_path))