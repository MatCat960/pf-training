# Imports

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy as dc

from numpy import linalg as LA
import pathlib
from tqdm import tqdm

# import from utils files
from utils import *
from particleFilter import ParticleFilter




# Parameters
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0
AREA_SIZE = 40.0
AREA_LEFT = -10.0
AREA_RIGHT = 30.0
AREA_TOP = 20.0
AREA_BOTTOM = -20.0
NUM_PARTICLES = 2000
NUM_STEPS = 1000
GRAPHICS_ON = True
GRID_STEPS = 64


# Create log file
path = pathlib.Path().resolve()
file = path / 'pics/pf_test.png'






robot = np.zeros((2, NUM_STEPS), dtype=float)
vels = np.zeros((NUM_STEPS, 2))
robot[0, 0] = 5.0
robot[1, 0] = 5.0
vel = 1.0
dt = 1.0
u = np.array([vel, vel])
x0 = np.array([5.0, 5.0, 0.0])
initCov = 5.0*np.ones((2))
theta = random.uniform(0, 2*math.pi)
for i in range(1, NUM_STEPS):
  # robot[:, i], vels[i, :] = move_random_get_vel(robot[:, i-1])
  if robot[0, i-1] <= AREA_LEFT or robot[0, i-1] >= AREA_RIGHT or robot[1, i-1] <= AREA_BOTTOM or robot[1, i-1] >= AREA_TOP:
    theta = random.uniform(0, 2*math.pi)
  
  vels[i, 0] = vel * math.cos(theta)
  vels[i, 1] = vel * math.sin(theta)
  robot[0, i] = max(AREA_LEFT, min(robot[0, i-1] + vel*math.cos(theta), AREA_RIGHT))
  robot[1, i] = max(AREA_BOTTOM, min(robot[1, i-1] + vel*math.sin(theta), AREA_TOP))

filter = ParticleFilter(NUM_PARTICLES, x0[:2], initCov)
samples = filter.getParticles()

# fig, axs = plt.subplots(2, 5, figsize=(18,5))

means_log = []
xg = np.linspace(AREA_LEFT, AREA_RIGHT, GRID_STEPS)
yg = np.linspace(AREA_LEFT, AREA_RIGHT, GRID_STEPS)
[xg, yg] = np.meshgrid(xg, yg)

# Features: 2 channels: prior distribution + measurements
# Labels: 1 channel : posterior distribution
Ztot = np.zeros((NUM_STEPS, 2, GRID_STEPS, GRID_STEPS), dtype="float32")       #[STEPS, channels, GRID_STEPS, GRID_STEPS]
ytot = np.zeros((NUM_STEPS, 1, GRID_STEPS, GRID_STEPS), dtype="float32")

parts = filter.getParticles()
cov = np.cov(parts)
cov = cov[:2, :2]
mean = np.mean(parts, 1)

Z = gauss_pdf(xg, yg, mean, cov)
# Z = Z / np.max(Z)
Z = Z.reshape(GRID_STEPS, GRID_STEPS)

# fig, axs = plt.subplots(2, 5, figsize=(12,8))
# row = 0
for i in tqdm(range(NUM_STEPS)):
  # row = 0
  # if i > 4:
  #   row = 1
  # ax = axs[row,i-5*row]

  # Save prior distribution
  Ztot[i, 0, :, :] = Z

  # random detection (30% chances)
  rnd = np.random.rand()
  filter.predict([0.0, 0.0], dt)
  if rnd < 0.25:
    # Robot detected
    filter.updateWeights(robot[:,i], 0.25)

    # Save measurements distribution
    Zmeas = gauss_pdf(xg, yg, robot[:, i], 0.1*np.identity(2))
    # Zmeas = Zmeas / np.max(Zmeas)
    Zmeas = Zmeas.reshape(GRID_STEPS, GRID_STEPS)
    Ztot[i, 1, :, :] = Zmeas

  # else:
  #   # filter.setProcessCovariance(np.ones(3))
  #   samples = filter.getParticles()
  #   for j in range(NUM_PARTICLES):
  #     sample = samples[:, j]
  #     if insideFOV(x0, sample, ROBOT_FOV, ROBOT_RANGE):
  #       filter.weights[j] = 0.0

  # print(f"Iteration num. {i}")
  filter.resample()

  parts = filter.getParticles()
  cov = np.cov(parts)
  cov = cov[:2, :2]
  rank = np.linalg.matrix_rank(cov)
  # if rank != 2:
  #   print("Singular matrix!")
  # mean = np.mean(parts, 1)
  mean = filter.getMean()

  Z = gauss_pdf(xg, yg, mean, cov)
  # Z = Z / np.max(Z)
  Z = Z.reshape(GRID_STEPS, GRID_STEPS)
  # print("z SHAPE: ", Z.shape)

  # Save posterior distribution
  ytot[i, 0, :, :] = Z

  # if GRAPHICS_ON:
  #   ax.scatter(xg, yg, c=Z)
  #   title = "Detected" if rnd < 0.25 else "Not Detected"
  #   ax.title.set_text(title)

# Save tensors
with open(str(path/"prior.npy"), "wb") as f:
  np.save(f, Ztot)

with open(str(path/"posterior.npy"), "wb") as f:
  np.save(f, ytot)


if GRAPHICS_ON:
  plt.savefig(str(file))
    

""" 
  parts = filter.getParticles()
  # mean = filter.getMean()
  cov = np.cov(parts)
  cov = cov[:2, :2]
  cov = cov.reshape(-1, 4)
  mean = np.mean(parts, 1)
  mean = mean.reshape(1, 2)
  mean = mean.squeeze(0)
  cov = cov.squeeze(0)
  means_log.append(mean)

  # print(mean.shape)
  # print(cov.shape)
  r = robot[:, i]
  txt_arr = np.hstack((r, cov))
  # txt = np.array2string(txt_arr)
  txt = ""
  for elem in txt_arr:
    txt += str(elem) + " "
  txt += '\n'
  """

  # with open(str(file), 'a') as f:
  #   f.writelines(txt)
  # print(f"Covariance: {cov}")

  # plot_fov(ROBOT_FOV, ROBOT_RANGE, ax)
  # ax.scatter(parts[0, :], parts[1, :], s=2.0, c='y')
  # ax.plot(target[i, 0], target[i, 1], 'xr', label="Ground Truth")
  # ax.plot(mean[0], mean[1], '*b')
  # plot_ellipse(mean, cov, ax)
  # ax.set_xticks([]); ax.set_yticks([])




