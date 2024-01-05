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
ROBOT_RANGE = 8.0
ROBOT_FOV = 80.0
AREA_SIZE = 40.0
AREA_LEFT = -10.0
AREA_RIGHT = 30.0
AREA_TOP = 20.0
AREA_BOTTOM = -20.0
NUM_PARTICLES = 2000
NUM_STEPS = 20000
GRAPHICS_ON = False


# Create log file
path = pathlib.Path().resolve()
file = path / 'logs/pf_dataset_with_obs/log4.txt'






robot = np.zeros((2, NUM_STEPS), dtype=float)
vels = np.zeros((NUM_STEPS, 2))
robot[0, 0] = 5.0
robot[1, 0] = 5.0
vel = 1.0
dt = 1.0
u = np.array([vel, vel])
x0 = np.zeros(3)
initCov = 1.0*np.ones((2))
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

for i in range(NUM_STEPS):
  # row = 0
  # if i > 4:
  #   row = 1
  # ax = axs[row,i-5*row]

  # check if target is inside

  if insideFOV(x0, robot[:, i], ROBOT_FOV, ROBOT_RANGE):
    # filter.setProcessCovariance(0.1*np.ones(3))
    filter.predict(vels[i, :], dt)
    filter.updateWeights(robot[:,i], 0.1)
  else:
    # filter.setProcessCovariance(np.ones(3))
    filter.predict(vels[i, :], dt)
    samples = filter.getParticles()
    for j in range(NUM_PARTICLES):
      sample = samples[:, j]
      if insideFOV(x0, sample, ROBOT_FOV, ROBOT_RANGE):
        filter.weights[j] = 0.0

  # print(f"Iteration num. {i}")
  filter.resample()

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

  # add observation to file (if any)
  dummy_obs = np.zeros((2))
  if insideFOV(x0, r, ROBOT_FOV, ROBOT_RANGE):
    txt_arr = np.hstack((txt_arr, r))
  else:
    txt_arr = np.hstack((txt_arr, dummy_obs))

  txt = ""
  for elem in txt_arr:
    txt += str(elem) + " "
  txt += '\n'


  with open(str(file), 'a') as f:
    f.writelines(txt)
  # print(f"Covariance: {cov}")

  # plot_fov(ROBOT_FOV, ROBOT_RANGE, ax)
  # ax.scatter(parts[0, :], parts[1, :], s=2.0, c='y')
  # ax.plot(target[i, 0], target[i, 1], 'xr', label="Ground Truth")
  # ax.plot(mean[0], mean[1], '*b')
  # plot_ellipse(mean, cov, ax)
  # ax.set_xticks([]); ax.set_yticks([])




if GRAPHICS_ON:
  fig, ax = plt.subplots(1, 1, figsize=(6,6))
  plot_ellipse(mean, cov.reshape(2,2), ax)
  m = np.array(means_log)
  plt.plot(m[:, 0], m[:, 1])
  plt.plot(robot[0, :], robot[1, :])