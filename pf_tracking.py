# -*- coding: utf-8 -*-
"""pf-tracking.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_wpMsQmYpslHL81ePOMD_E8an6Z8Eh21
"""

# Imports

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy as dc

from numpy import linalg as LA
import pathlib

# import from utils files
from utils import *
from particleFilter import ParticleFilter




# Parameters
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0
AREA_SIZE = 40.0
NUM_PARTICLES = 2000
NUM_STEPS = 1000
GRAPHICS_ON = False


# Create log file
path = pathlib.Path().resolve()
file = path / 'log.txt'






robot = np.zeros((2, NUM_STEPS), dtype=np.float)
vels = np.zeros((NUM_STEPS, 2))
print(robot.shape)
robot[0, 0] = 5.0
robot[1, 0] = 5.0
vel = 1.0
dt = 1.0
u = np.array([vel, vel])
x0 = np.zeros(3)
initCov = 1.0*np.ones((2))
for i in range(1, NUM_STEPS):
  robot[:, i], vels[i, :] = move_random_get_vel(robot[:, i-1])

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