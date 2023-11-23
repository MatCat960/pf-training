# Imports

import numpy as np
import math

from copy import deepcopy as dc

# ------------------------------------- Particle Filter class ------------------------------------- #

class ParticleFilter():
  def __init__(self, particles_num, initState, initCovariance):
    """
      Particle Filter constructor
      Args:
        particles_num (int): number of particles
        initState (np.array): 2D or 3D coordinates array
        initCovariance (np.array): 2D or 3D array (considered as diagonal matrix)
    """
    self.n = particles_num
    self.state = initState
    self.covariance = initCovariance
    self.size = initState.size

    self.particles = np.zeros((self.size, self.n), dtype=float)
    self.weights = 1 / self.n * np.ones((self.n), dtype=float)
    # print("Weights shape: {}".format(self.weights.shape))

    sigma_x = initCovariance[0]
    sigma_y = initCovariance[1]
    # sigma_th = initCovariance[2]

    for i in range(self.n):
      self.particles[0, i] = np.random.normal(self.state[0], sigma_x)
      self.particles[1, i] = np.random.normal(self.state[1], sigma_y)
      # self.particles[2, i] = np.random.normal(self.state[2], sigma_th)


  def setProcessCovariance(self, cov):
    """
    Set process covariance.
    Args:
      cov (np.array): 2D or 3D array (considered as diagonal matrix)
    """

    self.covariance = cov


  def predict(self, vel, dt=0.2):
    sigma_x = self.covariance[0]
    sigma_y = self.covariance[1]

    for i in range(self.n):
      x_next = self.particles[0, i] + vel[0]*dt
      y_next = self.particles[1, i] + vel[1]*dt

      self.particles[0, i] = np.random.normal(x_next, sigma_x)
      self.particles[1, i] = np.random.normal(y_next, sigma_y)

    self.mean = np.mean(self.particles)

  def predict2(self, mean, sigma):
    x_n = mean[0]
    y_n = mean[1]

    for i in range(self.n):
      self.particles[0, i] = np.random.normal(x_n, sigma)
      self.particles[1, i] = np.random.normal(y_n, sigma)

    self.mean = np.mean(self.particles)

  def getParticles(self):
    return self.particles

  def getWeights(self):
    return self.weights

  def getMean(self):
    return self.state

  def setWeights(self, weights):
    self.weights = weights

  def updateWeights(self, observation, sigma=0.01):
    """
    Update weigth from the detection of a neighbor. Detection is assumed to be accurate.
    """

    total_weight = 0.0
    for i in range(self.n):
      p = self.particles[:, i]
      obs = observation[:2]
      likelihood = math.exp(-0.5 * (math.pow(obs[0]-p[0], 2) / sigma**2 + (math.pow(obs[1]-p[1], 2)) / sigma**2))
      self.weights[i] = likelihood
      total_weight += likelihood

    # Normalize weights
    self.weigths = self.weights / total_weight

  def resample(self):

    normalized_weights = self.weights / np.sum(self.weights)
    nans = normalized_weights[np.isnan(normalized_weights)]
    if len(nans) > 0:
      print("Nan Values detected")
      print(f"Weights: {normalized_weights}")





    # Resampling step: randomly select particles with replacement based on their weights
    indices = np.random.choice(self.n, size=self.n, replace=True, p=normalized_weights)

    # Create a new set of particles based on the selected indices
    resampled_particles = self.particles[:, indices]
    resampled_weights = normalized_weights[indices]

    self.particles = resampled_particles
    self.weights = resampled_weights

    self.weights = self.weights / np.sum(self.weights)

    # print("Sum weights: {}")