import torch
from torch import nn
# import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

from pathlib import Path
from copy import deepcopy as dc

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

path = Path().resolve()
file = path / 'log.txt'

# len(files)

# ROBOTS_NUM = 5
# lookback = 7
# AREA_W = 30.0
# AREA_H = 30.0
# GRID_STEPS = 64
# ROBOT_RANGE = 15.0
# ROBOT_FOV = 120.0
# GRAPHICS_ON = False

"""## Utility Functions"""

def in_fov(robot, target, fov, range):
  fov_rad = fov * math.pi / 180.0
  xr = robot[0]
  yr = robot[1]
  phi = robot[2]
  dx = target[0] - xr
  dy = target[1] - yr
  dist = math.sqrt(dx**2 + dy**2)
  if dist > range:
    return 0

  xrel = dx * math.cos(phi) + dy * math.sin(phi)
  yrel = -dy * math.sin(phi) + dy * math.cos(phi)
  angle = abs(math.atan2(yrel, xrel))
  if (angle <= fov_rad) and (xrel >= 0.0):
    return 1
  else:
    return 0

def occupied(cell, robots, d_thresh=1.0):
  """
  Defines whether a cell is occupied or not.

  Args:
    cell (np.array) : (x,y) coords of the considered cell
    robots (list(np.array)) : [(x1,y1), (x2,y2), ...] list of robots
  """
  occ = False
  for robot in robots:
    d = np.linalg.norm(cell-robot)
    # print("Distance: {}".format(d))
    if (d < d_thresh):
      occ = occ or True

  # print("Occupied") if occ else print("Free")
  return occ

def gauss_pdf(x, y, means, covs):
  """
  Calculate the probability in the cell (x,y)

  Args:
    x (float) : x coord of the considered point
    y (float) : y coord of the considered point
    means (list(np.array)) : list of mean points
    covs (list(np.array)) : list of covariance matrices
  """

  prob = 0.0
  for i in range(len(means)):
    m = means[i]
    cov = covs[i]
    exp = -0.5 * np.sum

# X, Y : meshgrid
def multigauss_pdf(X, Y, means, sigmas):
  # Flatten the meshgrid coordinates
  points = np.column_stack([X.flatten(), Y.flatten()])

  # Number of components in the mixture model
  num_components = len(means)


  # Initialize the probabilities
  probabilities = np.zeros_like(X)

  # Calculate the probability for each component
  for i in range(num_components):
      mean = means[i]
      covariance = sigmas[i]

      # Calculate the multivariate Gaussian probability
      exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
      coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
      component_prob = coefficient * np.exp(exponent)

      # Add the component probability weighted by its weight
      probabilities += component_prob.reshape(X.shape)

  return probabilities

def plot_fov(fov_deg, radius, ax):
  # fig = plt.figure(figsize=(6,6))
  # plt.scatter(neighs[:, 0], neighs[:, 1], marker='*')

  x1 = np.array([0.0, 0.0, 0.0])
  fov = fov_deg * math.pi / 180
  arc_theta = np.arange(-0.5*fov, 0.5*fov, 0.01*math.pi)
  th = np.arange(fov/2, 2*math.pi+fov/2, 0.01*math.pi)

  # FOV
  xfov = radius * np.cos(arc_theta)
  xfov = np.append(x1[0], xfov)
  xfov = np.append(xfov, x1[0])
  yfov = radius * np.sin(arc_theta)
  yfov = np.append(x1[1], yfov)
  yfov = np.append(yfov, x1[1])
  ax.plot(xfov, yfov)

def generate_sequence_tensor(original_tensor, sequence_length=5):
    num_samples, cols = original_tensor.shape
    sequence_tensor = torch.zeros(num_samples, sequence_length, cols)

    # sequence_tensor[:, 0, :, :, :] = original_tensor.roll(0, dims=0)
    for i in range(0, sequence_length):
        sequence_tensor[:, i, :] = original_tensor.roll(-i, dims=0)

    return sequence_tensor

from numpy import linalg as LA

def plot_ellipse(ctr, cov, ax, s=4.605):
  """
  Args:
    ctr (np.array(1, 2)): center of the ellipse
    cov (np.array(2, 2)): covariance matrix
    s (double): confidence interval
  """

  epsilon = 0.01

  eigenvalues, eigenvectors = LA.eigh(cov)
  eigenvalues = eigenvalues + epsilon
  # print(f"Eigenvalues: {eigenvalues}")
  # if eigenvalues[0] < 0.0 or eigenvalues[1] < 0.0:
  #   print(f"Cov matrix: {cov}")
  #   print(f"Eigenvalues: {eigenvalues}")
  # eigenvalues[eigenvalues < 0.0] = 0.1
  a = math.sqrt(s*abs(eigenvalues[0]))
  b = math.sqrt(s*abs(eigenvalues[1]))

  if (a < b):
    temp = dc(a)
    a = dc(b)
    b = temp

  print(f"Major axis: {a}")

  m = 0
  l = 1
  if eigenvalues[1] > eigenvalues[0]:
    m = 1
    l = 0

  theta = math.atan2(eigenvectors[1,m], eigenvectors[0,m])
  if theta < 0.0:
    theta += math.pi

  vx = []; vy = []
  x = ctr[0]
  y = ctr[1]
  for phi in np.arange(0, 2*np.pi, 0.1):
    xs = x + a * np.cos(phi) * np.cos(theta) - b * np.sin(phi) * np.sin(theta)
    ys = y + a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
    vx.append(xs)
    vy.append(ys)

  vx.append(vx[0])
  vy.append(vy[0])

  # fig = plt.figure(figsize=(6,6))
  ax.plot(vx, vy)

def is_pos_definite(A):
  """
  Check if a matrix is positive definite.
  """

  # Ensure matrix is symmetric
  if not np.allclose(A, A.T):
    return False

  # Check if all eigenvalues are positive
  eigenvalues, _ = np.linalg.eigh(A)

  return np.all(eigenvalues > 0)

def cholesky(A):
  """
  Apply the Cholesky decomposition to ensure positive definiteness
  """

  # Check for NaN or inf values in input matrix
  if torch.isnan(A).any() or torch.isinf(A).any():
    raise ValueError("Input matrix contains NaN")

  L = torch.zeros_like(A)
  L[:, 0, 0] = torch.sqrt(A[:, 0, 0])

  # if torch.abs(L[:, 0, 0]) < 1e-10:
  #   raise ValueError("Division by zero.")
  L[:, 1, 0] = A[:, 1, 0] / L[:, 0, 0]
  L[:, 1, 1] = torch.sqrt(A[:, 1, 1] - L[:, 1, 0] ** 2)

  if torch.isnan(L).any() or torch.isinf(L).any():
    raise ValueError("Cholesky matrix contains NaN.")

  return L

def remove_nonpositive(A, b):
  """
  Remove non-positive definite matrices from tensors A and b.
  Checks non-positive matrices of A and removes the corresponding row in b too.
  Then the other way
  """

  # remove non-positive definite elements of A and corresponding rows in b
  counter = 0
  Ac = dc(A)
  Bc = dc(b)
  print(f"Shape of A before: {Ac.shape}")
  for i in range(Ac.shape[0]):
    M = A[i, 2:]
    M = M.view(2,2)
    m = M.cpu().numpy()
    pos = is_pos_definite(m)
    if not pos:
      Ac = torch.cat((Ac[:i-counter], Ac[i-counter+1:]))
      Bc = torch.cat((Bc[:i-counter], Bc[i-counter+1:]))
      counter += 1

  print(f"Shapes after checking A: {Ac.shape}, {Bc.shape}")

  # do the same for b
  print(f"Shape of b before: {Bc.shape}")
  counter = 0
  Bcc = dc(Bc)
  for i in range(Bc.shape[0]):
    M = Bc[i, 2:]
    M = M.view(2, 2)
    m = M.cpu().numpy()
    pos = is_pos_definite(m)
    if not pos:
      Ac = torch.cat((Ac[:i-counter], Ac[i-counter+1:]))
      Bcc = torch.cat((Bcc[:i-counter], Bcc[i-counter+1:]))
      counter += 1

  print(f"Final shape of A: {Ac.shape}")
  print(f"Final shape of b: {Bcc.shape}")




  count_final = 0
  for i in range(Ac.shape[0]):
    M = Ac[i, 2:]
    M = M.view(2, 2)
    m = M.cpu().numpy()
    pos = is_pos_definite(m)
    if not pos:
      count_final += 1

  print("Number of non positive definite matrices left in A: {}".format(count_final))

  count_final = 0
  for i in range(Bcc.shape[0]):
    M = Bcc[i, 2:]
    M = M.view(2, 2)
    m = M.cpu().numpy()
    pos = is_pos_definite(m)
    if not pos:
      count_final += 1

  print("Number of non positive definite matrices left in b: {}".format(count_final))

  return Ac, Bcc
