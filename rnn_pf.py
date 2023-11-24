# -*- coding: utf-8 -*-
"""10b_pf-rnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uJjIJSOTEYxt7TpOmk2SmwXrA-x2Jxyo

# Training a Neural Network to perform distributed Coverage
"""

import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

from pathlib import Path
from copy import deepcopy as dc

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

from google.colab import drive
drive.mount('/content/drive')

path = Path("/content/drive/MyDrive/Colab Notebooks/PyTorch tutorial/pf_dataset_py").glob('**/*')
files = [x for x in path if x.is_file()]

len(files)

ROBOTS_NUM = 5
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0

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

data = []
sizes = []
for file in files:
  with open(file) as f:
    lines = f.readlines()
    sizes.append(len(lines))

  for l in lines:
    data.append(l)

print(data[0])

poses = np.zeros([len(data), 6], dtype="float32")

for i in range(len(data)):
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')[:-1]))

print("Sizes: {}".format(sizes))
poses.shape

"""## Remove elements with null covariance matrix"""

# poses_new = np.zeros((6))
# print(poses[0].shape)
# for i in range(len(poses)):
#   if poses[i, 3] != 0.0:
#     poses_new = np.append(poses_new, poses[i, :], axis=0)

# poses_new.shape

"""## Convert numpy to torch.Tensor"""

y = torch.from_numpy(poses).to(device)

y.shape

"""## Generate input tensors

- Input tensor  : $X = [x_t, Σ_{t-1}]$
- Output tensor : $y = [x_t, Σ_t]$
"""

X = torch.zeros_like(y)
X[:, :2] = y[:, :2]

for i in range(1, y.shape[0]):
  X[i, 2:] = y[i-1, 2:]

X = X[1:, :]
y = y[1:, :]

X.shape, y.shape

"""## Create train and test split"""

from torch.utils.data import TensorDataset, DataLoader

train_size = int(0.8 * X.shape[0])
test_size = X.shape[0] - train_size

X, y = X.to(device), y.to(device)

# Transpose the tensors to have the channel dimensions first
# X = Zf.permute(2, 0, 1).unsqueeze(1)
# Xf = Zf.permute(2, 0, 1).unsqueeze(1)
# y = Zreal.permute(2, 0, 1).unsqueeze(1)

X.shape, y.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.2)

y_train = y_train.squeeze(1)
y_test = y_test.squeeze(1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# create TensorDatasets for training and testing sets
train_dataset = TensorDataset(X[:train_size], y[:train_size])
test_dataset = TensorDataset(X[train_size:], y[train_size:])

# create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for input, target in train_loader:
  print(input.shape, target.shape)
  break

"""## Define neural network model"""

class myCNN(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size

    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.fc3(x)

    x[:, 4] = x[:, 3]

    # split output in position and cov matrix
    # pos_output = x[:, :2]
    # cov_output = x[:, 2:].view(-1, 2, 2)
    # # print(f"Position output: {pos_output.shape}")
    # # print(f"Covariance output: {cov_output}")
    # # apply cholesky decomposition
    # cholesky_matrix = cholesky(cov_output)
    # # print(f"Cholesky matrix: {cholesky_matrix}")
    # cholesky_matrix = cholesky_matrix.view(-1, 4)
    # # print(f"Cholesky matrix shape: {cholesky_matrix.shape}")

    # # concatenate position with decomposed covariance matrix
    # x = torch.cat((pos_output, cholesky_matrix), dim=1)

    return x

class myCNN2(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size

    self.fc1 = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32)
    self.fc4 = nn.Linear(32, output_size)
    self.relu = nn.ReLU()
    self.activation = nn.Sigmoid()
    # self.activation = nn.Tanh()

    self.scale_param = nn.Parameter(torch.ones(1))

  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.fc4(x)

    x[:, 4] = x[:, 3]

    cov_matrix = x[:, 2:]
    cov_matrix = cov_matrix.view(-1, 2, 2)
    scaled_cov_matrix = cov_matrix * self.scale_param
    scaled_cov_matrix = scaled_cov_matrix.view(-1, 4)

    out = torch.cat((x[:, :2], scaled_cov_matrix), dim=1)

    return out

class LSTM_decoder(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers):
    super().__init__()
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers
    self.lstm = nn.LSTM(input_size, hidden_size**2, num_stacked_layers, batch_first=True)
    self.relu = nn.ReLU()

    self.encoder1 = nn.Sequential(
      nn.Conv2d(1, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, padding=1),
      nn.ReLU(),
    )

    self.encoder2 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
    )

    self.fc = nn.Linear(524288, 8192)
    self.fc_out = nn.Linear(8192, GRID_STEPS**2)

  def forward(self, x, y):
    """
    Args:
      x (torch.Tensor): (batch_size, seq_len, channels, height, width) tensor sequence
      y (torch.Tensor): (batch_size, channels, height, width) partially known image
    """
    batch_size, seq_len, channels, height, width = x.size()
    x = x.view(batch_size * seq_len, channels, height, width)
    x = x.view(batch_size, seq_len, -1)

    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size * self.hidden_size).to(device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size * self.hidden_size).to(device)
    lstm_out, _ = self.lstm(x, (h0, c0))
    # out, _ = self.lstm(x)
    lstm_out = self.relu(lstm_out[:, 0, :])

    lstm_out = lstm_out.view(-1, GRID_STEPS, GRID_STEPS)
    # print(f"Output shape 2: {outputs.shape}")
    lstm_out = lstm_out.unsqueeze(1)

    print(f"Lstm output shape: {lstm_out.shape}")

    x1 = self.encoder1(lstm_out)
    x1 = x1.view(x1.size(0), -1)   # Flatten

    print(f"partially known image shape: {y.shape}")
    x2 = self.encoder2(y)
    x2 = x2.view(x2.size(0), -1)    # Flatten

    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")
    out = torch.cat((x1, x2), dim=1)

    print(f"merged tensor shape: {out.shape}")

    # fc = nn.Linear(out.shape[1], 256)
    out = self.relu(self.fc(out))
    print("SONO QUI 1")
    out = self.fc_out(out)
    print("SONO QUI 2")

    return out

"""## Training"""

model = myCNN2(6, 6)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

RUN_BATCHED = False

# model_lstm2 = LSTM_decoder(GRID_STEPS**2, GRID_STEPS, 1)
# model_lstm2 = model_lstm2.to(device)
# loss_fn = nn.MSELoss()
# optimizer = torch.optim.Adam(model_lstm2.parameters(), lr=0.01)

"""## Check if training covariance matrix are positive definite"""

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

print("==== Train split ====")
X_train_chol, y_train_chol = remove_nonpositive(X_train, y_train)

print("==== Test split ====")
X_test_chol, y_test_chol = remove_nonpositive(X_test, y_test)

X_train, y_train = X_train_chol.to(device), y_train_chol.to(device)
X_test, y_test = X_test_chol.to(device), y_test_chol.to(device)

X_train_cov = X_train[:, 2:]
X_test_cov = X_test[:, 2:]
y_train_cov = y_train[:, 2:]
y_test_cov = y_test[:, 2:]

X_train_cov.shape, X_test_cov.shape, y_train_cov.shape, y_test_cov.shape

if RUN_BATCHED:

  epochs = 100

  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
      outputs = model(inputs)
      # outputs = outputs.squeeze(1)
      print(f"Output shape: {outputs.shape}")
      print(f"Targets shape: {targets.shape}")
      outputs = outputs.unsqueeze(1)
      loss = loss_fn(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print(f"Epoch: {epoch+1} | Loss: {running_loss/len(train_loader)}")

    ### Testing
    model.eval()
    with torch.no_grad():
      running_test_loss = 0.0
      for test_inputs, test_targets in test_loader:
        test_outputs = model(test_inputs)
        test_outputs = test_outputs.unsqueeze(1)
        # test_outputs = test_outputs.view(-1, GRID_STEPS, GRID_STEPS)
        # test_outputs = test_outputs.unsqueeze(1)
        test_loss = loss_fn(test_outputs, test_targets)
        running_test_loss += loss.item()

      print(f"Epoch: {epoch+1} | Test Loss: {running_test_loss/len(test_loader)}")

"""## Train on unbatched data"""

if not RUN_BATCHED:
  epochs = 10000
  epsilon = 0.01

  for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    if epoch == 0:
      print(f"output shape: {y_pred.shape}")
      print(f"Target shape: {y_train.shape}")
    loss = loss_fn(y_pred[:, 2:], y_train[:, 2:])
    # print(f"Matrix before cholesky: {y_train[:, 2:]}")
    # cholesky_output = y_pred[:, 2:].view(-1, 2, 2)
    # loss_pos = loss_fn(y_pred[:, :2], y_train[:, :2])
    # chol = torch.matmul(cholesky_output, cholesky_output.transpose(1,2)).view(-1, 4)
    # print(f"Cholesky matrix: {chol}")
    # print(f"Chol shape: {chol.shape}")
    # print(f"y_train cov shape: {y_train[:, 2:].shape}")
    # loss_cholesky = loss_fn(chol, y_train[:, 2:])
    # loss = loss_pos + loss_cholesky
    # print(f"Position loss: {loss_pos.item()} | Covariance loss: {loss_cholesky.item()}")
    # reg_term = torch.mean(torch.abs(y_pred[2:]))
    # loss = loss + reg_term
    # loss_covariance = loss_fn(y_pred[2:], y_train[2:])
    # alpha = 0.75
    # loss = alpha * loss_position + (1-alpha) * loss_covariance
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
      test_pred = model(X_test)
      test_loss = loss_fn(test_pred[:, 2:], y_test[:, 2:])
      # test_reg = torch.mean(torch.abs(test_pred[2:]))
      # test_loss = test_loss + test_reg

    if epoch % 100 == 0:
      print(f"Epoch: {epoch} | Loss: {loss.item()} | Test loss: {test_loss.item()}")

for name, param in model.named_parameters():
  if param.requires_grad:
    print(name, param)

"""## Test on simulated robots"""

import random

rnd = random.randint(0, test_size)
Xtest = X_test[rnd]
ytest = y_test[rnd]

print(Xtest, ytest)

fig, ax = plt.subplots(1, 1, figsize=(10,10))
# for i in range(lookback):
ctr = Xtest[:2]
cov_matrix = Xtest[2:]
cov_matrix = cov_matrix.view(2, 2)
plot_ellipse(ctr.cpu().detach().numpy(), cov_matrix.cpu().detach().numpy(), ax)
plot_fov(ROBOT_FOV, ROBOT_RANGE, ax)
ax.set_xticks([]); ax.set_yticks([])
# axs[i].set_xlim([-10, 10])
# axs[i].set_ylim([-10, 10])
# ax.set_title(f"t-{i+1}")

"""## Forecast next steps

### Robot starting inside the FoV, then moving outside and finally coming back inside.
"""

NUM_STEPS = 5
dt = 0.5
vel = 5.0

xp = []; yp = []
x = Xtest[0]
y = Xtest[1]
xp.append(x.cpu().detach().numpy())
yp.append(y.cpu().detach().numpy())
for _ in range(NUM_STEPS):
  x = x - dt * vel
  y = y - dt * vel
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

fig, ax = plt.subplots(1, 1, figsize=(10,10))

# # Xtest = Xtest.unsqueeze(0)
# y_pred = model(Xtest.unsqueeze(0)).squeeze(0)
# # y_pred = y_pred.squeeze(0)
# print(f"y pred shape: {y_pred.shape}")
# ctr = y_pred[:2]
# cov_matrix = y_pred[2:]
# cov_matrix[2] = cov_matrix[1]
# cov_matrix = cov_matrix.view(2, 2)
# print(cov_matrix)
# plot_ellipse(ctr.cpu().detach().numpy(), cov_matrix.cpu().detach().numpy(), ax)
plot_fov(ROBOT_FOV, ROBOT_RANGE, ax)

plt.plot(xp, yp)

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

"""### Robot starting inside, then moving outside and staying outside"""

NUM_STEPS = 5
dt = 0.5
vel = 5.0

xp = []; yp = []
x = Xtest[0]
y = Xtest[1]
xp.append(x.cpu().detach().numpy())
yp.append(y.cpu().detach().numpy())
for _ in range(NUM_STEPS):
  x = x - dt * vel
  y = y - dt * vel
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

# print(xp)

# fig, ax = plt.subplots(1, 1, figsize=(10,10))

# # Xtest = Xtest.unsqueeze(0)
# y_pred = model(Xtest.unsqueeze(0)).squeeze(0)
# # y_pred = y_pred.squeeze(0)
# print(f"y pred shape: {y_pred.shape}")
# ctr = y_pred[:2]
# cov_matrix = y_pred[2:]
# cov_matrix[2] = cov_matrix[1]
# cov_matrix = cov_matrix.view(2, 2)
# print(cov_matrix)
# plot_ellipse(ctr.cpu().detach().numpy(), cov_matrix.cpu().detach().numpy(), ax)
# plot_fov(ROBOT_FOV, ROBOT_RANGE, ax)

# plt.plot(xp, yp)

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

