import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv

from pathlib import Path
from copy import deepcopy as dc

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

from google.colab import drive
drive.mount('/content/drive')

# path = Path("/content/drive/MyDrive/Colab Notebooks/PyTorch tutorial/dynamic_coverage_vel3").glob('**/*')
# path = Path("/content/drive/MyDrive/Colab Notebooks/PyTorch tutorial/dynamic_coverage_vel/log_6_robots.txt")
path = Path("/content/drive/MyDrive/Colab Notebooks/PyTorch tutorial/pycov/30").glob('**/*')
files = [x for x in path if x.is_file()]

len(files)

ROBOTS_NUM = 30
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

# # data0 = []
# data = []
# sizes = []
# for file in files:
#   with open(file) as f:
#     lines = f.readlines()
#     sizes.append(len(lines))

#   data0 = []
#   write = False
#   for l in lines:
#     if not write and l[:5] != '100.0' and l[:5] != '99.90':
#       write = True

#     if write:
#       data0.append(l)

#   # print("Original length: {}".format(len(data0)))
#   s = int(len(data0)/ROBOTS_NUM) * ROBOTS_NUM
#   # print(f"Resized length: {s}")
#   for i in range(s):
#     data.append(data0[i])


# print(data[0])

# poses = np.zeros([len(data), 2], dtype="float32")

# for i in range(len(data)):
#   data[i] = data[i].replace('\n', '')
#   poses[i] = tuple(map(float, data[i].split(' ')))

data = []
sizes = []
for file in files:
  with open(file) as f:
    lines = f.readlines()
    sizes.append(len(lines))

  s = int(len(lines)/ROBOTS_NUM) * ROBOTS_NUM
  lines = lines[:s]

  for l in lines:
    data.append(l)

print(data[0])

poses = np.zeros([len(data), 2], dtype="float32")

for i in range(len(data)):
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')))

len(data)/ROBOTS_NUM

"""## Split poses and velocities"""

pos = np.zeros((int(len(data)/2), 2), dtype="float32")
vel = np.zeros_like(pos)

print(poses[0])
print(f"Original length: {len(poses)}")
print(f"Pos length: {len(pos)}")

for i in range(0, len(pos)):
  pos[i] = poses[2*i]
  vel[i] = poses[2*i+1]

pos[pos == 100.0] = 0.0
vel[vel == 99.9] = 0.0

pos[0]

"""## Convert numpy to torch.Tensor"""

X = torch.from_numpy(pos).to(device)
y = torch.from_numpy(vel).to(device)

print(X[:ROBOTS_NUM, :])
X.shape, y.shape



# X, y = X[:-4], y[:-4]
s = int(X.shape[0]/ROBOTS_NUM)

print(f"Original shape: {X.shape}")
X = X[:s*ROBOTS_NUM]
y = y[:s*ROBOTS_NUM]
print(f"Final shape: {X.shape}")

X[:ROBOTS_NUM, :]

X = X.view(-1, ROBOTS_NUM, 2)
Y = y.view(-1, ROBOTS_NUM, 2)

X.shape, Y.shape

import random
rnd = random.randint(0, int(X.shape[0]/ROBOTS_NUM))
print(rnd)
# X[rnd, :]

from torch_geometric.data import Data
dataset = []

for i in range(X.shape[0]):
  x = X[i, :, :]#.view(-1)
  y = Y[i, :, :]#.view(-1)
  edge_index = torch.combinations(torch.arange(ROBOTS_NUM), 2).t().contiguous()
  edge_index = edge_index.to(device)
  data = Data(x=x, edge_index=edge_index, y=y)
  dataset.append(data)

x.shape, y.shape, edge_index.shape

"""## Create train and test split"""

from torch.utils.data import random_split

s = len(dataset)
train_size = int(0.8 * s)
test_size = s - train_size

train_dataset = dataset[:s]
test_dataset = dataset[s:]

# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataset[0]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   Y,
                                                   test_size=0.2)

y_train = y_train.squeeze(1)
y_test = y_test.squeeze(1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

X_train = X_train.view(-1, 2*ROBOTS_NUM)
X_test = X_test.view(-1, 2*ROBOTS_NUM)
y_train = y_train.view(-1, 2*ROBOTS_NUM)
y_test = y_test.view(-1, 2*ROBOTS_NUM)
X_train.shape

from torch_geometric.data import Data

train_data = Data(x=X_train, edge_index=edge_index, y=y_train)
test_data = Data(x=X_test, edge_index=edge_index, y=y_test)
train_data, test_data

from torch_geometric.data import DataLoader

# create TensorDatasets for training and testing sets
# train_dataset = TensorDataset(X_train, y_train)
# test_dataset = TensorDataset(X_test, y_test)

# create DataLoaders
batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for batch in train_loader:
  print(batch)
  break

"""## Define neural network model"""

class GNNCoverageModel(nn.Module):
  def __init__(self, input_size, hidden_channels, num_robots, output_size):
    super(GNNCoverageModel, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.fc = nn.Linear(hidden_channels, output_size*num_robots)
    self.num_robots = num_robots
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()

  def forward(self, data):
    x, edge_id = data.x, data.edge_index

    # print(f"shape: {x.shape, edge_id.shape}")

    # print("SONO QUI 1")
    x = self.relu1(self.conv1(x, edge_id))
    # print(f"Shape after conv 1: {x.shape}")
    x = self.relu2(self.conv2(x, edge_id))
    # print(f"Shape after conv 2: {x.shape}")

    # print("SONO QUI 2")
    x = global_mean_pool(x, data.batch)

    # print("SONO QUI 3")
    # # separate nodes for each robot
    # print(f"Shape after global mean pool: {x.shape}")
    # print(f"x size 1 : {x.size(1)}")
    # x = x.view(-1, self.num_robots, x.size(1))

    # print("SONO QUI 4")

    # predict velocity for each robot
    vel_out = self.fc(x)
    vel_out = vel_out.view(-1, self.num_robots, 2)

    return vel_out

class CoverageModel(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size

    self.fc1 = nn.Linear(input_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, output_size)
    self.relu = nn.ReLU(inplace=True)
    # self.activation = nn.Sigmoid()
    self.activation = nn.Tanh()

  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.fc3(x)

    return x

class CoverageModel2(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size

    self.fc1 = nn.Linear(input_size, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 32)
    self.fc5 = nn.Linear(32, output_size)
    self.relu = nn.ReLU()
    # self.activation = nn.Sigmoid()
    self.activation = nn.Tanh()


  def forward(self, x):

    in_size = 0
    # print(x.shape)
    for i in range(x.shape[1]):
      # print(row.shape)
      if x[0, i] != 0.0:
        in_size += 1

    # print("x:")
    # print(x)
    # print("in_size: {}".format(in_size))

    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    x = self.fc5(x)

    out = np.zeros((x.shape[0], self.input_size), dtype="float32")
    out = torch.from_numpy(out).to(device)
    out[:, :in_size] = x[:, :in_size]

    # out = out.to(device)

    return out

class DropoutCoverageModel(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size

    self.fc1 = nn.Linear(input_size, 128)
    self.dropout1 = nn.Dropout(0.2)
    self.fc2 = nn.Linear(128, 64)
    self.dropout2 = nn.Dropout(0.2)
    self.fc3 = nn.Linear(64, output_size)
    self.relu = nn.ReLU()
    # self.activation = nn.Sigmoid()
    self.activation1 = nn.Tanh()
    self.activation2 = nn.Tanh()


  def forward(self, x):

    in_size = 0
    # print(x.shape)
    for i in range(x.shape[1]):
      # print(row.shape)
      if x[0, i] != 0.0:
        in_size += 1

    # print("x:")
    # print(x)
    # print("in_size: {}".format(in_size))

    x = self.dropout1(self.activation1(self.fc1(x)))
    x = self.dropout2(self.activation2(self.fc2(x)))
    # x = self.activation(self.fc3(x))
    # x = self.activation(self.fc4(x))
    x = self.fc3(x)

    out = np.zeros((x.shape[0], self.input_size), dtype="float32")
    out = torch.from_numpy(out).to(device)
    out[:, :in_size] = x[:, :in_size]

    # out = out.to(device)

    return out

"""## Training"""

model = GNNCoverageModel(2, 64, ROBOTS_NUM, 2)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

TRAIN_BATCHED = False

epochs = 10

if TRAIN_BATCHED:
  for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
      print(batch.x.shape, batch.y.shape)
      output = model(batch)
      loss = loss_fn(output, batch.y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item() * len(batch)

    avg_loss = total_loss / len(train_loader.dataset)

    ### Testing
    model.eval()
    test_loss_tot = 0.0

    with torch.no_grad():
      for batch in test_loader:
        test_output = model(batch)
        test_loss = loss_fn(test_output, batch.y)
        test_loss_tot += test_loss.item() * len(batch)

    avg_test_loss = test_loss_tot / len(test_loader.dataset)

    print(f"Epoch: {epoch} | Loss: {avg_loss} | Test Loss: {avg_test_loss}")



"""## Train on unbatched data"""

epochs = 10
LOG = True

for epoch in range(epochs):
  model.train()
  for data in train_dataset:
    # print(data.x.shape)
    vel_pred = model(data)
    if LOG:
      print(f"output shape: {vel_pred.shape}")
      print(f"Target shape: {data.y.shape}")
      LOG = False
    loss = loss_fn(vel_pred.squeeze(0), data.y)
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  ### Testing
  model.eval()
  with torch.inference_mode():
    for test_data in test_dataset:
      test_pred = model(test_data)
      test_loss = loss_fn(test_pred, test_data.y)

  # if epoch % 100 == 0:
  print(f"Epoch: {epoch} | Loss: {loss.item()}")

vel_pred.shape

# for name, param in model.named_parameters():
#   if param.requires_grad:
#     print(name, param)

"""## Test on simulated robots"""

import random
N_ROBOTS = 12
robots = np.zeros((N_ROBOTS, 2), dtype="float32")
for i in range(N_ROBOTS):
  robots[i, :] = -40.0 + 30.0 * np.random.rand(1, 2)

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

"""## Plot final position"""

for i in range(N_ROBOTS):
  plt.scatter(robots_hist[-1, i, 0].cpu().detach().numpy(), robots_hist[-1, i, 1].cpu().detach().numpy())

plt.plot(0.0, 0.0, '*')