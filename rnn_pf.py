import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math

from pathlib import Path
from copy import deepcopy as dc

# custom imports
from models import *
from train_utils import *

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

path = Path().resolve()
file = path / 'log.txt'

# len(files)

ROBOTS_NUM = 5
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0
GRAPHICS_ON = False

data = []
sizes = []
# for file in files:
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



train_size = int(0.8 * X.shape[0])
test_size = X.shape[0] - train_size

X, y = X.to(device), y.to(device)

# Transpose the tensors to have the channel dimensions first
# X = Zf.permute(2, 0, 1).unsqueeze(1)
# Xf = Zf.permute(2, 0, 1).unsqueeze(1)
# y = Zreal.permute(2, 0, 1).unsqueeze(1)

X.shape, y.shape



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
if GRAPHICS_ON:
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

