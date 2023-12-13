import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
import os

from pathlib import Path
from copy import deepcopy as dc

# custom imports
from models import *
from train_utils import *

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

path = Path().resolve()
logpath = (path / 'logs/dynamic_coverage_vel3/').glob('**/*')
print("Logpath: {}".format(logpath))
files = [x for x in logpath if x.is_file()]

len(files)

ROBOTS_NUM = 20
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0


data = []
sizes = []
for file in files:
  with open(file) as f:
    lines = f.readlines()
    sizes.append(len(lines))

  for l in lines:
    data.append(l)

print(data[0])

poses = np.zeros([len(data), 2], dtype="float32")

for i in range(len(data)):
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')))

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

X = X.view(-1, 2*ROBOTS_NUM)
y = y.view(-1, 2*ROBOTS_NUM)

X.shape, y.shape

"""## Generate sequence tensor"""

Xseq = generate_sequence_tensor(X, lookback)
Xseq.shape

"""## Remove first `lookback` elements"""

Xseq = Xseq[lookback:, :, :]
y = y[lookback:, :]

X = Xseq.to(device)
X.shape, y.shape

import random
rnd = random.randint(0, int(X.shape[0]/ROBOTS_NUM))
print(rnd)
X[rnd, :]

"""## Create train and test split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.2)

y_train = y_train.squeeze(1)
y_test = y_test.squeeze(1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

from torch.utils.data import TensorDataset, DataLoader

# create TensorDatasets for training and testing sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for input, target in train_loader:
  print(input.shape, target.shape)
  break

"""## Create model"""

model = CoverageRNN(input_size = 2 * ROBOTS_NUM,
                    output_size = 2 * ROBOTS_NUM,
                    hidden_size = 128,
                    num_stacked_layers= 4,
                    device=device)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

RUN_BATCHED = False

X_train[0, :]

"""## Train on unbatched data"""

if RUN_BATCHED:
  epochs = 100
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
      outputs = model(inputs)
      if epoch == 0.0:
        print(f"Output shape: {outputs.shape}")
        print(f"Targets shape: {targets.shape}")

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
        test_loss = loss_fn(test_outputs, test_targets)
        running_test_loss += test_loss.item()

      print(f"Epoch: {epoch+1} | Test Loss: {running_test_loss/len(test_loader)}")

if not RUN_BATCHED:
  epochs = 5000
  epsilon = 0.01

  for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    if epoch == 0:
      print(f"output shape: {y_pred.shape}")
      print(f"Target shape: {y_train.shape}")
    loss = loss_fn(y_pred, y_train)
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Testing
    model.eval()
    with torch.inference_mode():
      test_pred = model(X_test)
      test_loss = loss_fn(test_pred, y_test)

    if epoch % 100 == 0:
      print(f"Epoch: {epoch} | Loss: {loss.item()} | Test loss: {test_loss.item()}")

# for name, param in model.named_parameters():
#   if param.requires_grad:
#     print(name, param)

# SAVE TRAINED MODEL
dir_path = os.getcwd()
dir_path = os.path.join(dir_path, "models")
SAVE_MODEL_PATH = os.path.join(dir_path, "rnn_coverage_model.pth")
torch.save(model.state_dict(), SAVE_MODEL_PATH)