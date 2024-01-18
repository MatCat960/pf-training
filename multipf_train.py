import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
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
files = (path / 'logs/pf_dataset_with_obs/').glob('**/*')
checkpoint_log = (path / 'checkpoint.txt')

with open(str(checkpoint_log), 'w') as f:
  f.writelines("Ciao\n")

# len(files)

ROBOTS_NUM = 20
lookback = 7
AREA_W = 30.0
AREA_H = 30.0
GRID_STEPS = 64
ROBOT_RANGE = 15.0
ROBOT_FOV = 120.0
GRAPHICS_ON = False

X_tot = torch.zeros((1, 8)).to(device)
y_tot = torch.zeros((1,6)).to(device)

data = []
sizes = []
filecount = 0
for file in files:
  if filecount < 2:
    filecount += 1
    with open(file) as f:
      lines = f.readlines()
      sizes.append(len(lines))

    for l in lines:
      data.append(l)

# print(data[0])

poses = np.zeros([len(data), 8], dtype="float32")

for i in range(len(data)):
  data[i] = data[i].replace('\n', '')
  poses[i] = tuple(map(float, data[i].split(' ')[:-1]))

y = torch.from_numpy(poses).to(device)

X = torch.zeros_like(y)
X[:, :2] = y[:, :2]
X[:, -2:] = y[:, -2:]

# set covariance matrix values
for i in range(1, y.shape[0]):
  X[i, 2:-2] = y[i-1, 2:-2]

# print(X[1])

# for i in range(X.shape[0]):
#   if X[i, -2] != 0.0 and X[i, -1] != 0.0:
#     X[i, -2:] = X[i, :2]

X = X[1:, :]
y = y[1:, :-2]


X_tot = torch.cat((X_tot, X))
y_tot = torch.cat((y_tot, y))

# X.shape, y.shape

X = X_tot[2:, :]
y = y_tot[2:, :]

sx = int(X.shape[0]/ROBOTS_NUM/X.shape[-1]/3)*ROBOTS_NUM*X.shape[-1]*3
sy = int(X.shape[0]/ROBOTS_NUM/y.shape[-1])*ROBOTS_NUM*y.shape[-1]
X = X[:sx, :]
y = y[:sx, :]

X.shape, y.shape

with open(str(checkpoint_log), 'w') as f:
  f.writelines(f"Old train shape: {X.shape}, {y.shape}\n")
# print(f"Old train shape: {X.shape}, {y.shape}")
X = X.view((-1, ROBOTS_NUM, X.shape[-1]))
y = y.view((-1, ROBOTS_NUM, y.shape[-1]))
with open(str(checkpoint_log), 'w') as f:
  f.writelines(f"new train shape: {X.shape}, {y.shape}\n")



"""## Create train and test split"""



train_size = int(0.8 * X.shape[0])
test_size = X.shape[0] - train_size

X, y = X.to(device), y.to(device)

# Transpose the tensors to have the channel dimensions first
# X = Zf.permute(2, 0, 1).unsqueeze(1)
# Xf = Zf.permute(2, 0, 1).unsqueeze(1)
# y = Zreal.permute(2, 0, 1).unsqueeze(1)

# ROBOTS_NUM = 20
# X = X.view(-1, ROBOTS_NUM, X.shape[-1])
# y = y.view(-1, ROBOTS_NUM, y.shape[-1])
# print(f"New shapes: {X.shape}, {y.shape}")



X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                   test_size=0.2)

y_train = y_train.squeeze(1)
y_test = y_test.squeeze(1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

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

# from shapely import Polygon, Point, intersection

# def covarianceLoss(pred, fov_deg, radius):
#   fov = fov_deg * np.pi / 180.0
#   loss = 0.0
#   for output in pred.cpu().detach().numpy():
#     # print(f"Output shape: {output.shape}")
#     ctr = output[:2]
#     cov = output[2:]
#     # print(f"Center: {ctr}")
#     # print(f"Cov shape: {cov.shape}")
#     cov = cov.reshape(2,2)

#     s=4.605
#     eigenvalues, eigenvectors = LA.eigh(cov)
#     # eigenvalues = eigenvalues + epsilon
#     # print(f"Eigenvalues: {eigenvalues}")
#     # if eigenvalues[0] < 0.0 or eigenvalues[1] < 0.0:
#     #   print(f"Cov matrix: {cov}")
#     #   print(f"Eigenvalues: {eigenvalues}")
#     # eigenvalues[eigenvalues < 0.0] = 0.1
#     a = math.sqrt(s*abs(eigenvalues[0]))
#     b = math.sqrt(s*abs(eigenvalues[1]))

#     if (a < b):
#       temp = dc(a)
#       a = dc(b)
#       b = temp

#     # print(f"Major axis: {a}")

#     m = 0
#     l = 1
#     if eigenvalues[1] > eigenvalues[0]:
#       m = 1
#       l = 0

#     theta = math.atan2(eigenvectors[1,m], eigenvectors[0,m])
#     if theta < 0.0:
#       theta += math.pi

#     vx = []; vy = []
#     x = ctr[0]
#     y = ctr[1]
#     pts = []
#     for phi in np.arange(0, 2*np.pi, 0.1):
#       xs = x + a * np.cos(phi) * np.cos(theta) - b * np.sin(phi) * np.sin(theta)
#       ys = y + a * np.cos(phi) * np.sin(theta) + b * np.sin(phi) * np.cos(theta)
#       pts.append(Point(xs, ys))

#     ellipse_poly = Polygon(pts)

#     x1 = np.array([0.0, 0.0, 0.0])
#     # fov = fov_deg * math.pi / 180
#     arc_theta = np.arange(-0.5*fov, 0.5*fov, 0.1*math.pi)
#     th = np.arange(fov/2, 2*math.pi+fov/2, 0.1*math.pi)

#     # FOV
#     xfov = radius * np.cos(arc_theta)
#     xfov = np.append(x1[0], xfov)
#     xfov = np.append(xfov, x1[0])
#     yfov = radius * np.sin(arc_theta)
#     yfov = np.append(x1[1], yfov)
#     yfov = np.append(yfov, x1[1])

#     fov_pts = []
#     for i in range(xfov.shape[0]):
#       fov_pts.append(Point(xfov[i], yfov[i]))
    
#     fov_poly = Polygon(fov_pts)
#     # print(f"FoV area: {fov_poly.area}")

#     inters = intersection(ellipse_poly, fov_poly)
#     # print(f"Intersecting area: {inters.area}")
#     loss += inters.area

#   # ax.plot(xfov, yfov)
#   return inters.area

"""## Training"""

model = MultiPFModelWithObs(8*ROBOTS_NUM, 6*ROBOTS_NUM, 256)
model = model.to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

RUN_BATCHED = False


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
  alpha = 0.999              # weight of MSE loss vs Cov loss

  for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    if epoch == 0:
      print(f"output shape: {y_pred.shape}")
      print(f"Target shape: {y_train.shape}")
    loss = loss_fn(y_pred[:, 2:], y_train[:, 2:])
    # cov_loss = covarianceLoss(y_pred, ROBOT_FOV, ROBOT_RANGE)
    # total_loss = alpha*loss + (1-alpha)*cov_loss 
    torch.autograd.set_detect_anomaly(True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ### Testing
    model.eval()
    with torch.no_grad():
      test_pred = model(X_test)
      test_loss = loss_fn(test_pred[:, 2:], y_test[:, 2:])
      test_reg = torch.mean(torch.abs(test_pred[2:]))
      test_loss = test_loss + test_reg

    if epoch % 100 == 0:
      print(f"Epoch: {epoch} | Loss: {loss.item()}") # | Test loss: {test_loss.item()}")
      with open(str(checkpoint_log), 'a') as f:
        f.writelines(f"Epoch: {epoch} | Loss: {loss.item()} | Test Loss: {test_loss.item()}\n")

# for name, param in model.named_parameters():
#   if param.requires_grad:
#     print(name, param)

# SAVE TRAINED MODEL
dir_path = os.getcwd()
dir_path = os.path.join(dir_path, "models")
SAVE_MODEL_PATH = os.path.join(dir_path, "pf-model-with-obs.pth")
torch.save(model.state_dict(), SAVE_MODEL_PATH)