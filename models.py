import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import math

from pathlib import Path
from copy import deepcopy as dc



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

class CoverageModel(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size

    self.fc1 = nn.Linear(input_size, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, output_size)
    self.relu = nn.ReLU()
    self.activation = nn.Sigmoid()

  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.fc3(x)

    return x

class CoverageModel2(nn.Module):
  def __init__(self, input_size, output_size, dev):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size

    self.fc1 = nn.Linear(input_size, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 64)
    self.fc4 = nn.Linear(64, 32)
    self.fc5 = nn.Linear(32, output_size)
    self.relu = nn.ReLU()
    self.activation = nn.Sigmoid()
    # self.activation = nn.Tanh()

    self.device = dev
    

  def forward(self, x):

    in_size = 0
    # print(x.shape)
    for i in range(x.shape[1]):
      # print(row.shape)
      if x[0, i] != 0.0:
        in_size += 1

    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.activation(self.fc3(x))
    x = self.activation(self.fc4(x))
    x = self.fc5(x)

    out = np.zeros((x.shape[0], self.input_size), dtype="float32")
    out = torch.from_numpy(out).to(self.device)
    out[:, :in_size] = x[:, :in_size]

    return out

  
class DropoutCoverageModel(nn.Module):
  def __init__(self, input_size, output_size, dev):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.device = dev

    self.fc1 = nn.Linear(input_size, 128)
    self.dropout1 = nn.Dropout(0.2)
    self.fc2 = nn.Linear(128, 64)
    self.dropout2 = nn.Dropout(0.2)
    self.fc3 = nn.Linear(64, output_size)
  
    self.relu = nn.ReLU()
    self.activation1 = nn.Tanh()
    self.activation2 = nn.Tanh()
    # self.activation = nn.Sigmoid()

  def forward(self, x):
    in_size = 0
    for i in range(x.shape[1]):
      if x[0, i] != 0.0:
        in_size += 1

    x = self.dropout1(self.activation1(self.fc1(x)))
    x = self.dropout2(self.activation2(self.fc2(x)))
    x = self.fc3(x)

    out = np.zeros((x.shape[0], self.input_size), dtype="float32")
    out = torch.from_numpy(out).to(self.device)
    out[:, :in_size] = x[:, :in_size]

    return out
    



