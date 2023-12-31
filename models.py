import torch
from torch import nn
import numpy as np
import math

from pathlib import Path
from copy import deepcopy as dc

# from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv



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

class PFModel(nn.Module):
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

class PFModelWithObs(nn.Module):
  def __init__(self, input_size, output_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(0.2)
    # self.activation = nn.Tanh()
    self.activation = nn.Sigmoid()

    self.scale_param = nn.Parameter(torch.ones(1))

  def forward(self, x):
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.fc3(x)

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

    self.fc1 = nn.Linear(input_size, 512)
    self.dropout1 = nn.Dropout(0.2)
    self.fc2 = nn.Linear(512, 512)
    self.dropout2 = nn.Dropout(0.2)
    self.fc3 = nn.Linear(512, output_size)
  
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

    out = torch.zeros((x.shape[0], self.input_size)).to(self.device)
    # out = torch.from_numpy(out).to(self.device)
    out[:, :in_size] = x[:, :in_size]

    return out
  
class CoverageRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_stacked_layers, output_size, device):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.num_stacked_layers = num_stacked_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(hidden_size, hidden_size)
    self.dropout1 = nn.Dropout(0.2)
    self.fc2 = nn.Linear(hidden_size, 64)
    self.dropout2 = nn.Dropout(0.2)
    self.fc3 = nn.Linear(64, output_size)
    self.relu = nn.ReLU()
    # self.activation = nn.Sigmoid()
    self.activation1 = nn.Tanh()
    self.activation2 = nn.Tanh()
    self.device = device


  def forward(self, x):
    in_size = 0
    # print(x.shape)
    for i in range(x.shape[2]):
      # print(row.shape)
      if x[0, 0, i] != 0.0:
        in_size += 1

    batch_size, seq_len, _ = x.size()
    h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
    c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

    x, _ = self.lstm(x, (h0, c0))
    # print(f"x Shape: {x[:, -1, :].shape}")
    x = self.dropout1(self.activation1(self.fc1(x[:, -1, :])))
    x = self.dropout2(self.activation2(self.fc2(x)))
    x = self.fc3(x)

    out = np.zeros((x.shape[0], self.input_size), dtype="float32")
    out = torch.from_numpy(out).to(self.device)
    out[:, :in_size] = x[:, :in_size]

    return out
    
class GNNCoverageModel(nn.Module):
  def __init__(self, input_size, hidden_channels, num_robots, output_size, device):
    super(GNNCoverageModel, self).__init__()
    self.conv1 = GCNConv(input_size, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, hidden_channels)
    self.fc = nn.Linear(hidden_channels, output_size*num_robots)
    self.num_robots = num_robots
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.activation = nn.Tanh()
    self.device = device

  def forward(self, data):
    x, edge_id = data.x, data.edge_index
    # print(f"x shape: {x.shape}")
    in_size = 0
    # print(x.shape)
    for i in range(x.shape[0]):
      # print(row.shape)
      if x[i, 0] != 0.0 and x[i, 1] != 0.0:
        in_size += 1

    # print("in_size: {}".format(in_size))
    # print(f"shape: {x.shape, edge_id.shape}")

    # print("SONO QUI 1")
    x = self.activation(self.conv1(x, edge_id))
    # print(f"Shape after conv 1: {x.shape}")
    x = self.activation(self.conv2(x, edge_id))
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

    out = np.zeros((1, self.num_robots, 2), dtype="float32")
    out = torch.from_numpy(out).to(self.device)
    out[:, :in_size, :] = vel_out[:, :in_size, :]

    return out

  



