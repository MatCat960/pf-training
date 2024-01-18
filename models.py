import torch
from torch import nn
import numpy as np
import math

from pathlib import Path
from copy import deepcopy as dc

# from torch_geometric.nn import MessagePassing, global_mean_pool, GCNConv

GRID_STEPS = 100

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

class MultiPFModelWithObs(nn.Module):
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

    self.scale_param = nn.Parameter(torch.ones(4))

  def forward(self, x):
    torch.autograd.set_detect_anomaly(True)
    x = x.view((x.shape[0], -1))
    # print(f"Flattened shape: {x.shape}")
    x = self.activation(self.fc1(x))
    x = self.activation(self.fc2(x))
    x = self.fc3(x)

    x = x.view((x.shape[0], 20, 6))
    # print(f"Shape after FC layers: {x.shape}")

    x[:, :, 4] = x[:, :, 3]

    out = x[:, :, 2:] * self.scale_param

    # cov_matrix = x[:, :, 2:]
    # cov_matrix = cov_matrix.view(-1, 2, 2)
    # scaled_cov_matrix = cov_matrix * self.scale_param
    # scaled_cov_matrix = scaled_cov_matrix.view(-1, 4)

    out = torch.cat((x[:, :, :2], out), dim=2)


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

class GridCoverageModel(nn.Module):
  def __init__(self, input_size, output_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size

    # Convolutional layers for the grid
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()

    # Fully connected layers for robots positions
    self.fc1 = nn.Linear(input_size, hidden_size)
    # self.fc2 = nn.Linear(hidden_size, hidden_size*100)

    # Combine
    self.fc_comb1 = nn.Linear(hidden_size+64*100*100, hidden_size)
    self.fc_comb2 = nn.Linear(hidden_size, hidden_size)
    self.fc_comb3 = nn.Linear(hidden_size, output_size)
    # self.deconv1 = nn.Conv2d(in_channels=hidden_size+64*100*100, out_channels=hidden_size, kernel_size=3, padding=1)
    # self.deconv_out = nn.Conv2d(in_channels=hidden_size, out_channels=output_size, kernel_size=3, padding=1)

    self.activation = nn.Sigmoid()
    # self.activation = nn.Tanh()

  def forward(self, robots, grid):
    # Process occ grid
    # print(f"Grid shape: {grid.shape}")
    grid = grid.view(-1, 1, GRID_STEPS, GRID_STEPS)
    # print(f"Grid shape: {grid.shape}")
    conv_output = self.relu(self.conv1(grid))
    # print(f"Conv2d output shape: {conv_output.shape}")
    # grid_features = self.flatten(self.relu(conv_output))

    # process robot position
    pos_features = self.activation(self.fc1(robots))
    # pos_features = self.fc2(pos_features)
    # print(f"Pos features shape: {pos_features.shape}")

    # pos_features = pos_features.unsqueeze(-1).unsqueeze(-1)

    # match size of grid_features (batch_size, 128, GRID_STEPS, GRID_STEPS)
    # pos_features = pos_features.repeat(1, 1, GRID_STEPS, GRID_STEPS)

    x = conv_output.view(conv_output.shape[0], -1)
    y = pos_features.view(pos_features.shape[0], -1)

    # print(f"Grid features shape: {x.shape}")
    # print(f"Pos features shape: {y.shape}")

    # Combine robots and occupancy grid
    comb = torch.cat((x,y), dim=1)
    # print(f"comb shape: {comb.shape}")
    # x = comb.view(comb.shape[0], -1)          # flatten
    # print(f"flattened shape: {x.shape}")
    x = self.relu(self.fc_comb1(comb))
    x = self.relu(self.fc_comb2(x))
    x = self.fc_comb3(x)

    return x

class GridCoverageModel2(nn.Module):
  def __init__(self, input_size, output_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_size = hidden_size

    self.grid_layer = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(),
        # nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, padding=2),
        # nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2),
        # nn.Dropout(),
    )

    self.robot_layer = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.Tanh(),
        nn.Dropout(),
        nn.Linear(64, 64*64),
        nn.Tanh(),
        nn.Dropout(),
    )

    self.combined_layer = nn.Sequential(
        nn.Linear(644096, 64*64*2),
        nn.Tanh(),
        nn.Dropout(),
        # nn.Linear(64*64*2*2, 64*64*2),
        # nn.Tanh(),
        # nn.Dropout(),
        # nn.Linear(64*64*2*2, 64*64*2),
        # nn.Tanh(),
        nn.Linear(64*64*2, 64*64),
        nn.Tanh(),
        nn.Linear(64*64, output_size),
    )

  def forward(self, robots, grid):
    grid = grid.view(-1, 1, GRID_STEPS, GRID_STEPS)

    print(f"grid shape: {grid.shape}")

    x = self.grid_layer(grid)
    print(f"x shape before: {x.shape}")
    # x = x.view(-1, 64*64)
    x = x.view(64, -1)
    y = self.robot_layer(robots)
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    z = torch.cat((x,y), 1)
    print(f"z shape: {z.shape}")
    z = self.combined_layer(z)
    return z

  



