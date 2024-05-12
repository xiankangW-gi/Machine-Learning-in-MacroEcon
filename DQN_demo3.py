import import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rcParams["font.size"] = 15

psi = 1.0
theta = 1.0
n_int = 5
alpha = 0.36  # Capital share in the Cobb-Douglas production function
beta = 0.99  # Discount factor
delta = 0.1 # depreciation of capital
sigma_tfp = 0.04 # std. dev. for tfp process innvoations
rho_tfp = 0.9 # persistence of tfp process
x_int = x_int_norm * sigma_tfp # adjust the integration nodes
num_input = 2
num_hidden1 = 50
num_hidden2 = 50
num_output = 1
learning_rate = 0.001
num_episodes = 20001
n_data_per_epi = 128
z_lb = 0.7
z_ub = 1.3
k_lb = 0.9
k_ub = 12.0
num_periods = 50
n_tracks = 50
n_periods = 3

class DQN(nn.Module):
  def __init__(self, num_input,num_hidden1,num_hidden2,num_output):
    super(DQN,self).__init__()
    self.layer1 = nn.Linear(num_input,num_hidden1)
    self.layer2 = nn.Linear(num_hidden1,num_hidden2)
    self.layer3 = nn.Linear(num_hidden2, num_output)

  def forward(self,x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = torch.sigmoid(self.layer3(x))
    return x
  
model = DQN(num_input,num_hidden1,num_hidden2,num_output)

def get_singleinside(X_tplus1, model):
    # Extract the number of states and the dimensionality of the state
    n_data = X_tplus1.shape[0]  # Number of states is on the axis 0
    dim_state = X_tplus1.shape[1]  # Dimensionality of the state is on axis 1

    # Read out the state
    Z_tplus1 = X_tplus1[:, 0:1]
    K_tplus1 = X_tplus1[:, 1:2]

    # Compute output
    Y_tplus1 = Z_tplus1 * K_tplus1 ** alpha

    # Compute the return
    r_tplus1 = alpha * Z_tplus1 * K_tplus1 ** (alpha - 1.)

    # Use the model to predict the savings rate
    s_tplus1 = model(X_tplus1)

    # Compute consumption
    C_tplus1 = Y_tplus1 - Y_tplus1 * s_tplus1

    # Compute term inside the expectation
    ret = (1. / C_tplus1) * (1. - delta + r_tplus1)

    return ret

def compute_cost_tuple(X, model):
    n_data = X.shape[0] # number of states is on the axis 0
    dim_state = X.shape[1] # dimensionality of the state is on axis 1

    # read out the state
    Z_t = X[:, 0:1]
    K_t = X[:, 1:2]

    # compute output today
    Y_t = Z_t * K_t ** alpha

    # compute return (not really needed)
    r_t = alpha * Z_t * K_t ** (alpha - 1.)

    # use the neural network to predict the savings rate
    s_t = model(X)

    # get the implied capital in the next period
    K_tplus1 = (1. - delta) * K_t + Y_t * s_t

    # get consumption
    C_t = Y_t - Y_t * s_t

    # now we have to compute the expectation
    expectation = torch.zeros((n_data, dim_state))

    # we loop over the integration nodes
    for i in range(n_int):
        # integration weight
        weight_i = w_int[i]

        # innovation to the AR(1)
        innovation_i = x_int[i]

        # construct exogenous shock at t+1
        Z_tplus1 = torch.exp(rho_tfp * torch.log(Z_t) + innovation_i)

        # construct state at t+1
        X_tplus1 = torch.cat([Z_tplus1, K_tplus1], dim=1)

        # compute term inside the expeectation
        inside_i = get_singleinside(X_tplus1, model)

        # add term to the expectaion with the appropriate weight
        expectation += weight_i * inside_i

    # now we have all terms to construct the relative Euler error

    # Define the relative Euler error
    errREE = 1. - 1. / (C_t * beta * expectation)

    # compute the cost, i.e. the mean square error in the equilibrium conditions
    cost = torch.mean(errREE ** 2)

    # we return some more things for plotting
    LHS = 1. / C_t # LHS of Ee
    RHS = beta * expectation # RHS of Ee

    return cost, errREE, C_t, K_tplus1, r_t, LHS, RHS


def compute_cost(X, model):
    n_data = X.shape[0] # number of states is on the axis 0
    dim_state = X.shape[1] # dimensionality of the state is on axis 1

    # read out the state
    Z_t = X[:, 0:1]
    K_t = X[:, 1:2]

    # compute output today
    Y_t = Z_t * K_t ** alpha

    # compute return (not really needed)
    r_t = alpha * Z_t * K_t ** (alpha - 1.)

    # use the neural network to predict the savings rate
    s_t = model(X)

    # get the implied capital in the next period
    K_tplus1 = (1. - delta) * K_t + Y_t * s_t

    # get consumption
    C_t = Y_t - Y_t * s_t

    # now we have to compute the expectation
    expectation = torch.zeros((n_data, dim_state))

    # we loop over the integration nodes
    for i in range(n_int):
        # integration weight
        weight_i = w_int[i]

        # innovation to the AR(1)
        innovation_i = x_int[i]

        # construct exogenous shock at t+1
        Z_tplus1 = torch.exp(rho_tfp * torch.log(Z_t) + innovation_i)

        # construct state at t+1
        X_tplus1 = torch.cat([Z_tplus1, K_tplus1], dim=1)

        # compute term inside the expeectation
        inside_i = get_singleinside(X_tplus1, model)

        # add term to the expectaion with the appropriate weight
        expectation += weight_i * inside_i

    # now we have all terms to construct the relative Euler error

    # Define the relative Euler error
    errREE = 1. - 1. / (C_t * beta * expectation)

    # compute the cost, i.e. the mean square error in the equilibrium conditions
    cost = torch.mean(errREE ** 2)

    # we return some more things for plotting
    LHS = 1. / C_t # LHS of Ee
    RHS = beta * expectation # RHS of Ee

    return cost

def simulate_single_step(X_t, eps_tplus1, model):
    # Read out the state
    Z_t = X_t[:, 0:1]
    K_t = X_t[:, 1:2]
    
    # Get Z_tplus1
    Z_tplus1 = torch.exp(rho_tfp * torch.log(Z_t) + sigma_tfp * eps_tplus1)
    
    # Compute output today
    Y_t = Z_t * K_t ** alpha
    
    # Use the neural network to predict the savings rate
    s_t = model(X_t)
    
    # Get the implied capital in the next period
    K_tplus1 = (1. - delta) * K_t + Y_t * s_t
    
    # Construct the next step
    X_tplus1 = torch.cat([Z_tplus1, K_tplus1], dim=1)
    
    return X_tplus1

# Define the sim_periods function
def sim_periods(X_start, model, num_periods):
    n_tracks, dim_state = X_start.shape
    
    # Create an empty array to store the states
    X_simulation = torch.empty((num_periods, n_tracks, dim_state))
    
    # Draw random innovation
    eps = torch.randn((num_periods, n_tracks), dtype=torch.float32)
    
    # Set starting state
    X_simulation[0, :, :] = X_start
    
    X_old = X_start
    
    # Simulate the periods
    for t in range(1, num_periods):
        eps_use = eps[t, :, None]  # None ensures the shape is n_tracks x 1
        
        X_new = simulate_single_step(X_old, eps_use, model)
        
        X_simulation[t, :, :] = X_new
        
        X_old = X_new
        
    return X_simulation

def get_training_data(z_lb, z_ub, k_lb, k_ub, n_data):
    Z = torch.rand(n_data, 1, dtype=torch.float32) * (z_ub - z_lb) + z_lb
    K = torch.rand(n_data, 1, dtype=torch.float32) * (k_ub - k_lb) + k_lb
    X = torch.cat([Z, K], dim=1)
    return X

def get_training_data_simulation(X_start, model, n_periods):
    n_tracks, n_dim = X_start.shape

    X_simulation = sim_periods(X_start, model, n_periods)
    
    X_end = X_simulation[-1, :, :].to(torch.float32)
    
    X_training = X_simulation.view(n_tracks * n_periods, n_dim).to(torch.float32)
    
    return X_training, X_end

X_start = get_training_data(z_lb, z_ub, k_lb, k_ub, n_tracks)
X_training, X_end = get_training_data_simulation(X_start, model, n_periods)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

def create_dataloader()


train_loss = []
X_start = X_end
for ep in range(num_episodes):
    X,X_end = get_training_data_simulation(X_start, model, n_periods)
    X_start= X_end
    
    loss = compute_cost(X, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss.append(loss.item())



