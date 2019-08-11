import torch
import torch.nn as nn
import numpy as np

# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)

# def reset_parameters(layer):
#     if type(layer) == nn.Linear:
#         layer.weight.data.uniform_(*hidden_init(layer))

class ActorNet(nn.Module):
    '''Actor network with input size of state_size and two hidden layers hidden_size1, hidden_size2 
    with relu activation. The network has an output of dimension action_size. The output vector is passed through a tanh function elementwise 
    before it is returned.
    '''
    def __init__(self, state_size, action_size, hidden_size1=128, hidden_size2=64, seed = 128):
        
        super(ActorNet, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        print('torch seed in actor set to: ', seed)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = (hidden_size1,hidden_size2,hidden_size3)
        
        self.fc_in = nn.Linear(state_size, hidden_size1)
#         reset_parameters(self.fc_in)
        self.fc1 = nn.Linear(hidden_size1,hidden_size2)
#         reset_parameters(self.fc1)
        self.fc_out = nn.Linear(hidden_size2,action_size)
#         self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        
        self.activ = nn.functional.relu
        self.tanh = torch.tanh
        
    
    def forward(self,x):
        
        x = self.activ(self.fc_in(x))
        x = self.activ(self.fc1(x))
        x = self.fc_out(x)
        return self.tanh(x) # all actions between -1 and 1
    
class CriticNet(nn.Module):
    '''Critic network with input size of (state_size + action_size) * num_agents and two hidden layers hidden_size1, hidden_size2 
    with relu activation. There is a batch norm step before the input layer.
    '''
    def __init__(self, state_size, action_size, num_agents, hidden_size1=256, hidden_size2=64, seed = 128):
        super(CriticNet, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        print('torch seed in critic set to: ', seed)
        
        self.in_dim = (state_size + action_size) * num_agents
        self.hidden_size1 = hidden_size1
        self.activ = nn.functional.relu
        
        self.bn0 = nn.BatchNorm1d(num_features=self.in_dim)
        self.fc_in = nn.Linear(self.in_dim, hidden_size1)
#         reset_parameters(self.fc_in)
#         self.bn1 = nn.BatchNorm1d(num_features=hidden_size1)
        self.fc1 = nn.Linear(hidden_size1,hidden_size2)
#         reset_parameters(self.fc1)
        self.fc_out = nn.Linear(hidden_size2,1)
#         self.fc_out.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self,x):
        
        x = self.bn0(x)
        x = self.activ(self.fc_in(x))
#         x = self.bn1(x)
        x = self.activ(self.fc1(x))
        return self.fc_out(x)
