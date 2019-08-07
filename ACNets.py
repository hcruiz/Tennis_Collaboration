import torch
import torch.nn as nn

class ActorNet(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size1=128, hidden_size2=64, hidden_size3=16):
        
        super(ActorNet, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = (hidden_size1,hidden_size2,hidden_size3)
        
        self.bn = nn.BatchNorm1d(num_features=state_size)
        self.fc_in = nn.Linear(state_size, hidden_size1)
        self.fc1 = nn.Linear(hidden_size1,hidden_size2)
        self.fc2 = nn.Linear(hidden_size2,hidden_size3)
        self.fc_out = nn.Linear(hidden_size3,action_size)
        
        self.activ = nn.functional.relu
        self.tanh = torch.tanh
        
    
    def forward(self,x):
        
        #x = self.bn(x) #is batch norm needed?
        x = self.activ(self.fc_in(x))
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        
        return self.tanh(self.fc_out(x)) # all actions between -1 and 1
    
class CriticNet(nn.Module):
    
    def __init__(self, state_size, action_size, num_agents, hidden_size=256):
        super(CriticNet, self).__init__()
        
        self.in_dim = (state_size + action_size) * num_agents
        self.hidden_size = hidden_size
        self.activ = nn.functional.relu
        self.bn = nn.BatchNorm1d(num_features=self.in_dim)
        self.fc_in = nn.Linear(self.in_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size,int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2),int(hidden_size/4))
        self.fc_out = nn.Linear(int(hidden_size/4),1)
        
    def forward(self,x):
        
        x = self.bn(x)
        x = self.activ(self.fc_in(x))
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        
        return self.fc_out(x)