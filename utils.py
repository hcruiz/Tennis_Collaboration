'''Contains helper functions for MAPPGD to solve the collaboration Tennis project.
'''
import numpy as np
import torch
from collections import deque
import random

class OUprocess:
    '''Addapted from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
    Here, the OU process is centered at the origin so no mu parameter required
    '''
    def __init__(self,num_agents, action_dimension, theta=0.15, sigma=1.0, seed = 2158257210):
        self.num_agents = num_agents
        self.action_dimension = action_dimension
        self.theta = theta
        self.sigma = sigma
        
        np.random.seed(seed)
        print('numpy seed in ou-process set to :',seed)
        self.reset()

    def reset(self):
        self.state = np.zeros(self.action_dimension)

    def noise(self):
        dx = - self.theta * self.state + self.sigma * np.random.randn(self.num_agents,self.action_dimension)
        self.state = self.state + dx
        return self.state     
    
    
class Reply_buffer:
    '''The replay buffer saves the experiences of both agents in a single 'combined' experience tuple with the method store(samples) 
    and retrieves a mini-batch of batch_size using the method get_batch(batch_size).
    It is initialized with buffer_size.
    '''
    def __init__(self, buffer_size, seed=33231):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        random.seed(seed)
        print('Random seed in buffer is: ', seed)
        
    def store(self, samples):
#         for sars in list(zip(*samples)):
#             self.buffer.append(sars)
        self.buffer.append(samples)
        
    def get_batch(self,batch_size):
        '''get list of (s,a,r,s') tuples of length batch_size
        '''
        batch = random.sample(self.buffer,batch_size)
        return batch
    
