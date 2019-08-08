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
    def __init__(self, action_dimension, theta=0.15, sigma=1.0):
        self.action_dimension = action_dimension
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.sigma * np.random.randn(self.action_dimension)

    def noise(self):
        dx = - self.theta * self.state + self.sigma * np.random.randn(len(self.state))
        self.state = self.state + dx
        return self.state     
    
    
class Reply_buffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        
    def store(self, samples):
#         for sars in list(zip(*samples)):
#             self.buffer.append(sars)
        self.buffer.append(samples)
        
    def get_batch(self,batch_size):
        '''get list of (s,a,r,s') tuples of length batch_size
        '''
        batch = random.sample(self.buffer,batch_size)
        return batch
    