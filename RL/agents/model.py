import numpy as np
import torch
import torch.nn as nn

class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.size = size
        self.o_buff = np.zeros((size, *obs_dim), dtype=np.float32)
        self.a_buff = np.zeros((size, *act_dim), dtype=np.float32)
        self.rew_buff = np.zeros((size), dtype=np.float32)
        self.val_buff = np.zeros((size), dtype=np.float32)
        self.ptr = 0

    def read(self):
        """
        Read from the replay buffer and reset its content.
        """
        self.ptr = 0
        return [self.o_buff, self.a_buff, self.rew_buff, self.val_buff] 

    def write(self, obs, act, rew, val):
        """
        Write to cyclical replay buffer after every interaction.
        """
        if self.size == self.ptr:
            self.ptr = 0 
        
        self.o_buff[self.ptr] = obs
        self.a_buff[self.ptr] = act
        self.rew_buff[self.ptr] = rew # The reward agent currently helds, not the one it will get after act in obs.
        self.val_buff[self.ptr] = val
        self.ptr += 1

class MLP(nn.Module):
    def __init__(self, x, hidden_sizes=(32,32), activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()  
        
        for i, h in enumerate([x[0]] + list(hidden_sizes[:-1])): # x has shape of (a, ) and hidden_sizes is (a,b,c,..)
            #print((h, hidden_sizes[i]))
            self.layers.append(nn.Linear(h, hidden_sizes[i]))
       
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.output_activation(self.layers[-1](x)) if self.output_activation!=None else self.layers[-1](x)
        return x



'''
class ActorCritic(nn.Module,):
    def __init__(self):


    self.policy = 


    self.value = 
'''
