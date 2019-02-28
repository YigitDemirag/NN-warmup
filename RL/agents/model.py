import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

from RL.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.size = size
        self.o_buff = np.zeros((size, *obs_dim), dtype=np.float32)
        self.a_buff = np.zeros((size, *act_dim), dtype=np.float32)
        self.logp_buff = np.zeros((size), dtype=np.float32) 
        self.rew_buff = np.zeros((size), dtype=np.float32)
        self.rew2g_buff = np.zeros((size), dtype=np.float32)
        self.val_buff = np.zeros((size), dtype=np.float32)
        self.adv_buff = np.zeros((size), dtype=np.float32)
        self.start_ptr = 0
        self.gamma = 0.99
        self.lam = 0.97
        self.ptr = 0

    def read(self, on_policy=True):
        """
        Read from the replay buffer in any random time..
        """
        #self.gae_lamb_adv()
        if on_policy: # Returns the experience gathered with the latest policy.
            sl = slice(self.start_ptr, self.ptr)
            tmp = [self.o_buff[sl], self.logp_buff[sl], self.a_buff[sl], 
                    self.rew_buff[sl], self.rew2g_buff[sl], self.val_buff[sl], self.adv_buff[sl]]     
        else: # Returns the all experience buffer.
            tmp = [self.o_buff, self.logp_buff, self.a_buff, 
                    self.rew_buff, self.rew2g_buff, self.val_buff, self.adv_buff]

        self.start_ptr = self.ptr
        return [torch.Tensor(x) for x in tmp]
   
    def write(self, obs, logp, act, rew, val):
        """
        Write to replay buffer after every interaction with env.
        """
        if self.ptr == self.size:
            self.start_ptr = 0
            self.ptr = 0 
        
        self.o_buff[self.ptr] = obs
        self.logp_buff[self.ptr] = logp.detach().numpy()
        self.a_buff[self.ptr] = act.detach().numpy()
        self.rew_buff[self.ptr] = rew # The reward agent currently helds, not the one it will get after act in obs.
        self.val_buff[self.ptr] = val.detach().numpy()
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        """
        Return discount on given vector : x0 + gamma*x1 + gamma^2*x2 + gamma^3*x3 ...
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
    def calc_adv(self, last_val=0):
        """
        Calculate GAE-lambda advantage function 
        """
        path_slice = slice(self.start_ptr, self.ptr)
        rews = np.append(self.rew_buff[path_slice], last_val)
        vals = np.append(self.val_buff[path_slice], last_val)
        
        # Calculate GAE-lambda
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buff[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # Calculate rewards to go
        self.rew2g_buff[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        # Normalize A with mu=0, std=1
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buff)
        self.adv_buff = (self.adv_buff-adv_mean)/adv_std

class MLP(nn.Module):
    def __init__(self, x, hidden_sizes, activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()  
        print('Hidden sizes:')
        for i, h in enumerate([x[0]] + list(hidden_sizes[:-1])): # x has shape of (a, ) and hidden_sizes is (a,b,c,..)
            print((h, hidden_sizes[i]))
            self.layers.append(nn.Linear(h, hidden_sizes[i]))
       
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.output_activation(self.layers[-1](x)) if self.output_activation != None else self.layers[-1](x)
        return x

'''
Policies
'''
class mlp_categorical_policy(nn.Module):
    def __init__(self, x, a, hidden_sizes, activation, output_activation, action_space):
        super(mlp_categorical_policy, self).__init__()
        self.act_dim = action_space.n
        self.p_net = MLP(x, list(hidden_sizes)+[self.act_dim], activation, None)
        
    def forward(self, x, a=None):
        logits = self.p_net(x)
        policy = Categorical(logits=logits) # Provides better functions, use instead of torch.multinomial()
        pi = policy.sample() # Action agent takes 
        logp = policy.log_prob(a) if torch.is_tensor(a) else None # log probability of taking action a 
        logp_pi = policy.log_prob(pi) # log probability of taking action a by policy pi
        return pi, logp, logp_pi

class mlp_gaussian_policy(nn.Module):
    def __init__(self, x, a, hidden_sizes, activation, output_activation, action_space):
        super(mlp_gaussian_policy, self).__init__()
        self.act_dim = action_space.shape[0]
        self.p_net = MLP(x, list(hidden_sizes)+[self.act_dim], activation, output_activation)
        self.log_std = nn.Parameter(-0.5*torch.ones(self.act_dim,dtype=torch.float32))

    def forward(self, x, a=None):
        mu = self.p_net(x)
        policy = Normal(mu, self.log_std.exp())

        pi = policy.sample()
        logp = policy.log_prob(a).sum(dim=1)  if torch.is_tensor(a) else None 
        logp_pi = policy.log_prob(pi).sum()
        return pi, logp, logp_pi

"""
MLP Actor-Critic
"""
class mlp_actor_critic(nn.Module):
    def __init__(self, x, a, hidden_sizes=(64,64), activation=torch.tanh, 
        output_activation=None, action_space=None):
        super(mlp_actor_critic, self).__init__()
        
        # Policy Network 
        if isinstance(action_space, Box):
            print('Policy is Gaussian and action_space is Box.')
            self.p_net = mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space)
        elif isinstance(action_space, Discrete):
            print('Policy is Categorical and action_space is Discrete.')
            self.p_net = mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space)

        # Value Network
        self.v_net = MLP(x, list(hidden_sizes)+[1], activation, None) 

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.p_net(x, a)
        v = self.v_net(x)
        return pi, logp, logp_pi, v
