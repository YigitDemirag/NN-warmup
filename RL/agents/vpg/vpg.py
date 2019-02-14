import sys
import gym
import time
import numpy as np
import torch 
import torch.nn as nn

from RL.utils.logx import EpochLogger
from RL.utils.mpi_torch import average_gradients, sync_all_params
from RL.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from RL.agents.model import ReplayBuffer, MLP

# Hyperparameters
epochs = 100
steps_per_epoch = 1000

def vpg(env_fn, seed=0, logger_kwargs=dict()):
    """
    Vanilla Policy Gradient Algorithm implemented with GAE-lambda advantage function.
    """
    # Loggers
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    local_steps_per_epoch = int(steps_per_epoch/num_procs())
    env = env_fn()
    rb = ReplayBuffer(local_steps_per_epoch, env.observation_space.shape, env.action_space.shape)
    
    done = False
    ep_ret = ep_len = r = val = 0
    start_time = time.time()

####
# DELETE LATER
    mlp = MLP(env.observation_space.shape, (32,64,4), activation=torch.tanh) 
###
    # Agent in the Wild 
    for epoch in range(epochs):
        s = env.reset()
        for t in range(local_steps_per_epoch):
            env.render()
            a = env.action_space.sample()
            rb.write(s,a,r,val)
            
            s, r, done, _ = env.step(a)

            ep_ret += r
            ep_len += 1
            
            # Terminal state
            if done or t==local_steps_per_epoch:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                s, r, done, ep_ret, ep_len = env.reset(), 0,  False, 0, 0

        # Logger Monitor
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
    env.close()

if __name__ == '__main__':
    import argparse
    from RL.utils.run_utils import setup_logger_kwargs
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLander-v2')
    parser.add_argument('--cpu', type=int, default=4)
    args = parser.parse_args()
    logger_kwargs = setup_logger_kwargs('vpg', 0)
    mpi_fork(args.cpu)
    
    vpg(lambda: gym.make(args.env), seed=0, logger_kwargs=logger_kwargs)
