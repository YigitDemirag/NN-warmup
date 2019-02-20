import sys
import gym
import time
import numpy as np
import torch 
import torch.nn as nn
import RL.agents.model as model

from RL.utils.logx import EpochLogger
from RL.utils.mpi_torch import average_gradients, sync_all_params
from RL.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from RL.agents.model import ReplayBuffer, mlp_actor_critic

# Hyperparameters
epochs = 100
steps_per_epoch = 1000

def vpg(env_fn, actor_critic=model.mlp_actor_critic, seed=0, logger_kwargs=dict()):
    """
    Vanilla Policy Gradient Algorithm implemented with GAE-lambda advantage function.
    """
    # Loggers
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    # Update the seed
    seed = 1000 * proc_id()
    torch.manual_seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    local_steps_per_epoch = int(steps_per_epoch/num_procs())
    
    actor_critic = mlp_actor_critic(obs_dim, act_dim, action_space=env.action_space)
    rb = ReplayBuffer(local_steps_per_epoch, env.observation_space.shape, env.action_space.shape)
    
    # Optimizer
    p_optimizer = torch.optim.Adam(actor_critic.p_net.parameters(), lr = 3e-4)
    v_optimizer = torch.optim.Adam(actor_critic.v_net.parameters(), lr = 1e-3)
    sync_all_params(actor_critic.parameters())

    def update():
        o, logp, a, _, rew2g, val, adv = rb.read()
        _, logp, _, val = actor_critic(o,a) 
       
        p_loss = -(logp * adv).mean()
        p_optimizer.zero_grad()
        p_loss.backward()
        p_optimizer.step()

        v_loss = (val - rew2g).pow(2).mean()
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

    # Initializations
    ep_ret, ep_len, r, val, done = 0, 0, 0, 0, False
    start_time = time.time()

    # Agent in the Wild 
    for epoch in range(epochs):
        obs = env.reset()
        for t in range(local_steps_per_epoch):
            #env.render()
            a, _, logp, v = actor_critic(torch.FloatTensor(obs))
            rb.write(obs, logp, a, r, v)
            obs, r, done, _ = env.step(a.detach().numpy())
            
            ep_ret += r
            ep_len += 1
            
            # Terminal state
            if done or t==local_steps_per_epoch:
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, r, done, ep_ret, ep_len = env.reset(), 0,  False, 0, 0


        update()
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
    parser.add_argument('--hid', type=int, default=64)
    args = parser.parse_args()
    logger_kwargs = setup_logger_kwargs('vpg', 0)
    mpi_fork(args.cpu)
    
    vpg(lambda: gym.make(args.env), actor_critic=model.mlp_actor_critic, seed=0, logger_kwargs=logger_kwargs)
