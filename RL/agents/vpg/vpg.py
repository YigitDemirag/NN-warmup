import sys
import gym
import time
import numpy as np

from RL.utils.logx import EpochLogger
from RL.utils.mpi_torch import average_gradients, sync_all_params
from RL.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# Hyperparameters
epochs = 100
steps_per_epoch = 1000

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
        Write to replay buffer after every interaction.
        """
        if self.size == self.ptr:
            self.ptr = 0 
        
        self.o_buff[self.ptr] = obs
        self.a_buff[self.ptr] = act
        self.rew_buff[self.ptr] = rew # The reward agent currently helds, not the one it will get after act in obs.
        self.val_buff[self.ptr] = val
        self.ptr += 1

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
