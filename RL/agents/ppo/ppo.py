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

def ppo(env_fn, actor_critic=model.mlp_actor_critic, ac_kwargs=dict(), seed=0, steps_per_epoch=4000, ent_coeff=0.0, 
        epochs=50, clip=0.2, p_lr=3e-4, v_lr=1e-3, ppo_epochs=80, target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization implemented with GAE-lambda advantage function.
    """
    # Loggers
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    
    # Update the seed
    seed += 10000*proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    local_steps_per_epoch = int(steps_per_epoch/num_procs())
    env = env_fn()

    actor_critic = mlp_actor_critic(env.observation_space.shape,  action_space=env.action_space, **ac_kwargs)
    rb = ReplayBuffer(local_steps_per_epoch, env.observation_space.shape, env.action_space.shape)

    # Number of parameters
    var_counts =  tuple(sum(p.numel() for p in module.parameters() if p.requires_grad) 
        for module in [actor_critic.p_net, actor_critic.v_net]) 
    logger.log('Number of parameters: \t pi: %d, \t v: %d\n' % var_counts)   
  
    # Optimizers
    p_optimizer = torch.optim.Adam(actor_critic.p_net.parameters(), lr=p_lr)
    v_optimizer = torch.optim.Adam(actor_critic.v_net.parameters(), lr=v_lr)
    sync_all_params(actor_critic.parameters())

    # Initializations
    ep_ret, ep_len, r, val, done = 0, 0, 0, 0, False
    start_time = time.time()
    
    def update():
        actor_critic.train()
        o, logp_old, a, _, rew2g, val, adv = rb.read()
        for i in range(ppo_epochs):
            _, logp, _ = actor_critic.p_net(o,a)
            ratio = (logp - logp_old).exp() # Ratio of policies
            ent = (-logp).mean() # Entropy
            kl = (logp_old - logp).mean()

            kl = mpi_avg(kl.item())
            if kl > 1.5 * target_kl: # Put additional KL-div limit between consequent policies
                logger.log("Early stopping at step %d due to reaching max KL-divergence" %i)
                break
            
            p_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-clip, 1+clip) * adv).mean()
            p_loss = p_loss - ent * ent_coeff
            p_optimizer.zero_grad()
            p_loss.backward()
            average_gradients(p_optimizer.param_groups)
            p_optimizer.step()

            val = actor_critic.v_net(o)
            v_loss = (val - rew2g).pow(2).mean()
            v_optimizer.zero_grad()
            v_loss.backward()
            average_gradients(v_optimizer.param_groups)
            v_optimizer.step()

    # Agent in the Wild 
    for epoch in range(epochs):
        obs = env.reset()
        actor_critic.eval()
        for t in range(local_steps_per_epoch):
            #env.render()
            a, _, logp, v = actor_critic(torch.Tensor(obs.reshape(1,-1)))
            rb.write(obs, logp, a, r, v)
            obs, r, done, _ = env.step(a.detach().numpy()[0])

            ep_ret += r
            ep_len += 1
            
            # Do not lose r,v at terminal states
            if done or t==local_steps_per_epoch-1:
                v_d = r if done else actor_critic.v_net(torch.Tensor(obs.reshape(1,-1))).item()
                rb.calc_adv(v_d)
                if done:
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                obs, r, done, ep_ret, ep_len = env.reset(), 0,  False, 0, 0
        
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, actor_critic, None)

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
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()
    logger_kwargs = setup_logger_kwargs(args.exp_name, "0")
    
    mpi_fork(args.cpu)
   
    ppo(lambda: gym.make(args.env), 
        actor_critic=model.mlp_actor_critic, 
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), 
        seed=0, 
        steps_per_epoch=4000, 
	ent_coeff=0.0,
        epochs=50, 
        clip=0.2, 
        p_lr=3e-4, 
        v_lr=1e-3, 
        ppo_epochs=80, 
        target_kl=0.01, 
        logger_kwargs=dict(), 
        save_freq=10)
