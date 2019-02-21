import time
import joblib
import os
import os.path as osp
import numpy as np
from matplotlib import pyplot as plt
import torch
from RL import EpochLogger

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_policy(fpath, itr='last'):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[10:-3]) for x in os.listdir(fpath) if 'torch_save' in x and len(x)>13]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    model = torch.load(osp.join(fpath, 'torch_save'+itr+'.pt'))
    model.eval()

    # get the model's policy and value networks
    get_action = model.p_net
    get_value = model.v_net

    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action, get_value


def run_policy(env, get_action, get_value, max_ep_len=None, num_episodes=100, render=True, nn_visual=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, v, d, ep_ret, ep_len, n, cnt = env.reset(), 0, 0, False, 0, 0, 0, 0
    
    if nn_visual:
        # Prepare canvas
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.canvas.draw()
        x = np.linspace(1, 10000, num=10000)
        h, = ax.plot(x, lw=3)
        text = ax.text(0.8,1.5, "")
        ax.set_ylim([-20,20])
        plt.xlabel('Interactions')
        plt.ylabel('Value network')
        axbackground = fig.canvas.copy_from_bbox(ax.bbox)
        vlist = np.empty(10000) * np.nan
   
    while n < num_episodes:
        if render:
            env.render()

        a = get_action(torch.Tensor(o.reshape(1,-1)))[0]
        
        if nn_visual:
            v = get_value(torch.Tensor(o.reshape(1,-1)))[0]
            if cnt > 0:
                ax.set_ylim([int(np.nanmin(vlist))-5, int(np.nanmax(vlist))+5])
                #ax.set_ylim([-20,20])
                ax.set_xlim([cnt-100, cnt+100])
            vlist[cnt] = v.item()
            cnt += 1
            h.set_ydata(vlist)
            fig.canvas.restore_region(axbackground)
            ax.draw_artist(h)
            plt.pause(0.000000000001)

        o, r, d, _ = env.step(a.data.numpy()[0])
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=2000)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--vis', type=str2bool, default=True)
    args = parser.parse_args()
    env, get_action, get_value = load_policy(args.fpath,
                                  args.itr if args.itr >=0 else 'last')
    run_policy(env, get_action, get_value, args.len, args.episodes, not(args.norender), args.vis)

