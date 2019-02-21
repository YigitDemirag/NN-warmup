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
        ax1 = fig.add_subplot(111)
        fig.canvas.draw()
        x = np.linspace(1, 1000, num=1000)
        h1, = ax1.plot(x, lw=3)
        text = plt.text(60, 1.1, "Reward:")
        ax1.set_ylabel('Value network')
        ax1.set_xlabel('Interactions')
        axbackground1 = fig.canvas.copy_from_bbox(ax1.bbox)
        vlist = np.empty(1000) * np.nan

    while n < num_episodes:
        if render:
            env.render()

        a = get_action(torch.Tensor(o.reshape(1,-1)))[0]
        
        if nn_visual:
            v = get_value(torch.Tensor(o.reshape(1,-1)))[0]
            if cnt > 0:
                ax1.set_ylim([int(np.nanmin(vlist))-1, int(np.nanmax(vlist))+1])
                ax1.set_xlim([cnt-100, cnt+100])
            vlist[cnt] = v.item()
            cnt += 1
            h1.set_ydata(vlist)
            text.set_x(cnt+60)
            text.set_text("Reward: "+ str(round(r,2)))
            fig.canvas.restore_region(axbackground1)
            ax1.draw_artist(h1)
            fig.canvas.blit(ax1.bbox)
            plt.pause(0.000000000001)

        o, r, d, _ = env.step(a.data.numpy()[0])
        
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len, cnt, vlist = env.reset(), 0, False, 0, 0, 0, np.empty(1000)*np.nan
            n += 1
    
    #vlist, cnt = np.empty(1000)*np.nan, 0
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

