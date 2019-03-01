# Warmup for Neural Networks
**Status:** Active (under active development, breaking changes may occur)

This is my personal minimalistic neural network repository written in [PyTorch](https://pytorch.org/). Currently nothing fancy, nothing complex. Reinforcement learning side of this repo will be more active in the future. 

# DL - Deep Learning 
Following architectures are implemented in [DL](https://github.com/YigitDemirag/NN-warmup/tree/master/DL) package:

- [X] Feed forward network
- [X] Convolutional neural network
- [X] Recurrent neural network


# RL - Reinforcement Learning 
Following architectures are implemented in [RL](https://github.com/YigitDemirag/NN-warmup/tree/master/RL) package:

- [X] Vanilla Policy Gradient (VPG)
- [X] Proximal Policy Optimization (PPO)
- [ ] Deep Q Network (DQN)

# Utilities
- [X] MPI support
- [X] Replay buffer 
- [X] Watch agent's internal value estimation as it interacts with env
- [ ] Prioritized replay buffer
- [ ] Agent comm module 

![Internal Value Representation](data/val.png)
## Installation

NN-Warmup requires Python3, PyTorch, OpenAI Gym and OpenMPI. _Mujoco_ physics engine is optional but can be installed with [mujoco-py](https://github.com/openai/mujoco-py).

### Install Python
Install Python using [Anaconda](https://www.anaconda.com/distribution/#download-section):

```
conda create -n warmup python=3.7
source activate warmup
```

### Install OpenMPI
*Ubuntu*
```
sudo apt update && sudo apt install libopenmpi-dev
```
*Mac OS X*

```
brew install openmpi
```
### Install NN-warmup
```
git clone https://github.com/yigitdemirag/NN-warmup.git
cd NN-warmup
pip install -e .
```
### Check Your Install
To see if you've successfully installed NN-warmup, try running PPO in the OpenAI Gym's `LunarLander-v2` environment with:

```
python -m RL.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name initialtest --epoch 50
```

After it finishes training, watch a video of the trained policy with:
```
python -m RL.run test_policy data/initialtest/initialtest_s0
```

And plot the results with:

```
python -m RL.run plot data/initialtest/initialtest_s0
```

### Disclaimer
* The structure of Reinforcement Learning part is highly inspired from [Spinning Up in Deep RL of OpenAI](https://spinningup.openai.com/).
* MPI support is taken from [firedup Repo](https://github.com/kashif/firedup).
