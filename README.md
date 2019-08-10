# Tennis Collaboration
This repository implements the "cooperation project" of the Udacity DRL Nanodegree. It uses self-learning in the Multiagent DDPG approach. 

## The Tennis Environment

In Tennis, two agents must control their rackets to bounce a ball over a net. The rewards are given as follow: 
* +0.1 to the agent that hits the ball over the net
* -0.01 if the agent lets a ball hit the ground or hits the ball out of bounds

With this reward structure, the goal of each agent is to keep the ball in play, so both have to cooperate to increase each their reward.

Although the observation space consists of 8 variables corresponding to the position and velocity of the ball and racket, the environment stacks 3 observations, so the state is 24 dimensional for each agent. Each agent receives its own, local observation. The action space is continuous and 2D, corresponding to movement toward (or away from) the net, and jumping.
In this episodic task, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents) to solve the environment. In other words, during each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores after each episode. We then take the maximum of these 2 scores. Hence, the environment is considered solved, when the running average (over 100 episodes) of those maximum values is at least +0.5.

__Note__: The project environment is similar to, but not identical to the Tennis environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

## Getting Started

First, install Anaconda (python 3) and clone/download this repository (from terminal use the `git clone` command). To install all the required packages needed to run this task you can create an environment using the .yml file in this repository. Just run on your terminal

`conda env create -f environment.yml`

where *environment.yml* is either `drlnd_Win64.yml` or `drlnd_ubuntu18.yml`. This environment is based on the environment provided by Udacity for this project, with the addition of the specific [PyTorch](https://pytorch.org/) version that I required and the Unity environment. To activate the environment run `conda activate drlnd` and verify that the environment is installed correctly using `conda list`.

__NOTE__: I was able to run on both my machines; however, there might be compatibility issues with yours, so make sure you have the proper environment set up.

Finally, you have to use Udacity's Tennis environment. There are different versions depending on your operating system, so please make sure you have the correct version of the environment. The files of the environment must be placed in the repository directory or, if 
placed somewhere else, the initialization of the environment in the notebook must contain the path to the environment.

__NOTE:__ The torch version in this environment assumes Windows 10 and __no CUDA__ installation. If you want to run the neural networks using CUDA, please make sure you install the proper PyTorch version found [here](https://pytorch.org/get-started/locally/) or try the environment `drlnd_ubuntu18.yml` (with CUDA 10.0). 

## Instructions

The code is structured as follows. There are three modules [ACnets.py](https://github.com/hcruiz/Tennis_Collaboration/blob/master/ACNets.py), [Tennis_agent.py](https://github.com/hcruiz/Tennis_Collaboration/blob/master/Tennis_agent.py) and [utils.py](https://github.com/hcruiz/Tennis_Collaboration/blob/master/utils.py) containing all the necessary functions and classes. 
As the name suggests, ACnets contains the Actor and the Critic network classes. The Actor class, called ActorNet, is a deterministic policy that uses a neural network to estimate the correct action given a state. The Critic class, called CriticNet, is a neural network approximating the Q-function of the combined state and action pairs for all agents.
The Tennis_agent file contains a class SelfPlay_Agent implementing a self-playing agent. This class has the necessary methods for training itself and for updating the target networks.
The user should open the jupyter notebook [Main.ipynb](https://github.com/hcruiz/Tennis_Collaboration/blob/master/Main.ipynb) and simply run all cells to start training. 
For details on the implementation and the results, please have a look at the [Report](https://github.com/hcruiz/Tennis_Collaboration/blob/master/Report.md).
