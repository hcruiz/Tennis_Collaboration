# A self-playing agent for Tennis using the MADDPG approach 

This report describes the implementation details of my approach to solve the Tennis environment, it shows the results obtained of a successful agent and discusses further possible improvements and experiments.

## Learning Algorithm

Before discussing the algorithm, one should notice that we are dealing with a cooperative task in a symmetric environment. These characteristics allow self-learning and the reduction of the multi-agent scenario to a single agent scenario that could be solved via DDPG. Nevertheless, I took a mixed approach where the environment is solved using an adaptation of the MADDPG for self-playing agents. More specific, there is only one agent that plays with itself, but the Q-function treats the state-action pairs of the agent as if there were two distinct actors, one for each field in the tennis court. 

The idea behind using the MADDPG Q-function (instead of the single state-action pair Q-function as in DDPG), was to 'connect' the state-action pairs of the agent on each side of the tennis court to account for the fact that, even if the ball is not yet on your side, you can position yourself to receive it, if your actions acknowledge the events on the opposite side of the court. Thus, our RL-agent is composed of two neural networks, a single actor/policy network for both rackets and a 'critic' network (or Q-function), which takes both state-action pairs to evaluate their Q-value. 

Both networks are trained with the same data--or experiences--gathered during an episode. The experiences are saved in a replay buffer and are sampled in mini-batches to optimize two different functions. The negative Q-function (approximated by the critic network) is the cost of the actor, while the critic network optimises a simple quadratic error (MSE) between the targets and the state-action value. Training is done every certain amount of time steps and several gradient descent steps are made at each training step. For more details on the network architecture and the hyper-parameters, see below.

Also, there are target networks for both the actor and the critic. These are used to construct the target values for the critic MSE loss function and are updated now and then. It is possible to update them in a hard way (copying the current network completely into the target network) or in a soft way (by an exponential rolling average).

Although the actor network is a deterministic policy, the action gets some noise from an Ornstein-Uhlenbeck process to enhance exploration. 

For more details on the MADDPG algorithm, the reader is referred to this [paper](https://arxiv.org/pdf/1706.02275.pdf).

### Neural Network Architectures
Both the actor and the critic neural networks have two hidden layers; the actor with 128 and 64 nodes and the critic with 256 and 64 nodes. However, the critic has an additional batch norm layer **before** the first layer to account for the differences in scale between the state and actions. Both networks have ReLU activation functions. 

While the critic outputs a single linear node approximating the state-value function, the actor outputs a 2-dimensional vector with the elementwise application of a hyperbolic tangent. This output represents moving forward/backward and jumping.

### Parameters

Parameter | nr_episodes | lra | lrc | discount | noise_lvl |update_steps | GD_steps | batch_size | tau | tau update_steps |
|---|---|---|---|---|---|---|---|---|---|---|
Value | 1500 | 1.e-3 | 1.e-3 | 0.999 | 0.2 |5 | 4 | 256 | 1.0 | 2x update_steps | 
Description | max. # episodes | learning rate actor | learning rate critic | reward discount factor | ou noise | time steps between training | nr. grad. desc. epochs | mini-batch | strength of target update (1 == hard) | time steps between target update |

## Results

The environment was solved in 1127 episodes and continued above +0.5 throughout the rest of the trial except for a small number of episodes. Hence, the agent remained stable, albeit learning being slow, see fig. "Mean Rewards". Nevertheless, the maximum score achieved was +0.62. From other values reported, there is room for improvement by avoiding the plateau seen towards the end of the run.

![Mean Rewards](https://github.com/hcruiz/Tennis_Collaboration/blob/master/Scores.png "Mean Rewards")

The plateau in the score profile can be understood by looking at the losses of both the actor and the critic. In both cases, towards the end of the trial (consisting of 1500 episodes) both losses explode. This explosion hindered learning and the performance of the agent would have collapsed eventually if the trial had continued for some more episodes, see fig. "AC losses".

![AC losses](https://github.com/hcruiz/Tennis_Collaboration/blob/master/AC_loss.png "AC losses")

## Future Work
