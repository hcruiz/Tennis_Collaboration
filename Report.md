# A self-playing agent for Tennis using the MADDPG approach 

This report describes the implementation details of my approach to solve the Tennis environment, it shows the results obtained of a successful agent and discusses further possible improvements and experiments.

## Learning Algorithm

This environment is solved using an Actor-Critic method with . More specific, our RL-agent is composed of two neural networks, the actor/policy network and the critic network. Both are trained with the same data gathered during an episode but optimizing two different functions ![equation](https://latex.codecogs.com/gif.latex?L%5E%7B%5Cepsilon%7D_%7BPPO%7D) and a simple quadratic cost function (MSE) respectively,

![equation](https://latex.codecogs.com/gif.latex?L%5E%7B%5Cepsilon%7D_%7BPPO%7D%20%3D%20%5Cfrac%7B1%7D%7BM%7D%5Csum_%7Bt%2Ci%7Dmin%5Cleft%20%5B%20A%5Ei_t%5Cfrac%7B%5Cpi_%7Bnew%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%7B%5Cpi_%7Bold%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%2Cclip_%7B%5Cepsilon%7D%5Cleft%28%20A%5Ei_t%5Cfrac%7B%5Cpi_%7Bnew%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%7B%5Cpi_%7Bold%7D%28a%5Ei_t%7Cs%5Ei_t%29%7D%20%5Cright%20%29%20%5Cright%20%5D)

![equation](https://latex.codecogs.com/gif.latex?L_%7BMSE%7D%20%3D%20%5Cfrac%7B1%7D%7BMT%7D%5Csum_%7Bt%2Ci%7D%5Cleft%20%5C%7C%20V%28s_t%5Ei%29%20-%20%5Chat%7BR%7D_t%5Ei%20%5Cright%20%5C%7C%5E%7B2%7D)

where  ![equation](https://latex.codecogs.com/gif.latex?M%2C%20%5Cpi%28a%5Ei_t%7Cs%5Ei_t%29%2C%20V%28s_t%5Ei%29%20%2C%20%5Chat%7BR%7D_t%5Ei)  are the number of samples (=timepoints x agents), the policy, value function, and the discounted rewards respectively.

The actor network is a deterministic policy, however, the action gets some noise from an Ornstein-Uhlenbeck process.

For more details on the MADDPG algorithm, the reader is referred to this [paper](https://arxiv.org/pdf/1706.02275.pdf).

### Neural Network Architectures
Both the actor and the critic neural networks have two layers with 256 nodes each. Both have ReLUs as activation functions and hence, the only difference is in their output. While the critic outputs a single linear node giving the state-value function estimation, the actor's output is 4-dimensional representing the torch in both joints. Furthermore, this output is passed through a hyperbolic tangent before it is fed as the mean to a Normal distribution, from which the final action is sampled. 

### Parameters

Parameter | nr_episodes | lra | lrc | discount | noise_lvl |update_steps | GD_steps | batch_size | tau | tau update_steps |
---|---|---|---|---|---|---|---|---|---|---|---|---|
Value | 1500 | 1.e-3 | 1.e-3 | 0.999 | 0.2 |5 | 4 | 256 | 1.0 | 2x update_steps | 
Description | max. # episodes | learning rate actor | learning rate critic | reward discount factor | ou noise | time steps between training | nr. grad. desc. epochs | mini-batch | strength of target update (1 == hard) | time steps between target update |

## Results

, see fig. "Mean Rewards". 

![Mean Rewards]( "Mean Rewards")

The instability can be appreciated when looking at the actor and critic losses, see fig. "AC losses". Here, the 'Episodes' denote the gradient descent epochs made over the entire training. On the left, we see that the policy/actor loss decreases rapidly at the end and seems to be exploding. On the right, we see the same effect in the critic loss, i.e. it explodes towards the end. This makes the agent lose stability, as it can be appreciated from the reward profile, see fig. "Mean Rewards".

![AC losses]( "AC losses")

## Future Work
