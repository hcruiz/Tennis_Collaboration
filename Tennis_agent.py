'''
'''
import torch
import torch.nn as nn
from ACNets import ActorNet, CriticNet
from torch.optim import Adam
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device: ",device)
GRADIENT_CLIP = None #10
if GRADIENT_CLIP:
    print('Gradient clipping at ',GRADIENT_CLIP)
else:
    print('NO grad. clipping used.')

class SelfPlay_Agent(nn.Module):
    
    def __init__(self, state_size, action_size, num_agents, discount=0.99, tau=0.01, lr_act=5.e-4, lr_crit=5.e-4, seed=232324):
        
        super(SelfPlay_Agent, self).__init__()
        
        self.num_agents = num_agents
        self.discount = discount
        self.tau = tau
        self.lr_act = lr_act
        self.lr_crit = lr_crit
        
        # Define Actor and Critic Networks
        self.actor = ActorNet(state_size, action_size,seed=seed).to(device)
        self.actor_target = ActorNet(state_size, action_size,seed=seed).to(device)
        
        self.critic = CriticNet(state_size, action_size, num_agents,seed=seed).to(device)
        self.critic_target = CriticNet(state_size, action_size, num_agents,seed=seed).to(device)
            
        # Copy AC networks to target networks
        self.copy2target(self.actor, self.actor_target)
        self.copy2target(self.critic, self.critic_target)
        
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # Initialize Optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr_act)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr_crit)
        # Define loss functions
        self.critic_loss = nn.MSELoss()

    def optim_step(self,loss,optim,network):
        optim.zero_grad()
        loss.backward()
        if GRADIENT_CLIP: 
            nn.utils.clip_grad_norm_(network.parameters(), GRADIENT_CLIP)
        optim.step()
        
    ############### Critic updater ########################
    def update_critic(self, minibatch, agent_i):
        q_inputs = self.get_qinputs(minibatch, agent_i)
        qtarget_inputs = self.get_qtarget_next(minibatch, agent_i)
        gamma_Qnext = self.discount*self.critic_target(qtarget_inputs)
        q_vals = self.critic(q_inputs).view(-1)
#         rewards = torch.tensor([minibatch[i][2][j] for i,j in enumerate(agent_i)], dtype=torch.float, device=device)
#         dones = [torch.tensor(np.asarray(minibatch[i][-1][j])[np.newaxis],dtype=torch.float, device=device) for i,j in enumerate(agent_i)]
        rewards = torch.tensor([minibatch[i][2] for i in range(len(minibatch))], dtype=torch.float, device=device)
        dones = [torch.tensor(np.asarray(minibatch[i][-1])[np.newaxis],dtype=torch.float, device=device) for i in range(len(minibatch))]
        dones = torch.cat(dones,dim=0)   
#         print(rewards.shape,dones.shape,gamma_Qnext.shape)
        y = rewards + gamma_Qnext*(1-dones)
#         print(y.shape)
        loss = (self.critic_loss(q_vals,y[:,0]) + self.critic_loss(q_vals,y[:,1]))/2.0
        self.optim_step(loss, self.critic_optim, self.critic)
        return loss.data.cpu().numpy()
    
    def get_qinputs(self,batch, agent_i):
        state_batch = [torch.tensor(batch[i][0].flatten()[np.newaxis,:],dtype=torch.float,device=device) for i in range(len(batch))]
        actions_batch = [torch.tensor(batch[i][1].flatten()[np.newaxis,:],dtype=torch.float,device=device) for i in range(len(batch))]
#         state_batch = [torch.tensor(batch[i][0][np.newaxis],dtype=torch.float,device=device) for i in range(len(batch))]
#         actions_batch = [torch.tensor(batch[i][1][np.newaxis],dtype=torch.float,device=device) for i in range(len(batch))]
        #========================================#
        state_batch = torch.cat(state_batch,dim=0)
        actions_batch = torch.cat(actions_batch,dim=0)
        return torch.cat([state_batch,actions_batch],dim=1)
    
    def get_qtarget_next(self,batch, agent_i):
        state_batch = [torch.tensor(batch[i][-2][np.newaxis],dtype=torch.float,device=device) for i in range(len(batch))]
        state_batch = torch.cat(state_batch,dim=0)
        actions_batch = [self.actor_target(state_batch[:,0]),self.actor_target(state_batch[:,1])]
        actions_batch = torch.cat(actions_batch,dim=1)
        state_batch = state_batch.view(len(batch),-1)
        target_inputs = torch.cat([state_batch,actions_batch],dim=1)
        return target_inputs
    
    ############### Actor updater ########################
    def update_actor(self,minibatch, agent_i):
        state_batch, actions_batch = self.get_states_actions(minibatch)
        #get actions of minibatch for claculating policy gradient
        actions_batch[torch.arange(len(agent_i)), agent_i] = self.actor(state_batch[torch.arange(len(agent_i)),agent_i])
        state_batch = state_batch.view(len(minibatch),-1)
        actions_batch = actions_batch.view(len(minibatch),-1)
        q_inputs = torch.cat([state_batch,actions_batch],dim=1)
        loss = -self.critic(q_inputs).mean()    
        self.optim_step(loss, self.actor_optim, self.actor)    
        return loss.data.cpu().numpy()
    
    def get_states_actions(self,batch):
        state_batch = [torch.tensor(batch[i][0][np.newaxis],dtype=torch.float,device=device) for i in range(len(batch))]
        state_batch = torch.cat(state_batch,dim=0)
        actions_batch = [torch.tensor(batch[i][1][np.newaxis],dtype=torch.float,device=device) for i in range(len(batch))]
        actions_batch = torch.cat(actions_batch,dim=0)
        return state_batch, actions_batch
    #######################################################
    def target_update(self):
        # Soft update ACTOR
        target_actor_params = self.actor_target.parameters()
        actor_params = self.actor.parameters()
        for target_params, params in zip(target_actor_params,actor_params): 
            target_params.data.copy_(self.tau*params.data+(1.-self.tau)*target_params.data)
        # Soft update CRITIC
        target_critic_params = self.critic_target.parameters()
        critic_params = self.critic.parameters()
        for target_params, params in zip(target_critic_params,critic_params):
            target_params.data.copy_(self.tau*params.data+(1-self.tau)*target_params.data)
    
    def copy2target(self, network, target):
        target_params = target.parameters()
        network_params = network.parameters()
        for t_params, n_params in zip(target_params,network_params): 
            t_params.data.copy_(n_params.data)

    def act(self, state):
        
        x = torch.tensor(state, dtype=torch.float, device=device)
        
        return self.actor(x)
   