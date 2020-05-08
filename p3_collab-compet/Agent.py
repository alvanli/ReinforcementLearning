import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ActorCritic import Actor, Critic
from Utils import OUNoise, ReplayBuffer
import random 
import numpy as np


BUFFER_SIZE = int(1e5)  
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3              
LR_ACTOR = 8.5e-5      
LR_CRITIC = 2e-4    
WEIGHT_DECAY = 0. 
# 8e-5 and 1.5e-4 had a high of 0.38 !!


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.epsilon = 1.0
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, random_seed)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, 2)
        
        self.count = 0
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.count += 1
        if len(self.memory) > BATCH_SIZE and self.count%5 == 0:
            self.count = 0
            for _ in range(1):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        action=np.zeros((2, 2))
        with torch.no_grad():
            for i in enumerate(state):
                if random.random() > eps:
                    action_to_take = self.actor_local(i[1]).cpu().data.numpy()
                else:
                    action_to_take = random.choice(np.arange(self.action_size))
                action[i[0], :] = action_to_take
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
#         states, actions, rewards, next_states, dones, prob, indices = experiences
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)     
#         self.epsilon -= 1e-6
#         self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

