from PIL import Image as PILImage
import math
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import os
import sys

"""## Step 1: We initialize the Experience Replay memory"""

class ReplayBuffer(object):

    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states1, batch_states2, batch_next_states1, batch_next_states2, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
        for i in ind:
            state1, state2, next_state1, next_state2, action, reward, done = self.storage[i]
            batch_states1.append(state1)
            batch_states2.append(np.array(state2, copy=False))
            batch_next_states1.append(next_state1)
            batch_next_states2.append(np.array(next_state2, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states1), np.array(batch_states2), np.array(batch_next_states1), np.array(batch_next_states2), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


"""## Step 2: We build one neural network for the Actor model and one neural network for the Actor target"""

# This helps calculate the final output dim of CNN
# def conv2d_size_out(size, kernel_size=3, stride=2):
#     return (size - (kernel_size - 1) - 1) // stride + 1
# conv2d_size_out(conv2d_size_out(60))


class AC_conv(nn.Module):

    def __init__(self, state_dim=1):
        super(AC_conv, self).__init__()
        self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=3, stride=2) # 16 
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1) # 16
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 9, kernel_size=3, stride=1) # 9 : 15x15
        self.bn3 = nn.BatchNorm2d(9)  # sq of an odd number, because just!
        self.conv4 = nn.Conv2d(9, 1, kernel_size=1) # 1 : 15x15 | combining 9 channels to one

    def forward(self, x):
        # final output is 5x5 which later be flattened to 25
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        return torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=3) # 5x5 


# Actor Models
class Actor(AC_conv):
    def __init__(self, state_dim, action_dim, max_action):
        AC_conv.__init__(self)
        super(Actor, self).__init__()

        linear_input_size = 25+5

        self.layer_1 = nn.Linear(linear_input_size, 30)  # if on road or sand
        self.layer_2 = nn.Linear(30, 50)
        self.layer_3 = nn.Linear(50, action_dim)

        self.max_action = max_action

    def forward(self, x1, x2):
        x1 = AC_conv.forward(self, x1)

        x = torch.cat(((x1.view(x1.size(0), -1)),
                       x2), 1)

        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))


"""## Step 3: We build two neural networks for the two Critic models and two neural networks for the two Critic targets"""


class Critic(AC_conv):
  
  def __init__(self, state_dim, action_dim):
    AC_conv.__init__(self)
    super(Critic, self).__init__()
    # Defining the first Critic neural network

    linear_input_size = 25+5  # add state["orientation"]

    self.layer_1 = nn.Linear(linear_input_size + action_dim, 30)# if on road or sand
    self.layer_2 = nn.Linear(30, 50)
    self.layer_3 = nn.Linear(50, 1)

    # Defining the second Critic neural network

    self.layer_4 = nn.Linear(linear_input_size + action_dim, 30)# if on road or sand
    self.layer_5 = nn.Linear(30, 50)
    self.layer_6 = nn.Linear(50, 1)


  def forward(self, x1, x2, u):
    # Forward-Propagation on the first Critic Neural Network
    x1_1 = AC_conv.forward(self,x1)
    
    xu_1 = torch.cat(((x1_1.view(x1_1.size(0), -1)),
                   x2, u),1)

    x_1 = F.relu(self.layer_1(xu_1))
    x_1 = F.relu(self.layer_2(x_1))
    x_1 = self.layer_3(x_1)

    # Forward-Propagation on the second Critic Neural Network
    x1_2 = AC_conv.forward(self,x1)

    xu_2 = torch.cat(((x1_2.view(x1_1.size(0), -1)),
                   x2, u),1)
    
    x_2 = F.relu(self.layer_4(xu_2))
    x_2 = F.relu(self.layer_5(x_2))
    x_2 = self.layer_6(x_2)
    return x_1, x_2

  def Q1(self, x1, x2, u):
    x1_1 = AC_conv.forward(self,x1)
    
    xu_1 = torch.cat(((x1_1.view(x1_1.size(0), -1)),
                   x2, u),1)

    x_1 = F.relu(self.layer_1(xu_1))
    x_1 = F.relu(self.layer_2(x_1))
    x_1 = self.layer_3(x_1)
    return x_1


"""## Steps 4 to 15: Training Process"""

# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):

    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # for name, p in self.actor.named_parameters():
        #   if "layer" not in name:
        #     p.requires_grad = False
        # self.actor_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()), lr = 0.001245)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 0.0007)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # for name, p in self.critic.named_parameters():
        #   if "layer" not in name:
        #     p.requires_grad = False
        # self.critic_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()), lr = 0.001245)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0007)
        self.max_action = max_action

    def select_action(self, state1, state2):
        state1 = torch.from_numpy(state1).float().permute(2, 0, 1).unsqueeze(0).to(device)
        state2 = torch.Tensor(state2).unsqueeze(0).to(device)
        # print(f'shape of state1: {state1.shape}; state2{state2.shape}')
        return self.actor(state1, state2).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            # batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            batch_states1, batch_states2, batch_next_states1, batch_next_states2, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
                batch_size)
            state1 = torch.from_numpy(batch_states1).float().permute(0, 3, 1, 2).to(device)
            state2 = torch.Tensor(batch_states2).to(device)
            # next_state1 = torch.Tensor(batch_next_states1).to(device)
            next_state1 = torch.from_numpy(batch_next_states1).float().permute(0, 3, 1, 2).to(device)
            next_state2 = torch.Tensor(batch_next_states2).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state1, next_state2)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(
                next_state1, next_state2, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state1, state2, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(
                current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = - self.critic.Q1(state1, state2, self.actor(state1, state2)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

        # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(
            '%s/%s_actor.pth' % (directory, filename), map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(
            '%s/%s_critic.pth' % (directory, filename), map_location=lambda storage, loc: storage))
