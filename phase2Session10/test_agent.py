"""
Trained model weight files from colab used here for inference.
"""

# import libraries
import pygame
import gym_dabbewala
from gym import wrappers
import gym
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

import ai

"""## We make a function that evaluates the policy by calculating its average reward over 10 episodes"""


def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        # print(f'pickup{env.x1, env.y1}; drop{env.x2,env.y2}')
        done = False
        while not done:
            action = policy.select_action(obs['surround'], obs['orientation'])
            obs, reward, done, _ = env.step(action)
            env.render()

            avg_reward += reward
    avg_reward /= eval_episodes
    print("---------------------------------------")
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "DabbeWala-v0"
    # seed = 0
    eval_episodes = 20
    save_env_vid = True
    env = gym.make(env_name)

    # env.seed(seed)
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    max_episode_steps = env._max_episode_steps
    env.reset()
    # env.render(lidar=True)
    env.render()
    file_name = "%s_%s" % ("TD3", env_name)

    state_dim = env.observation_space["surround"].shape[2]  # channel size
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    policy = ai.TD3(state_dim, action_dim, max_action)
    policy.load(file_name, './pytorch_models/')
    _ = evaluate_policy(policy, eval_episodes=eval_episodes)
