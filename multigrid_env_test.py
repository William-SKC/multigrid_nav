import argparse
import json
import os
import random
import ray
import torch
import gymnasium as gym
import imageio
import multigrid.envs # WHY???

if __name__ == "__main__":
    # build the env
    num_agents = 2 
    env = gym.make("MultiGrid-Empty-8x8-v0", agents=num_agents, render_mode="rgb_array")

    # Inspect environment properties
    print(f"Environment class: {env.__class__}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Reset the environment to start
    observations, _ = env.reset()
    # Interact with the environment
    total_reward = [0] * num_agents  # Track cumulative reward for each agent
    terminations = {i: False for i in range(num_agents)}
    truncations = {i: False for i in range(num_agents)}
    frames = []

    while not (all(terminations.values()) or all(truncations.values())):  # Continue until all agents are done
        # Sample random actions for all agents
        actions = env.action_space.sample()
        # print('actions:', actions)

        # Take a step in the environment
        observations, rewards, terminations, truncations, info = env.step(actions)
        # Update total rewards for agents
        for i in range(num_agents):
            total_reward[i] += rewards[i]
        
        # Render the environment
        # env.render(mode='human')
        
        frame = env.render()
        frames.append(frame)

    # Print final results
    print(f"Total rewards for agents: {total_reward}")
    imageio.mimsave("multigrid_test.gif", frames, fps=10)
    print("Animation saved as multigrid_test.gif")
    # Close the environment
    env.close()