import gymnasium as gym
import torch
import torch.nn as nn
import imageio
import numpy as np

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from minigrid_extractor import MinigridFeaturesExtractor

# ✅ Load the trained PPO model from checkpoint
CHECKPOINT_PATH = "runs/torch/MiniGrid/25-02-12_14-37-25-193657_PPO/checkpoints/best_agent.pt"  # Change if needed

# ✅ Load & Wrap MiniGrid Environment
env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array") 
env = ImgObsWrapper(env)  # Extract only the "image"
env = wrap_env(env)  # Wrap for PyTorch-based SKRL compatibility

device = "cuda" if torch.cuda.is_available() else "cpu"


# 🧠 Define Policy Model (with CNN Feature Extractor)
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)

        self.feature_extractor = MinigridFeaturesExtractor(observation_space, features_dim=512)
        
        self.policy_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def compute(self, inputs, role):
        features = self.feature_extractor(inputs["states"])
        logits = self.policy_net(features)
        return logits, {}


# 🧠 Define Value Model (Critic)
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)  # ✅ Inherit to avoid act() error

        self.feature_extractor = MinigridFeaturesExtractor(observation_space, features_dim=512)

        self.value_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        features = self.feature_extractor(inputs["states"])
        return self.value_net(features), {}


# 🔄 Setup Memory Buffer
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
}

# Load agent
agent = PPO(models=models, memory=memory, cfg=PPO_DEFAULT_CONFIG,
            observation_space=env.observation_space,
            action_space=env.action_space, device=device)

agent.load(CHECKPOINT_PATH)  # 🔥 Load trained weights

print(f"✅ Loaded PPO model from {CHECKPOINT_PATH}")

# 🎥 Record environment for GIF
frames = []
obs, _ = env.reset()
done = False
max_timesteps = 200  # Change if you want longer/shorter videos

for timestep in range(max_timesteps):
    output = agent.act(obs,
                            timestep=timestep, timesteps=max_timesteps)
    actions = output[-1].get("mean_actions", output[0])
    # print(output)
    # print('actions:', actions)

    obs, _, terminated, truncated, _ = env.step(actions)
    
    frames.append(env.render())  # Capture frame
    done = terminated or truncated

    if done:
        break  # Stop if agent reaches goal

env.close()

# 🎬 Save frames as a GIF
gif_path = "minigrid_PPO_run.gif"
imageio.mimsave(gif_path, [np.array(frame) for frame in frames], fps=30)

print(f"🎥 GIF saved as {gif_path}!")