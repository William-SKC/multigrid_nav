import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from minigrid_extractor import MinigridFeaturesExtractor

# Set seed for reproducibility
set_seed(42)


# ✅ Load & Wrap MiniGrid Environment
env = gym.make("MiniGrid-Empty-16x16-v0")  # Change this to other MiniGrid tasks if needed
env = ImgObsWrapper(env)  # Extract only the "image"
env = wrap_env(env)  # Wrap for PyTorch-based SKRL compatibility

#check for GPU

# device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available(): #MacOS devices
#     device = "mps"
else:
    device = "cpu"

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
class Value(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)  # ✅ Inherit to avoid act() error

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


# 🔧 Define PPO models
models = {
    "policy": Policy(env.observation_space, env.action_space, device),
    "value": Value(env.observation_space, env.action_space, device),
}


# 🎛️ Configure PPO Agent
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 0.5
cfg["clip_predicted_values"] = False
cfg["grad_norm_clip"] = 0.5
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["kl_threshold"] = 0
cfg["mixed_precision"] = True

# Save logs & checkpoints
cfg["experiment"]["directory"] = "runs/torch/MiniGrid"
cfg["experiment"]["write_interval"] = 500
cfg["experiment"]["checkpoint_interval"] = 5000

# 🚀 Initialize PPO Agent
agent = PPO(models=models, memory=memory, cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space, device=device)

# 🏋️ Setup Trainer
cfg_trainer = {"timesteps": 200000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# 🎯 Start Training
trainer.train()
