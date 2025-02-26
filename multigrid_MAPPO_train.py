import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from multigrid import envs  # Ensure MultiGrid environments are registered
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

# Import CNN Feature Extractor
from minigrid_extractor import MinigridFeaturesExtractor  

# ✅ Set seed for reproducibility
set_seed(42)

# ✅ Load & Wrap MultiGrid Environment
num_agents = 3
env = gym.make('MultiGrid-Empty-Random-6x6-v0', agents=num_agents)
env = wrap_env(env, wrapper="multigrid")  # ✅ Required for PyTorch training
env_core = env.unwrapped
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Reset environment and check shapes
obs, _ = env.reset()
print("Observation Shape (Per Agent):", obs[0].shape)

# ✅ Extract correct observation and action spaces (ONLY IMAGE)
print("Agents:", env.possible_agents)
print("Observation Spaces:", env.observation_spaces)
print("Action Spaces:", env.action_spaces)
print("State Spaces:", env.state_spaces)


# 🧠 Define Policy Model (CNN Feature Extractor)
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
        DeterministicMixin.__init__(self)  

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


# 🔧 Define Separate Models for Each Agent
models = {
    agent_name : {
        "policy": Policy(env.observation_space(agent_name), env.action_space(agent_name), device),
        "value": Value(env.state_space(agent_name), env.action_space(agent_name), device)
    }
    for agent_name in env.possible_agents
}

# instantiate memories as rollout buffer (any memory can be used for this)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

# 🎛 Configure IPPO Agent
cfg = MAPPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 8  
cfg["discount_factor"] = 0.95
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": next(iter(env.observation_spaces.values())), "device": device}
cfg["shared_state_preprocessor"] = RunningStandardScaler
cfg["shared_state_preprocessor_kwargs"] = {"size": next(iter(env.state_spaces.values())), "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# ✅ Set up logging & checkpoints
cfg["experiment"]["directory"] = "runs/torch/MultiGrid_MAPPO_CustomReward"
cfg["experiment"]["write_interval"] = 1000
cfg["experiment"]["checkpoint_interval"] = 5000

training_agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,  
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.state_spaces
    )

# 🏋️ Configure & Start Training
cfg_trainer = {"timesteps": 200000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=training_agent)

print("🚀 Starting MAPPO Training...")
trainer.train()
print("🎯 Training Complete!")