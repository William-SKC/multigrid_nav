import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from multigrid import envs  # Ensure MultiGrid environments are registered
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# from multigrid_wrapper import MultiGridMAPPOWrapper

# Import CNN Feature Extractor
from minigrid_extractor import MinigridFeaturesExtractor

class MultiGridMAPPOWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def state(self):
        print("🔥 state() was called!")
        return np.zeros(10)  # Dummy return to check if it’s used

# ✅ Set seed for reproducibility
set_seed(42)

# ✅ Load & Wrap MultiGrid Environment
num_agents = 2
env = gym.make('MultiGrid-Empty-8x8-v0', agents=num_agents, render_mode="rgb_array")
env = wrap_env(env, wrapper="multigrid")  # ✅ Required for PyTorch training
env_core = env._unwrapped
device = "cuda" if torch.cuda.is_available() else "cpu"
# ✅ Get observation & action space from `env.unwrapped`

agent_ids = []
for str_element in env_core.agent_dict.keys():
    agent_ids.append(int(str_element))
obs, _ = env.reset()
print(obs[0].shape)
print(env.state().shape)

observation_spaces = {agent_id: env_core.observation_space[agent_id]["image"] for agent_id in agent_ids}
action_spaces = {agent_id: env_core.action_space[agent_id] for agent_id in agent_ids}
# shared_observation_spaces = env.shared_observation_spaces

print("Agents:", agent_ids)
print("Observation Spaces:", observation_spaces)
print("Action Spaces:", action_spaces)
# print("Shared Observation Spaces:", shared_observation_spaces)

# 🔄 Setup Memory Buffer (for each agent)
memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)


# 🧠 Define Policy Model (with CNN Feature Extractor)
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)
        print("Policy Observation Space:", observation_space)
        self.feature_extractor = MinigridFeaturesExtractor(observation_space, features_dim=512)

        self.policy_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

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
print(observation_spaces[0])
models = {}
for agent_id in agent_ids:
    models[agent_id] = {
        "policy": Policy(observation_spaces[agent_id], action_spaces[agent_id], device),
        "value": Value(observation_spaces[agent_id], action_spaces[agent_id], device)
    }

# 🔄 Setup Memory Buffer for Each Agent
memories = {}
for agent_id in agent_ids:
    memories[agent_id] = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

# 🎛 Configure MAPPO Agent
cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
cfg_agent["rollouts"] = 1024
cfg_agent["learning_epochs"] = 10
cfg_agent["mini_batches"] = 32
cfg_agent["discount_factor"] = 0.99
cfg_agent["lambda"] = 0.95
cfg_agent["learning_rate"] = 3e-4
cfg_agent["entropy_loss_scale"] = 0.01
cfg_agent["value_loss_scale"] = 0.5
cfg_agent["clip_predicted_values"] = False
cfg_agent["grad_norm_clip"] = 0.5
cfg_agent["ratio_clip"] = 0.2
cfg_agent["value_clip"] = 0.2
cfg_agent["kl_threshold"] = 0
cfg_agent["mixed_precision"] = True

# ✅ Set up logging & checkpoints
cfg_agent["experiment"]["directory"] = "runs/torch/MultiGrid_MAPPO"
cfg_agent["experiment"]["write_interval"] = 500
cfg_agent["experiment"]["checkpoint_interval"] = 5000

# 🏆 Initialize MAPPO Agent
agent_names = env_core.agent_dict.keys()
agent = MAPPO(
    possible_agents=agent_ids,
    models=models,
    memories=memories,  
    cfg=cfg_agent,
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    device=device,
    shared_observation_spaces = observation_spaces
)

# 🏋️ Configure & Start Training
cfg_trainer = {"timesteps": 200000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

print("🚀 Starting MAPPO Training...")
trainer.train()
print("🎯 Training Complete!")