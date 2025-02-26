import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import imageio


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

# ✅ Load the trained model from checkpoint
CHECKPOINT_PATH = "runs/torch/MultiGrid_MAPPO_CustomReward/25-02-24_17-30-55-111835_MAPPO/checkpoints/best_agent.pt"  


# ✅ Load & Wrap MultiGrid Environment
num_agents = 3
env = gym.make('MultiGrid-Empty-Random-6x6-v0', agents=num_agents, render_mode="rgb_array")
env = wrap_env(env, wrapper="multigrid")  # ✅ Required for PyTorch training

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    memories[agent_name] = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)

# 🎛 Configure IPPO Agent
cfg = MAPPO_DEFAULT_CONFIG.copy()
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": next(iter(env.observation_spaces.values())), "device": device}
cfg["shared_state_preprocessor"] = RunningStandardScaler
cfg["shared_state_preprocessor_kwargs"] = {"size": next(iter(env.state_spaces.values())), "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

trained_agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,  
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        action_spaces=env.action_spaces,
        device=device,
        shared_observation_spaces=env.state_spaces
    )

trained_agent.load(CHECKPOINT_PATH)  # 🔥 Load trained weights
print(f"✅ Loaded MAPPO model from {CHECKPOINT_PATH}")
# ✅ Run the agent in the environment and save frames
frames = []
obs, _ = env.reset()

# Run for a fixed number of steps
max_timesteps = 20000  # Adjust for longer demos
for timestep in range(max_timesteps):
    # Get actions from the trained model
    # for id, ob in obs.items():
    #     print('obs: ', id, ob.shape)
    output = trained_agent.act(obs, timestep=timestep, timesteps=max_timesteps)
    actions = output[-1].get("mean_actions", output[0])

    # Step the environment
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Render and store frames
    frame = env.render()
    frames.append(frame)
    
    # Stop if all agents are done
    if all(terminated.values()) or all(truncated.values()):
        break

# ✅ Save the GIF
gif_path = "multigrid_mappo_demo.gif"
imageio.mimsave(gif_path, frames, fps=10)  # Adjust FPS for speed control

print(f"🎬 GIF saved successfully at {gif_path}!")