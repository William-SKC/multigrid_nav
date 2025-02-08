import gymnasium as gym
import torch
import torch.nn as nn
import imageio
import numpy as np


from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, Model


# ✅ Load the trained PPO model from checkpoint
CHECKPOINT_PATH = "runs/torch/MiniGrid/25-02-07_16-56-39-786902_PPO/checkpoints/best_agent.pt"  # Change if needed

class MinigridFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__()
        
        # Get input shape (H, W, C) from observation space
        n_input_channels = observation_space.shape[2]  # MiniGrid provides (H, W, C)
        self.observation_space = observation_space
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()).float()
            sample_input = sample_input.permute(2, 0, 1).unsqueeze(0)  # Ensure (1, C, H, W)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() == 3:  # If missing batch dimension
            observations = observations.unsqueeze(0)  # Add batch dimension
        
        if observations.dim() == 2:  # If completely flattened, reshape it back
            batch_size = observations.shape[0]
            observations = observations.view(batch_size, *self.observation_space.shape)
        
        observations = observations.permute(0, 3, 1, 2)  # Convert from (B, H, W, C) to (B, C, H, W)
        return self.linear(self.cnn(observations))


# ✅ Load & Wrap MiniGrid Environment
env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")  # Change this to other MiniGrid tasks if needed
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
print(obs)
done = False
timestep = 0
max_timesteps = 200  # Change if you want longer/shorter videos

for _ in range(max_timesteps):
    output = agent.act(obs,
                            timestep=timestep, timesteps=max_timesteps)
    actions = output[-1].get("mean_actions", output[0])

    obs, _, terminated, truncated, _ = env.step(actions)
    
    frames.append(env.render())  # Capture frame
    done = terminated or truncated
    timestep += 1

    if done:
        break  # Stop if agent reaches goal

env.close()

# 🎬 Save frames as a GIF
gif_path = "ppo_minigrid_run.gif"
imageio.mimsave(gif_path, [np.array(frame) for frame in frames], fps=30)

print(f"🎥 GIF saved as {gif_path}!")