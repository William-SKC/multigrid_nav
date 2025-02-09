import gymnasium as gym
import torch
import imageio
import numpy as np

from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model





# Define Policy (Must match training architecture)
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.num_actions)
        )
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return 2 * torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}


# Define Value Model (Used by PPO)
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# Load and wrap environment
env = gym.make("Pendulum-v1", render_mode="rgb_array")
env = wrap_env(env)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Memory (Dummy buffer, required for PPO)
memory = RandomMemory(memory_size=1, num_envs=1, device=device)

# Define PPO models
models = {
    "policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
    "value": Value(env.observation_space, env.action_space, device),
}

# Initialize PPO agent with same configuration as training
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 1024  # Ensure it matches training setup
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 32
cfg["discount_factor"] = 0.9
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["entropy_loss_scale"] = 0.0

agent = PPO(models=models, memory=memory, cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space, device=device)

# Load the trained model checkpoint
agent.load("runs/torch/Pendulum/25-02-06_21-19-37-880880_PPO/checkpoints/best_agent.pt")  # Adjust path if needed
# Run environment and collect frames
frames = []
done = False
max_timesteps = 200


states, infos = env.reset()

for timestep in range(max_timesteps):  
    with torch.no_grad():
        output = agent.act(states, timestep=timestep, timesteps=max_timesteps)
        actions = output[-1].get("mean_actions", output[0])
        next_states, rewards, terminated, truncated, infos = env.step(actions)
        # print(rewards)
        frames.append(env.render())  # Capture frame
        done = terminated or truncated
        if done:
            # obs, _ = env.reset()  # Restart after episode ends
            print(timestep)
            break

env.close()

# Save frames as a GIF
imageio.mimsave("pendulum_PPO.gif", [np.array(frame) for frame in frames], fps=30)