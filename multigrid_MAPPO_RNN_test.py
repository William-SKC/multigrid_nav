import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import imageio

# Ensure MultiGrid environments are registered
from multigrid import envs  

# SKRL imports
from skrl.multi_agents.torch.mappo.mappo_rnn import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils import set_seed

# Custom CNN feature extractor for MiniGrid-like observations
from minigrid_extractor import MinigridFeaturesExtractor  

# Set seed for reproducibility (optional)
set_seed(42)

# ✅ Load the trained model from checkpoint
CHECKPOINT_PATH = "" 

# Environment Setup
num_agents = 3
env = gym.make('MultiGrid-Empty-Random-16x16-v0', agents=num_agents)
env = wrap_env(env, wrapper="multigrid")  # ✅ Required for PyTorch training
device = "cuda" if torch.cuda.is_available() else "cpu"


# 🧠 Define Policy Model: CNN+RNN(GRU)+MLP
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 num_envs=1, num_rnn_layers=1, hidden_size=64, sequence_length=64):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self)

        self.num_envs = num_envs
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.feature_extractor = MinigridFeaturesExtractor(observation_space, features_dim=512)

        self.rnn = nn.GRU(input_size = 512,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_rnn_layers,
                          batch_first=True)

        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_rnn_layers, self.num_envs, self.hidden_size)]}}
    
    def compute(self, inputs, role):
        # B=batch_size, L=sequence_length, d=feture dimentions, h = hidden state dimension 
        features = self.feature_extractor(inputs["states"]) # shape (B*L, d) B*L = memory_size/Num. mini-batches
        terminated = inputs.get("terminated", None) # shape (B*L, 1)
        hidden_states = inputs["rnn"][0] # shape (num_rnn_layers, B*L, h)

        # training
        if self.training:
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])  # reshaping to (B, L, d)
            hidden_states = hidden_states.view(self.num_rnn_layers, -1, self.sequence_length, hidden_states.shape[-1])  # reshaping to (num_rnn_layers, B, L, h)

            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (num_rnn_layers, B, h)

            # reset the RNN state in the middle of a sequence
            # TODO: debug may be needed now is an all-or-nothing approach 
            if terminated is not None and torch.any(terminated): 
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states) # slice from time step i0 to i1-1 
                    hidden_states[:, (terminated[:,i1-1]), :] = 0 # reset, zero out hidden states for all batch elements that ended at i1-1
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)

            # no need to reset the RNN state in the sequence
            else:
                # hidden_states: (num_rnn_layers, B, h), rnn_input: (B, L, d)
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states) # x_{1:t}, h_{t} = rnn(x_{0:t-1}, h_{0})
                # rnn_output: (B, L, d)

        # rollout
        else:
            rnn_input = features.view(-1, 1, features.shape[-1])  # reshaping to (B, d)
            # rnn_input (B, 1, h), hidden_states (num_rnn_layers, B, h)
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
            # rnn_output (B, 1, h)


        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (B, L, h) -> (N * L, D ∗ H_out)
        logits = self.policy_net(rnn_output)

        return logits, {"rnn": [hidden_states]}


# 🧠 Define Value Model (Critic)
class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 num_envs=1, num_rnn_layers=1, hidden_size=64, sequence_length=128):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)  

        self.num_envs = num_envs
        self.num_rnn_layers = num_rnn_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.feature_extractor = MinigridFeaturesExtractor(observation_space, features_dim=512)

        self.rnn = nn.GRU(input_size = 512,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_rnn_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_rnn_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        features = self.feature_extractor(inputs["states"])
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = features.view(-1, self.sequence_length, features.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_rnn_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
            
        # rollout
        else:
            rnn_input = features.view(-1, 1, features.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.value_net(rnn_output), {"rnn": [hidden_states]}

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

# 🎛 Configure MAPPO_RNN Agent
cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": next(iter(env.observation_spaces.values())), "device": device}
cfg["shared_state_preprocessor"] = RunningStandardScaler
cfg["shared_state_preprocessor_kwargs"] = {"size": next(iter(env.state_spaces.values())), "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

# ✅ Set up logging & checkpoints
cfg["experiment"]["directory"] = "runs/torch/MultiGrid_MAPPO_RNN_CustomReward"
cfg["experiment"]["write_interval"] = 5000
cfg["experiment"]["checkpoint_interval"] = 10000

trained_agent = MAPPO_RNN(
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
    with torch.no_grad():
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
gif_path = "multigrid_mappo_rnn_demo.gif"
imageio.mimsave(gif_path, frames, fps=10)  # Adjust FPS for speed control

print(f"🎬 GIF saved successfully at {gif_path}!")