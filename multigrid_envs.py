from typing import Any, Mapping, Tuple
import torch
import numpy as np
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space
)

class MultiGridWrapper(MultiAgentEnvWrapper):
    def __init__(self, env: Any) -> None:
        """Custom MultiGrid wrapper for skrl
        
        :param env: MultiGrid environment to wrap
        :type env: MultiGridEnv
        """
        super().__init__(env)


    def step(self, actions: Mapping[str, torch.Tensor]) -> Tuple[
        Mapping[str, torch.Tensor],  # Observations
        Mapping[str, torch.Tensor],  # Rewards
        Mapping[str, torch.Tensor],  # Termination flags
        Mapping[str, torch.Tensor],  # Truncation flags
        Mapping[str, Any],           # Additional info
    ]:
        """Perform a step in the environment"""

        # Convert PyTorch tensors to NumPy actions
        numpy_actions = {
            agent_id: actions[agent_id].cpu().numpy().item() for agent_id in actions
        }

        # Step through MultiGrid environment
        observations, rewards, terminations, truncations, infos = self._env.step(numpy_actions)

        # Convert observations to torch tensors
        observations = self._obs2tensor(observations)

        # Convert rewards, terminations, and truncations to torch tensors
        rewards = {uid: torch.tensor(rew, dtype=torch.float32, device=self.device).view(self.num_envs, -1) for uid, rew in rewards.items()}
        terminations = {uid: torch.tensor(done, dtype=torch.bool, device=self.device).view(self.num_envs, -1) for uid, done in terminations.items()}
        truncations = {uid: torch.tensor(truncated, dtype=torch.bool, device=self.device).view(self.num_envs, -1) for uid, truncated in truncations.items()}
        return observations, rewards, terminations, truncations, infos

    def state(self) -> torch.Tensor:
        """Get the global environment state"""

        # For IPPO:
        # Combine all agent observations into a single global state tensor

        # obs = self._unwrapped.gen_obs() # Use gen_obs() to gather agent-specific observations
        # global_state_tensor = torch.cat([
        #     flatten_tensorized_space(
        #         tensorize_space(self.env.observation_space[uid]['image'], value['image'], device=self.device)
        #     )
        #     for uid, value in obs.items()
        # ], dim=0)

        # For MAPPO:
        global_state = self._unwrapped.grid.encode()  # Get the global grid encoding

        # Overlay agent positions onto the global state
        for agent in self._unwrapped.agents:
            x, y = agent.state.pos  # Get agent's position
            global_state[x, y] = agent.encode()  # Embed agent information into the grid state

        # Convert global state to PyTorch tensor
        global_state = np.array(global_state, dtype=np.float32)  # Ensure NumPy format
        global_state_tensor = flatten_tensorized_space(
            tensorize_space(next(iter(self.state_spaces.values())), global_state, device=self.device)
        )

        # print('MultiGridWrapper Global state shape:', global_state_tensor.shape)
        return global_state_tensor

    def reset(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, Any]]:
        """Reset the environment"""
        ValueError("wrapper reset")
        observations, infos = self._env.reset()

        # Convert observations to torch tensors
        observations = self._obs2tensor(observations)

        return observations, infos

    def render(self, *args, **kwargs) -> Any:
        """Render the environment"""
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment"""
        self._env.close()

    def _obs2tensor(self, observation):
        """Convert a dictionary of NumPy observations to PyTorch tensors"""
        # for uid, value in observation.items():
        #     print(uid, self.env.observation_space[uid]['image'], value['image'].shape)
        return {
            uid: flatten_tensorized_space(
                tensorize_space(self._env.observation_space[uid]['image'], value['image'], device=self.device)
            )
            for uid, value in observation.items()
        }