import gymnasium as gym
import numpy as np

class MultiGridMAPPOWrapper(gym.Wrapper):
    """MultiGrid Wrapper for MAPPO:
       - Agents observe only their local environment (`7x7` grid).
       - The shared observation space sees the **entire** environment (`8x8` grid).
       - Uses a **single environment instance** without `FullyObsWrapper`.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.env_core = env.unwrapped  # ✅ Access raw MultiGrid environment
        self.num_agents = self.env_core.num_agents  # ✅ Get number of agents
        self.agents = list(self.env_core.action_space.keys())  # ✅ Extract agent IDs

        # ✅ Define individual observation space: Each agent gets only its `7x7` local image
        self.observation_spaces = {
            agent_id: env.observation_space[agent_id]["image"]  # Local `7x7` grid
            for agent_id in self.agents
        }

        # ✅ Define action space: Each agent gets its own Discrete action space
        self.action_spaces = {
            agent_id: env.action_space[agent_id]
            for agent_id in self.agents
        }

        # ✅ Define shared observation space: Manually reconstruct full `8x8` grid
        shared_shape = list(env.observation_space[self.agents[0]]["image"].shape)  # Get shape
        shared_shape[0] = self.env_core.width  # Use full grid width (8)
        shared_shape[1] = self.env_core.height  # Use full grid height (8)

        self.shared_observation_spaces = {
            agent_id: gym.spaces.Box(low=0, high=255, shape=tuple(shared_shape), dtype=np.uint8)
            for agent_id in self.agents
        }

    def reset(self, **kwargs):
        """Reset the environment and return:
           - Individual observations (local `7x7` grid per agent).
           - Shared full environment observation (`8x8` grid).
        """
        obs, info = self.env.reset(**kwargs)

        # ✅ Extract **local** observations (`7x7` per agent)
        local_obs = {agent_id: obs[agent_id]["image"] for agent_id in self.agents}

        # ✅ Manually construct the full `8x8` shared observation
        shared_obs = self._construct_global_observation()

        return local_obs, shared_obs, info  # ✅ Return both local & shared observations

    def step(self, action_dict):
        """Take a step with provided actions and return:
           - Individual observations (local `7x7` per agent).
           - Shared observation (full `8x8` environment).
        """
        obs, rewards, terminations, truncations, info = self.env.step(action_dict)
        
        # ✅ Convert observations to only local `7x7` images
        local_obs = {agent_id: obs[agent_id]["image"] for agent_id in self.agents}

        # ✅ Manually construct the full `8x8` shared observation
        shared_obs = self._construct_global_observation()

        return local_obs, shared_obs, rewards, terminations, truncations, info  # ✅ Return both

    def _construct_global_observation(self):
        """Manually stitch together all agents' views to form the full `8x8` grid."""
        global_obs = self.env_core.grid.encode()  # ✅ Get full environment encoding

        # ✅ Overlay agent positions onto the global observation
        for agent in self.env_core.agents:
            global_obs[agent.state.pos] = agent.encode()

        # ✅ Return shared observation for all agents
        return {agent_id: global_obs for agent_id in self.agents}
