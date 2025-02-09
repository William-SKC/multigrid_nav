import gymnasium as gym
import torch
import torch.nn as nn



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
