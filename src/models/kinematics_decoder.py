import torch
from torch import nn


class KinematicsDecoder(nn.Module):
    def __init__(self, state_dim=38, latent_dim=256, horizon=10, motion_dim=45):
        super().__init__()
        if min(state_dim, latent_dim, horizon, motion_dim) <= 0:
            raise ValueError("state_dim, latent_dim, horizon and motion_dim must be positive")

        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.motion_dim = motion_dim
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, horizon * motion_dim),
        )

    def forward(self, current_state, z):
        if (
            current_state.ndim != 2
            or current_state.shape[1] != self.state_dim
            or current_state.shape[0] == 0
        ):
            raise ValueError(f"Expected nonempty current state shaped (B, {self.state_dim})")
        batch_size = current_state.shape[0]
        if z.shape != (batch_size, self.latent_dim):
            raise ValueError(f"Expected z shaped ({batch_size}, {self.latent_dim})")

        features = torch.cat((current_state, z), dim=-1)
        return self.decoder(features).reshape(batch_size, self.horizon, self.motion_dim)
