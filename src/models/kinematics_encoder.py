from torch import nn


class KinematicsEncoder(nn.Module):
    def __init__(self, horizon=10, motion_dim=45, latent_dim=256):
        super().__init__()
        if min(horizon, motion_dim, latent_dim) <= 0:
            raise ValueError("horizon, motion_dim and latent_dim must be positive")

        self.horizon = horizon
        self.motion_dim = motion_dim
        self.encoder = nn.Sequential(
            nn.Linear(horizon * motion_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(latent_dim, 2 * latent_dim)

    def forward(self, future_motion):
        if (
            future_motion.ndim != 3
            or future_motion.shape[1:] != (self.horizon, self.motion_dim)
            or future_motion.shape[0] == 0
        ):
            raise ValueError(
                f"Expected nonempty future motion shaped (B, {self.horizon}, {self.motion_dim})"
            )

        features = self.encoder(future_motion.flatten(start_dim=1))
        mean_delta, posterior_log_variance = self.head(features).chunk(2, dim=-1)
        return mean_delta, posterior_log_variance
