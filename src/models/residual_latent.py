import torch
from torch import nn


class ResidualLatent(nn.Module):
    @staticmethod
    def _check_pair(mean, other):
        if mean.ndim != 2 or mean.numel() == 0 or mean.shape != other.shape:
            raise ValueError("Expected matching nonempty tensors shaped (B, latent_dim)")
        if mean.device != other.device:
            raise ValueError("Latent parameters must be on the same device")
        if not mean.is_floating_point() or not other.is_floating_point():
            raise ValueError("Latent parameters must be floating point")

    def posterior(self, prior_mean, mean_delta, posterior_log_variance):
        self._check_pair(prior_mean, mean_delta)
        self._check_pair(prior_mean, posterior_log_variance)
        return prior_mean + mean_delta, posterior_log_variance

    def sample(self, mean, log_variance):
        self._check_pair(mean, log_variance)
        std = torch.exp(0.5 * log_variance)
        return mean + std * torch.randn_like(std)

    def forward(
        self,
        prior_mean,
        prior_log_variance,
        *,
        mean_delta=None,
        posterior_log_variance=None,
        deterministic=False,
    ):
        self._check_pair(prior_mean, prior_log_variance)
        if (mean_delta is None) != (posterior_log_variance is None):
            raise ValueError("Provide both posterior inputs or neither")

        if mean_delta is None:
            mean, log_variance = prior_mean, prior_log_variance
        else:
            mean, log_variance = self.posterior(prior_mean, mean_delta, posterior_log_variance)

        z = mean if deterministic else self.sample(mean, log_variance)
        return z, mean, log_variance
