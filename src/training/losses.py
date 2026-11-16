import torch
from torch import nn
from torch.nn import functional as F


class TrajectoryReconstructionLoss(nn.Module):
    def forward(self, predicted_motion, target_motion):
        if (
            predicted_motion.ndim != 3
            or predicted_motion.numel() == 0
            or predicted_motion.shape != target_motion.shape
        ):
            raise ValueError("Expected matching nonempty motion tensors shaped (B, horizon, motion_dim)")
        return F.mse_loss(predicted_motion, target_motion)


class LatentKLLoss(nn.Module):
    def forward(self, posterior_mean, posterior_log_variance, prior_mean, prior_log_variance):
        shape = posterior_mean.shape
        if posterior_mean.ndim != 2 or posterior_mean.numel() == 0 or any(
            tensor.shape != shape for tensor in (posterior_log_variance, prior_mean, prior_log_variance)
        ):
            raise ValueError("Expected matching nonempty latent tensors shaped (B, latent_dim)")

        log_variance_ratio = posterior_log_variance - prior_log_variance
        mean_term = (posterior_mean - prior_mean).square() * torch.exp(-prior_log_variance)
        kl_per_dimension = 0.5 * (torch.expm1(log_variance_ratio) - log_variance_ratio + mean_term)
        return kl_per_dimension.sum(dim=-1).mean()
