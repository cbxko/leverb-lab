from torch import nn

from .kinematics_decoder import KinematicsDecoder
from .kinematics_encoder import KinematicsEncoder
from .residual_latent import ResidualLatent
from .vl_transformer import VLTransformerPrior


class LeVERBVL(nn.Module):
    def __init__(self, siglip_encoder, *, state_dim=38, latent_dim=256, horizon=10, motion_dim=45):
        super().__init__()
        self.siglip_encoder = siglip_encoder
        self.prior = VLTransformerPrior(feature_dim=siglip_encoder.feature_dim, latent_dim=latent_dim)
        self.motion_encoder = KinematicsEncoder(horizon=horizon, motion_dim=motion_dim, latent_dim=latent_dim)
        self.latent = ResidualLatent()
        self.motion_decoder = KinematicsDecoder(
            state_dim=state_dim, latent_dim=latent_dim, horizon=horizon, motion_dim=motion_dim
        )

    def forward(self, fpv_images, tpv_images, instructions, current_state, *, future_motion=None,
                deterministic=None):
        if self.training and future_motion is None:
            raise ValueError("Training forward requires future_motion")
        if not self.training and future_motion is not None:
            raise ValueError("Evaluation forward uses only the prior; omit future_motion")

        features = self.siglip_encoder(fpv_images, tpv_images, instructions)
        prior_mean, prior_log_variance = self.prior(**features)
        if deterministic is None:
            deterministic = not self.training

        outputs = {"prior_mean": prior_mean, "prior_log_variance": prior_log_variance}
        if self.training:
            mean_delta, posterior_log_variance = self.motion_encoder(future_motion)
            z, posterior_mean, posterior_log_variance = self.latent(
                prior_mean, prior_log_variance, mean_delta=mean_delta,
                posterior_log_variance=posterior_log_variance, deterministic=deterministic,
            )
            outputs.update(posterior_mean=posterior_mean, posterior_log_variance=posterior_log_variance)
        else:
            z, _, _ = self.latent(prior_mean, prior_log_variance, deterministic=deterministic)

        outputs["z"] = z
        outputs["predicted_motion"] = self.motion_decoder(current_state, z)
        return outputs
