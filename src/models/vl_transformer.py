import torch
from torch import nn


class VLTransformerPrior(nn.Module):
    def __init__(self, feature_dim=768, latent_dim=256, num_layers=12, num_heads=12, mlp_dim=3072):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.modality_embedding = nn.Parameter(torch.empty(1, 3, feature_dim))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.encoder = nn.Sequential(*(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ), nn.LayerNorm(feature_dim))
        self.head = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 2 * latent_dim),
        )

    def forward(self, text, fpv, tpv):
        if text.ndim != 2 or text.shape[1] != self.feature_dim or text.shape[0] == 0:
            raise ValueError(f"Expected nonempty features shaped (B, {self.feature_dim})")
        if text.shape != fpv.shape or text.shape != tpv.shape:
            raise ValueError("Text and both camera features must have the same shape")

        tokens = torch.stack((text, fpv, tpv), dim=1)
        fused = self.encoder(tokens + self.modality_embedding).mean(dim=1)
        mean, log_variance = self.head(fused).chunk(2, dim=-1)
        return mean, log_variance
