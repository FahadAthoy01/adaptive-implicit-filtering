import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveImplicitFilter(nn.Module):
    """Locally adaptive spatial and angular bilateral filter over k-NN neighborhoods."""

    def __init__(
        self,
        base_spatial_sigma: float = 0.05,
        base_angular_sigma: float = 0.35,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.base_spatial_sigma = base_spatial_sigma
        self.base_angular_sigma = base_angular_sigma
        self.eps = eps

    def forward(
        self,
        query_pts: torch.Tensor,      # (B, 3)
        nbr_pts: torch.Tensor,        # (B, K, 3)
        query_normals: torch.Tensor,  # (B, 3)
        nbr_normals: torch.Tensor,    # (B, K, 3)
    ) -> torch.Tensor:
        q_n = F.normalize(query_normals, dim=-1, eps=self.eps).unsqueeze(1)
        k_n = F.normalize(nbr_normals, dim=-1, eps=self.eps)

        diff = nbr_pts - query_pts.unsqueeze(1)
        dist_sq = (diff ** 2).sum(dim=-1)

        # Scale spatial bandwidth by local median neighbor distance to handle sparsity
        local_scale = dist_sq.median(dim=-1, keepdim=True).values.clamp(min=self.eps)
        spatial_w = torch.exp(-dist_sq / (2.0 * (self.base_spatial_sigma ** 2) * local_scale))

        # Angular weight suppresses cross-edge smoothing at sharp features
        cos_sim = (q_n * k_n).sum(dim=-1).clamp(-1.0, 1.0)
        angular_diff = 1.0 - cos_sim.abs()
        angular_w = torch.exp(-(angular_diff ** 2) / (2.0 * (self.base_angular_sigma ** 2)))

        weights = spatial_w * angular_w
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)

        # Project along local surface normal
        signed_proj = (diff * q_n).sum(dim=-1)
        filtered_offset = (weights * signed_proj).sum(dim=-1, keepdim=True)
        return query_pts + filtered_offset * q_n.squeeze(1)
