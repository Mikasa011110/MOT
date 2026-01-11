import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from models.resnet_encoder import ResNet50Encoder


class SimpleObjNavExtractor(BaseFeaturesExtractor):
    """
    B1 extractor (NO OSM/OMT):
      rgb -> ResNet50Encoder (frozen) -> proj
      grid -> proj
      goal_id -> Embedding

    Robust to SB3 Discrete preprocessing and rollout buffer flattening.
    Uses true batch ResNet forward (no PIL/no numpy loop) for speed.
    """

    def __init__(self, observation_space, device="cpu",
                 d_vis=256, d_grid=128, d_goal=64, **kwargs):
        super().__init__(observation_space, features_dim=1)

        self.device = device

        grid_space = observation_space.spaces["grid"]
        goal_space = observation_space.spaces["goal_id"]

        self.n_goals = int(goal_space.n)
        grid_dim = int(np.prod(grid_space.shape))

        # visual encoder (frozen, batch)
        self.resnet = ResNet50Encoder(device=device)

        self.vis_proj = nn.Sequential(
            nn.Linear(2048, int(d_vis)),
            nn.ReLU(),
        )

        self.grid_proj = nn.Sequential(
            nn.Linear(grid_dim, int(d_grid)),
            nn.ReLU(),
        )

        self.goal_emb = nn.Embedding(self.n_goals, int(d_goal))

        self._features_dim = int(d_vis) + int(d_grid) + int(d_goal)
        self.to(device)

    def _recover_goal_index(self, goal_id: torch.Tensor, B: int) -> torch.Tensor:
        """
        Convert goal_id to (B,) long indices robustly.

        Handles:
        - (B, n_goals) one-hot
        - (n_goals,) one-hot
        - (B,) indices
        - (B*n_goals,) flattened one-hot  (training buffer flatten case)
        """
        # (B, n_goals)
        if goal_id.dim() == 2:
            return goal_id.argmax(dim=1).long().view(-1)

        if goal_id.dim() == 1:
            n = goal_id.numel()

            # (B,) indices already
            if n == B and goal_id.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                return goal_id.long().view(-1)

            # (n_goals,) one-hot single sample
            if (
                n == self.n_goals
                and goal_id.dtype != torch.long
                and goal_id.max() <= 1.0 and goal_id.min() >= 0.0
                and torch.isclose(goal_id.sum(), torch.tensor(1.0, device=goal_id.device))
            ):
                return goal_id.argmax().view(1).long()

            # (B*n_goals,) flattened one-hot
            if (
                n == B * self.n_goals
                and goal_id.dtype != torch.long
                and goal_id.max() <= 1.0 and goal_id.min() >= 0.0
            ):
                oh = goal_id.view(B, self.n_goals)
                return oh.argmax(dim=1).long().view(-1)

            # more general flattened one-hot: (k*n_goals,)
            if (
                n % self.n_goals == 0
                and goal_id.dtype != torch.long
                and goal_id.max() <= 1.0 and goal_id.min() >= 0.0
            ):
                k = n // self.n_goals
                oh = goal_id.view(k, self.n_goals)
                idx = oh.argmax(dim=1).long().view(-1)
                if idx.numel() == B:
                    return idx
                if idx.numel() > B:
                    return idx[:B]
                pad = idx.new_zeros((B - idx.numel(),))
                return torch.cat([idx, pad], dim=0)

            # otherwise treat as indices
            idx = goal_id.long().view(-1)
            if idx.numel() == B:
                return idx
            if idx.numel() > B:
                return idx[:B]
            pad = idx.new_zeros((B - idx.numel(),))
            return torch.cat([idx, pad], dim=0)

        # fallback
        idx = goal_id.long().view(-1)
        if idx.numel() == B:
            return idx
        if idx.numel() > B:
            return idx[:B]
        pad = idx.new_zeros((B - idx.numel(),))
        return torch.cat([idx, pad], dim=0)

    def forward(self, observations):
        rgb = observations["rgb"].to(self.device)
        grid = observations["grid"].to(self.device)
        goal_id = observations["goal_id"].to(self.device)

        B = rgb.shape[0]

        # SB3 VecTransposeImage gives (B,C,H,W). Make sure dtype is okay:
        # ResNet encoder handles uint8 or float
        goal_idx = self._recover_goal_index(goal_id, B)

        v2048 = self.resnet.encode_torch_batch(rgb)      # (B,2048) on device
        v = self.vis_proj(v2048)                         # (B,d_vis)
        g = self.grid_proj(grid.view(B, -1))             # (B,d_grid)
        w = self.goal_emb(goal_idx)                      # (B,d_goal)

        return torch.cat([v, g, w], dim=1)
