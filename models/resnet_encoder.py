# models/resnet_encoder.py
"""
ResNet-50 visual feature encoder.

- Input (legacy): HxWx3 uint8 numpy array
- Input (batch, preferred): BxCxHxW torch tensor (uint8 in [0,255] or float in [0,1]/[0,255])
- Output: 2048-d features (torch on device for batch; CPU tensor for legacy encode by default)

This version avoids PIL and supports true batch inference for speed.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    def __init__(self, device: Union[str, torch.device] = "cpu", out_dim: int = 2048):
        super().__init__()
        self.device = torch.device(device)

        weights = models.ResNet50_Weights.DEFAULT
        net = models.resnet50(weights=weights)
        # Remove classifier, keep global pooled features
        net.fc = nn.Identity()
        self.backbone = net

        # Freeze
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Normalization stats
        mean = torch.tensor(weights.transforms().mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(weights.transforms().std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

        self.out_dim = out_dim
        self.to(self.device)

    @torch.no_grad()
    def encode_torch_batch(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images.

        Args:
            rgb: torch tensor with shape (B, C, H, W).
                 Can be uint8 [0,255] or float (either [0,1] or [0,255]).
        Returns:
            feats: (B, 2048) torch.float32 on self.device
        """
        if rgb.dim() != 4:
            raise ValueError(f"encode_torch_batch expects (B,C,H,W), got {tuple(rgb.shape)}")

        x = rgb.to(self.device)

        # Convert to float in [0,1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            # Heuristic: if values look like [0,255], scale down
            if x.max() > 1.5:
                x = (x / 255.0).clamp(0.0, 1.0)

        # Ensure channels-first 3
        if x.shape[1] != 3:
            raise ValueError(f"Expected C=3, got C={x.shape[1]}")

        # Resize to 224x224 (ResNet default)
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Normalize
        x = (x - self._mean) / self._std

        feats = self.backbone(x)  # (B,2048)
        if feats.dim() != 2:
            feats = feats.flatten(1)
        return feats

    @torch.no_grad()
    def encode(self, rgb_uint8: np.ndarray, return_cpu: bool = True) -> torch.Tensor:
        """
        Legacy single-image API.

        Args:
            rgb_uint8: HxWx3 uint8 numpy array.
            return_cpu: if True returns CPU tensor (2048,), else device tensor.
        """
        if not isinstance(rgb_uint8, np.ndarray):
            raise TypeError("encode expects a numpy array HxWx3 uint8")
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            raise ValueError(f"encode expects HxWx3, got {rgb_uint8.shape}")

        # HWC -> CHW
        x = torch.from_numpy(rgb_uint8).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        feats = self.encode_torch_batch(x).squeeze(0)  # (2048,)
        return feats.cpu() if return_cpu else feats
