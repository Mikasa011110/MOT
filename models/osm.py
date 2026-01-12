"""models/osm.py

Scheme-A (CPU actors + GPU learner) friendly OSM implementation.

We split OSM into:
  - OSMCore: trainable parameters (fv, fo, fm)
  - OSMState: per-actor ring buffer (stores v2048 + grid)

Why: in a centralized GPU learner setup, *parameters* should be shared (single
copy on GPU), while *memory state* must be independent per actor/episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Tuple

import torch
import torch.nn as nn


@dataclass
class OSMState:
    """Per-actor OSM ring buffer state (no parameters).

    Stores the last ``hist_len`` tuples of:
      - v2048: torch.Tensor shape (2048,) (typically CPU tensor)
      - grid256: torch.Tensor shape (256,) (typically CPU tensor)
    """

    hist_len: int
    v_buf: Deque[torch.Tensor]
    g_buf: Deque[torch.Tensor]

    @classmethod
    def create(cls, hist_len: int) -> "OSMState":
        return cls(hist_len=hist_len, v_buf=deque(maxlen=hist_len), g_buf=deque(maxlen=hist_len))

    def reset(self):
        self.v_buf.clear()
        self.g_buf.clear()

    def push(self, v2048: torch.Tensor, grid256: torch.Tensor):
        # Store on CPU by default to keep actor state light; learner will move to GPU when needed.
        self.v_buf.append(v2048.detach().float().cpu())
        self.g_buf.append(grid256.detach().float().cpu())

    def padded_stacks(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return padded stacks (T,2048) and (T,256) on ``device``.

        Pads *in front* with zeros if shorter than hist_len.
        """

        T = self.hist_len
        n = len(self.v_buf)
        if n == 0:
            v = torch.zeros((T, 2048), device=device)
            g = torch.zeros((T, 256), device=device)
            return v, g

        v_list = list(self.v_buf)
        g_list = list(self.g_buf)

        v_stack = torch.stack(v_list, dim=0).to(device, non_blocking=True)  # (n,2048)
        g_stack = torch.stack(g_list, dim=0).to(device, non_blocking=True)  # (n,256)

        if n < T:
            pad_v = torch.zeros((T - n, 2048), device=device)
            pad_g = torch.zeros((T - n, 256), device=device)
            v_stack = torch.cat([pad_v, v_stack], dim=0)
            g_stack = torch.cat([pad_g, g_stack], dim=0)

        return v_stack, g_stack


class OSMCore(nn.Module):
    """Trainable part of OSM: fv, fo, fm.

    Matches the paper's Eq.(1): m_t = f_m( f_v(v_t), f_o(o_t) )
    """

    def __init__(self, grid_size: int = 16):
        super().__init__()

        self.fv = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 300),
        )

        self.fo = nn.Sequential(
            nn.Linear(grid_size * grid_size, 512),
            nn.ReLU(),
            nn.Linear(512, 300),
        )

        self.fm = nn.Sequential(
            nn.Linear(600, 300),
            nn.ReLU(),
        )

    def fuse_tokens(self, v_stack: torch.Tensor, g_stack: torch.Tensor) -> torch.Tensor:
        """Fuse stacked memory slots.

        Args:
            v_stack: (T,2048) on learner device
            g_stack: (T,256)  on learner device
        Returns:
            mem_tokens: (T,300) on learner device
        """
        vv = self.fv(v_stack)  # (T,300)
        oo = self.fo(g_stack)  # (T,300)
        m = self.fm(torch.cat([vv, oo], dim=-1))  # (T,300)
        return m


class OSM(nn.Module):
    """Backward-compatible OSM module used by train_a3c.py.

    Your current training code expects:
      - osm.reset()
      - osm.push(v2048: Tensor(2048,), grid: np.ndarray(256,) or Tensor(256,))
      - osm.forward() -> Tensor(T,300)  where T=hist_len

    Internally this wrapper uses:
      - OSMCore (trainable parameters)
      - OSMState (per-episode ring buffer)

    Note: This keeps the original "per-worker buffer inside the module" behavior,
    which is fine for your current A3C implementation where each worker owns a
    local_model instance.
    """

    def __init__(self, hist_len: int = 5, grid_size: int = 16, device: str | torch.device = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.core = OSMCore(grid_size=grid_size)
        self.state = OSMState.create(hist_len=hist_len)
        self.core.to(self.device)

    @torch.no_grad()
    def reset(self):
        self.state.reset()

    @torch.no_grad()
    def push(self, v2048: torch.Tensor, grid256):
        # v2048: (2048,) tensor
        v = v2048.detach()
        # grid256: np.ndarray(256,) or torch.Tensor(256,)
        if isinstance(grid256, torch.Tensor):
            g = grid256.detach()
        else:
            # assume numpy / list
            g = torch.as_tensor(grid256)
        g = g.float().view(-1)
        if g.numel() != 256:
            raise ValueError(f"grid256 must have 256 elements (16x16), got {g.numel()}")
        self.state.push(v, g)

    def forward(self) -> torch.Tensor:
        v_stack, g_stack = self.state.padded_stacks(self.device)  # (T,2048), (T,256)
        return self.core.fuse_tokens(v_stack, g_stack)            # (T,300)
