#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""a3c_mp.py

Multiprocessing / Hogwild helpers for A3C, with CUDA-safe gradient copying.

- SharedAdam: Adam optimizer with shared-memory state (shared_model must be CPU).
- ensure_shared_grads: copy local grads to shared model params; supports local_model on CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedAdam(torch.optim.Adam):
    """Adam optimizer whose state tensors are placed in shared memory (classic A3C)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def ensure_shared_grads(local_model: nn.Module, shared_model: nn.Module) -> None:
    """Copy local gradients to shared model parameters (in-place).

    Typical pattern for GPU compute:
      - local_model lives on CUDA (per-worker)
      - shared_model lives on CPU and is in shared memory
      - optimizer.step() updates shared_model on CPU
      - each worker periodically loads shared_model.state_dict() back to its CUDA local_model

    This function moves CUDA grads to CPU before copying.
    """
    for lp, sp in zip(local_model.parameters(), shared_model.parameters()):
        if lp.grad is None:
            continue
        g = lp.grad.detach()
        if g.device != sp.device:
            g = g.to(sp.device)
        if sp.grad is None:
            sp.grad = g.clone()
        else:
            sp.grad.copy_(g)
