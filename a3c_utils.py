#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for A3C ObjNav training (AI2-THOR):
- seeding
- scene/target selection
- Word2Vec table loading + goal embedding build
- observation preprocessing (numpy -> torch)

Kept intentionally free of training-loop logic so it can be reused by PPO/A2C scripts too.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, List
import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Set numpy/torch seeds (including CUDA if available)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_scenes_and_targets(smalltest: bool) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Return (scenes, targets_by_room).

    smalltest=True is ONLY for debugging: fewer scenes + a single goal.
    """
    if smalltest:
        scenes = [f"FloorPlan{i}" for i in range(1, 21)]
        targets_by_room = {"Kitchen": ["GarbageCan"]}
        return scenes, targets_by_room

    scenes = [f"FloorPlan{i}" for i in range(1, 31)] + \
             [f"FloorPlan{200+i}" for i in range(1, 31)] + \
             [f"FloorPlan{300+i}" for i in range(1, 31)] + \
             [f"FloorPlan{400+i}" for i in range(1, 31)]

    targets_by_room = {
        "Kitchen": ["GarbageCan", "Mug", "Apple", "Bowl"],
        "LivingRoom": ["RemoteControl", "Laptop", "Book"],
        "Bedroom": ["Pillow", "AlarmClock", "TeddyBear"],
        "Bathroom": ["SoapBottle", "ToiletPaper", "Towel"],
    }
    return scenes, targets_by_room


def load_w2v_table(table_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load w2v tokens/vectors from a .npz file (created earlier).

    Returns:
      tokens: list[str]
      vecs:   np.ndarray shape (N, 300), dtype float32
    """
    d = np.load(table_path, allow_pickle=True)
    tokens = d["tokens"].tolist()
    vecs = d["vectors"]

    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    tokens = [str(t) for t in tokens]

    vecs = np.asarray(vecs, dtype=np.float32)
    assert vecs.ndim == 2 and vecs.shape[1] == 300, f"w2v vectors must be (N,300), got {vecs.shape}"
    return tokens, vecs


def build_goal_w2v_embedding(goal_vocab: List[str], w2v_tokens: List[str], w2v_vecs: np.ndarray) -> torch.Tensor:
    """
    Build (n_goals, 300) embedding weights aligned with goal_vocab strings.

    Missing goals are left as zero vectors (should be rare if vocab is aligned).
    """
    token_to_idx = {t: i for i, t in enumerate(w2v_tokens)}
    W = np.zeros((len(goal_vocab), 300), dtype=np.float32)

    missing = 0
    for i, g in enumerate(goal_vocab):
        if g in token_to_idx:
            W[i] = w2v_vecs[token_to_idx[g]]
        else:
            missing += 1

    if missing > 0:
        print(f"[W2V] OOV goals: {missing}/{len(goal_vocab)} -> zero vectors", flush=True)

    return torch.from_numpy(W)


def preprocess_obs(obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert env obs (numpy) -> torch tensors safely.

    Returns:
      rgb:     (1, 3, H, W) uint8
      grid:    (1, G, G) float32
      goal_id: (1,) long
    """
    rgb = obs["rgb"]
    if not isinstance(rgb, np.ndarray):
        raise ValueError("obs['rgb'] must be a numpy array")

    # Fix non-writable / negative-stride / non-contiguous arrays from AI2-THOR buffers
    if (not rgb.flags.get("C_CONTIGUOUS", False)) or (not rgb.flags.get("WRITEABLE", True)) or any(s < 0 for s in rgb.strides):
        rgb = np.ascontiguousarray(rgb)

    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
    elif rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb_t = torch.from_numpy(rgb).unsqueeze(0)  # already CHW
    else:
        raise ValueError(f"Unexpected rgb shape: {rgb.shape}")

    grid = obs["grid"]
    if isinstance(grid, np.ndarray):
        grid_t = torch.from_numpy(grid).unsqueeze(0).float()
    else:
        grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)

    goal_id = obs.get("goal_id", obs.get("goal", 0))
    goal_t = torch.tensor([int(goal_id)], dtype=torch.long)

    return rgb_t, grid_t, goal_t
