#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3C-style "threads" training for AI2-THOR ObjNav with OSM + OMT (paper-aligned):
- Multiprocessing workers run their own env instances ("threads" in the paper).
- Shared global model updated asynchronously (A3C/Hogwild).
- Each worker maintains its own OSM *buffer* (reset per episode); OSM parameters are trained.
- Visual backbone (ResNet50) is frozen (your ResNet50Encoder already does this).
- Goal embedding uses frozen Word2Vec 300-d vectors from --w2v-table (paper-aligned).

Key robustness features:
- spawn start method set in __main__ before any mp objects are created
- err_q/ready_q/stats_q to avoid silent crashes
- torch.set_num_threads(1) per worker to avoid CPU thread explosion
- preprocess_obs makes rgb contiguous/writable to avoid negative-stride crashes

Run:
  python -u train_a3c_osm_threads_omt_fixed.py --smalltest --num-workers 8 --device cpu \
      --w2v-table /path/to/thor_w2v_table.npz
"""

import os
import time
import argparse
import csv
from collections import deque
import traceback
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.distributions import Categorical

from configs import CFG
from models.osm import OSM
from models.resnet_encoder import ResNet50Encoder
from models.omt_transformer import OMTTransformer
from envs.thor_objnav_env import ThorObjNavEnv


# ---------------------------
# Shared RMSprop (Hogwild) — paper-aligned
# ---------------------------
class SharedRMSprop(torch.optim.RMSprop):
    """RMSprop optimizer whose state tensors are placed in shared memory (A3C classic).

    Paper alignment (Appendix A - Implementation Details):
      - Optimizer: RMSprop
      - LR: 7e-4 linearly decayed to 0 over training
    """

    def __init__(
        self,
        params,
        lr: float = 7e-4,
        alpha: float = 0.99,
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        # Place optimizer state in shared memory for Hogwild updates
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["square_avg"] = torch.zeros_like(p.data)
                state["step"].share_memory_()
                state["square_avg"].share_memory_()
                if momentum != 0.0:
                    state["momentum_buffer"] = torch.zeros_like(p.data)
                    state["momentum_buffer"].share_memory_()
                if centered:
                    state["grad_avg"] = torch.zeros_like(p.data)
                    state["grad_avg"].share_memory_()


# ---------------------------
# Utils
# ---------------------------
def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_scenes_and_targets(smalltest: bool):
    # NOTE: smalltest is for debugging only.
    if smalltest:
        scenes = [f"FloorPlan{i}" for i in range(1, 21)]
        targets_by_room = {"Kitchen": ["GarbageCan"]}
        return scenes, targets_by_room

    # ===== Paper-aligned setting =====
    scenes = (
        [f"FloorPlan{i}" for i in range(1, 21)] +
        [f"FloorPlan{200+i}" for i in range(1, 21)] +
        [f"FloorPlan{300+i}" for i in range(1, 21)] +
        [f"FloorPlan{400+i}" for i in range(1, 21)]
    )

    targets_by_room = {
        "Kitchen": [
            "Toaster",
            "Microwave",
            "Fridge",
            "CoffeeMachine",
            "GarbageCan",
            "Bowl",
        ],
        "LivingRoom": [
            "Pillow",
            "Laptop",
            "Television",
            "GarbageCan",
            "Bowl",
        ],
        "Bedroom": [
            "HousePlant",
            "Lamp",
            "Book",
            "AlarmClock",
        ],
        "Bathroom": [
            "Sink",
            "ToiletPaper",
            "SoapBottle",
            "LightSwitch",
        ],
    }
    return scenes, targets_by_room

def assert_targets_in_w2v_table(
    targets_by_room: Dict[str, List[str]],
    w2v_tokens: List[str],
    *,
    smalltest: bool = False,
):
    """
    Abort training if any target objectType is missing from the W2V table.

    This prevents silent OOV -> zero-vector failures and ensures
    strict paper-aligned semantics.
    """
    if smalltest:
        # smalltest is for debugging only; do not enforce paper vocab
        return

    token_set = set(str(t) for t in w2v_tokens)

    missing = []
    for room, targets in targets_by_room.items():
        for t in targets:
            if t not in token_set:
                missing.append((room, t))

    if missing:
        lines = [
            "[FATAL] Paper target(s) missing from W2V table:",
        ]
        for room, t in missing:
            lines.append(f"  - {room}: {t}")
        lines.append("")
        lines.append("You must regenerate the W2V table with paper targets included.")
        lines.append("Hint:")
        lines.append("  python get_all_w2v.py --include-targets ...")
        raise RuntimeError("\n".join(lines))

def load_w2v_table(table_path: str) -> Tuple[List[str], np.ndarray]:
    """Load w2v tokens/vectors from .npz created earlier."""
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
    """Build (n_goals,300) embedding weight aligned with goal_vocab strings."""
    token_to_idx = {t: i for i, t in enumerate(w2v_tokens)}
    W = np.zeros((len(goal_vocab), 300), dtype=np.float32)
    missing = 0
    for i, g in enumerate(goal_vocab):
        if g in token_to_idx:
            W[i] = w2v_vecs[token_to_idx[g]]
        else:
            # keep zeros for OOV (paper uses fixed word embeddings; OOV shouldn't happen if vocab matches)
            missing += 1
    if missing > 0:
        print(f"[W2V] OOV goals: {missing}/{len(goal_vocab)} -> zero vectors", flush=True)
    return torch.from_numpy(W)


def preprocess_obs(obs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert env obs (numpy) -> torch tensors safely.

    Returns:
      rgb: (1,3,H,W) uint8
      grid: (1,G,G) float32
      goal_id: (1,) long
    """
    rgb = obs["rgb"]
    if not isinstance(rgb, np.ndarray):
        raise ValueError("rgb must be numpy array")

    # Fix non-writable / negative-stride / non-contiguous arrays from AI2-THOR buffers
    if (not rgb.flags["C_CONTIGUOUS"]) or (not rgb.flags["WRITEABLE"]) or any(s < 0 for s in rgb.strides):
        rgb = np.ascontiguousarray(rgb)

    if rgb.ndim == 3 and rgb.shape[-1] == 3:
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # NCHW
    elif rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb_t = torch.from_numpy(rgb).unsqueeze(0)
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


def ensure_shared_grads(local_model: nn.Module, shared_model: nn.Module):
    """Copy local gradients to shared model parameters.

    Important when local_model runs on CUDA but shared_model stays on CPU (classic A3C):
    we must move grads to CPU before assigning to shared params, otherwise the
    optimizer step can hang / behave unpredictably.
    """
    for lp, sp in zip(local_model.parameters(), shared_model.parameters()):
        if lp.grad is None:
            continue
        g = lp.grad.detach()
        if g.is_cuda:
            g = g.cpu()
        if sp.grad is None:
            sp.grad = g.clone()
        else:
            sp.grad.copy_(g)


# ---------------------------
# Model: ResNet + OSM + OMT + ActorCritic
# ---------------------------
class A3CObjNavNet(nn.Module):
    """
    Paper-aligned policy input:
      - Visual: frozen ResNet50 -> proj
      - Grid: object grid -> proj
      - Goal: frozen Word2Vec 300-d -> (a) OMT query, (b) small trainable proj for policy
      - Memory: OSM produces (T,300) tokens; OMT reads memory conditioned on goal -> vector
    """

    def __init__(
        self,
        goal_vocab: List[str],
        w2v_weight_300: torch.Tensor,  # (n_goals,300)
        device: str = "cpu",
        hist_len: int = 32,
        grid_size: int = 16,
        d_vis: int = 256,
        d_grid: int = 128,
        d_goal: int = 64,
        d_omt: int = 128,
        action_dim: int = 9,
    ):
        super().__init__()
        self.device = device
        self.goal_vocab = list(goal_vocab)
        self.n_goals = int(len(goal_vocab))
        self.action_dim = int(action_dim)

        # Frozen visual backbone (your implementation freezes internally)
        self.resnet = ResNet50Encoder(device=device)
        self.vis_proj = nn.Sequential(nn.Linear(2048, d_vis), nn.ReLU())

        self.grid_proj = nn.Sequential(nn.Linear(grid_size * grid_size, d_grid), nn.ReLU())

        # Frozen Word2Vec goal embedding (paper-aligned)
        self.goal_w2v = nn.Embedding.from_pretrained(w2v_weight_300, freeze=True)
        # Small trainable projection for policy feature concat
        self.goal_proj = nn.Sequential(nn.Linear(300, d_goal), nn.ReLU())

        # OSM (trainable parameters; buffer is per-worker because local_model per worker)
        self.osm = OSM(hist_len=hist_len, grid_size=grid_size, device=device)

        # OMT (trainable)
        self.omt = OMTTransformer(d_model=300, nhead=CFG.nhead, num_layers=1, device=device)
        self.omt_proj = nn.Sequential(nn.Linear(300, d_omt), nn.ReLU())

        feat_dim = d_vis + d_grid + d_goal + d_omt
        self.pi = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, action_dim))
        self.v = nn.Sequential(nn.Linear(feat_dim, 256), nn.ReLU(), nn.Linear(256, 1))

        self.to(device)

    @torch.no_grad()
    def reset_memory(self):
        self.osm.reset()

    def _ensure_uint8_nchw(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.dtype != torch.uint8:
            rgb = (rgb * 255.0).clamp(0, 255).to(torch.uint8)
        return rgb

    def forward_features(self, rgb: torch.Tensor, grid: torch.Tensor, goal_id: torch.Tensor) -> torch.Tensor:
        B = rgb.shape[0]
        rgb = rgb.to(self.device)
        grid = grid.to(self.device)
        goal_id = goal_id.to(self.device).long().view(-1)

        rgb = self._ensure_uint8_nchw(rgb)
        v2048 = self.resnet.encode_torch_batch(rgb)           # (B,2048)
        v = self.vis_proj(v2048)                               # (B,d_vis)
        g = self.grid_proj(grid.view(B, -1))                   # (B,d_grid)

        wg300 = self.goal_w2v(goal_id)                         # (B,300) frozen
        w = self.goal_proj(wg300)                               # (B,d_goal)

        # OSM push per step (A3C worker B=1 typically, loop is cheap)
        for i in range(B):
            self.osm.push(v2048[i].detach(), grid[i].detach().cpu().numpy())

        mem = self.osm.forward()                                # (T,300)
        x = self.omt(mem, wg300)                                # (B,300) goal-conditioned
        m = self.omt_proj(x)                                    # (B,d_omt)

        return torch.cat([v, g, w, m], dim=1)

    def forward(self, rgb: torch.Tensor, grid: torch.Tensor, goal_id: torch.Tensor):
        feat = self.forward_features(rgb, grid, goal_id)
        logits = self.pi(feat)
        value = self.v(feat).squeeze(-1)
        return logits, value


# ---------------------------
# Worker
# ---------------------------
@dataclass
class WorkerCfg:
    gamma: float = 0.99
    n_steps: int = 5
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 40.0


def worker_loop(
    rank: int,
    args,
    shared_model: A3CObjNavNet,
    optimizer: SharedRMSprop,
    global_step: mp.Value,
    opt_lock: mp.Lock,
    stats_q,
    err_q,
    ready_q,
):
    try:
        import faulthandler
        faulthandler.enable()
        torch.set_num_threads(1)

        pid = os.getpid()
        ready_q.put(("alive", rank, pid, time.time()))
        time.sleep(0.2 * rank)

        set_global_seeds(args.seed + rank)
        scenes, targets_by_room = choose_scenes_and_targets(args.smalltest)

        # env (light)
        env = ThorObjNavEnv(
            scenes=scenes,
            targets_by_room=targets_by_room,
            embed=args.embed,
            # Env should stay on CPU; using CUDA here can create extra GPU contexts per worker
            # and interact badly with Unity rendering.
            device="cpu",
            debug=args.debug,
            headless=bool(getattr(args, "headless", False)),
        )

        local_model = A3CObjNavNet(
            goal_vocab=args.goal_vocab,
            w2v_weight_300=args.goal_w2v_weight,
            device=args.device,
            hist_len=CFG.hist_len,
            grid_size=CFG.grid_size,
            action_dim=env.action_space.n if hasattr(env, "action_space") else 9,
        )
        local_model.load_state_dict(shared_model.state_dict())
        local_model.train()

        cfg = WorkerCfg(
            gamma=args.gamma,
            n_steps=args.n_steps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
        )

        obs, _ = env.reset(seed=args.seed + rank)
        local_model.reset_memory()
        ep_return = 0.0
        ep_count = 0

        local_steps = 0
        last_hb = time.time()

        while True:
            with global_step.get_lock():
                if global_step.value >= args.total_frames:
                    break

            # sync local params
            local_model.load_state_dict(shared_model.state_dict())

            log_probs, values, rewards, entropies, dones = [], [], [], [], []

            for _ in range(cfg.n_steps):
                rgb_t, grid_t, goal_t = preprocess_obs(obs)

                logits, value = local_model(rgb_t, grid_t, goal_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                next_obs, reward, terminated, truncated, info = env.step(int(action.item()))
                done = bool(terminated or truncated)

                ep_return += float(reward)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(float(reward), dtype=torch.float32, device=args.device))
                entropies.append(entropy)
                dones.append(done)

                obs = next_obs

                with global_step.get_lock():
                    global_step.value += 1
                    gs = int(global_step.value)
                local_steps += 1

                if (local_steps % args.report_every) == 0:
                    stats_q.put(("steps", rank, args.report_every, time.time()))
                if (time.time() - last_hb) >= args.heartbeat_sec:
                    stats_q.put(("hb", rank, local_steps, time.time()))
                    if bool(getattr(args, "debug_workers", False)):
                        print(f"[ALIVE] worker{rank} pid={os.getpid()} step={local_steps}", flush=True)
                    last_hb = time.time()

                if done:
                    # episode logging (worker -> main)
                    ep_count += 1
                    try:
                        stats_q.put((
                            "episode",
                            rank,
                            {
                                "global_step": int(gs),
                                "episode_index": int(ep_count),
                                "episode_return": float(ep_return),
                                "episode_len": int(info.get("episode_len", 0)),
                                "episode_success": int(bool(info.get("episode_success", False))),
                                "best_sbbox": float(round(float(info.get("best_sbbox", 0.0)), 3)),
                                "episode_ever_visible": int(bool(info.get("episode_ever_visible", False))),
                                "episode_min_dist": float(round(float(info.get("episode_min_dist", -1.0)), 3)),
                                "scene": str(getattr(env, "scene", "")),
                                "goal": str(getattr(env, "goal", "")),
                            },
                            time.time(),
                        ))
                    except Exception:
                        pass

                    obs, _ = env.reset()
                    local_model.reset_memory()
                    ep_return = 0.0
                    break

                if gs >= args.total_frames:
                    break

            # bootstrap
            with torch.no_grad():
                if dones and dones[-1]:
                    R = torch.zeros((), device=args.device)
                else:
                    rgb_t, grid_t, goal_t = preprocess_obs(obs)
                    _, v_next = local_model(rgb_t, grid_t, goal_t)
                    R = v_next.detach().squeeze(0)

            returns = []
            for r, done in zip(reversed(rewards), reversed(dones)):
                R = r + cfg.gamma * R * (0.0 if done else 1.0)
                returns.append(R)
            returns = list(reversed(returns))
            returns_t = torch.stack(returns)  # (T,)

            values_t = torch.stack(values).squeeze(-1)  # (T,)
            log_probs_t = torch.stack(log_probs)        # (T,)
            ent_t = torch.stack(entropies)              # (T,)

            adv = returns_t - values_t

            policy_loss = -(log_probs_t * adv.detach()).mean() - cfg.entropy_coef * ent_t.mean()
            value_loss = cfg.value_coef * adv.pow(2).mean()
            loss = policy_loss + value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), cfg.max_grad_norm)

            with opt_lock:
                ensure_shared_grads(local_model, shared_model)
                # Paper: lr starts at args.lr (7e-4) and linearly decays to 0 over training
                with global_step.get_lock():
                    gs_now = int(global_step.value)
                frac = max(0.0, 1.0 - (gs_now / float(max(1, args.total_frames))))
                lr_now = float(args.lr) * frac
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_now
                optimizer.step()

        env.close()

    except Exception as e:
        tb = traceback.format_exc()
        try:
            err_q.put(("exception", rank, os.getpid(), str(e), tb))
        except Exception:
            pass
        print(f"[WORKER {rank}] crashed: {e}\n{tb}", flush=True)
        raise


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smalltest", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--total-frames", type=int, default=200_000)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=40.0)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-5)
    parser.add_argument("--rmsprop-momentum", type=float, default=0.0)
    parser.add_argument("--rmsprop-centered", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-workers", action="store_true", help="Print per-worker ALIVE heartbeats and enable deadlock watchdog in main.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run ALL AI2-THOR controllers in headless off-screen mode (platform=CloudRendering).",
    )
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--heartbeat-sec", type=float, default=5.0)
    parser.add_argument("--init-timeout-sec", type=float, default=20.0)
    parser.add_argument("--w2v-table", type=str, required=True, help="path to thor_w2v_table.npz")
    args = parser.parse_args()

    # Load w2v and build goal vocab
    scenes, targets_by_room = choose_scenes_and_targets(args.smalltest)
    all_goals = []
    for lst in targets_by_room.values():
        all_goals.extend([str(x) for x in lst])
    goal_vocab = sorted(list(set(all_goals)))
    args.goal_vocab = goal_vocab  # pass to workers via args (picklable)

    w2v_tokens, w2v_vecs = load_w2v_table(args.w2v_table)
    assert_targets_in_w2v_table(targets_by_room, w2v_tokens, smalltest=args.smalltest,)# 论文目标必须在 w2v table 中
    print(
    f"[OK] W2V table loaded and verified: {args.w2v_table} "
    f"(tokens={len(w2v_tokens)})",
    flush=True,
    )
    goal_w2v_weight = build_goal_w2v_embedding(goal_vocab, w2v_tokens, w2v_vecs)
    args.goal_w2v_weight = goal_w2v_weight  # torch tensor is picklable under spawn

    # Env needs WordEmbed to build grid (your current env expects embed)
    from models.word2vec_embed import WordEmbed
    args.embed = WordEmbed(table_path=args.w2v_table)

    shared_model = A3CObjNavNet(
        goal_vocab=goal_vocab,
        w2v_weight_300=goal_w2v_weight,
        device="cpu",  # classic A3C: shared params on CPU
        hist_len=CFG.hist_len,
        grid_size=CFG.grid_size,
        action_dim=9,
    )
    shared_model.share_memory()

    optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha=args.rmsprop_alpha, eps=args.rmsprop_eps, momentum=args.rmsprop_momentum, centered=bool(args.rmsprop_centered))

    global_step = mp.Value("i", 0, lock=True)
    opt_lock = mp.Lock()

    stats_q = mp.SimpleQueue()
    err_q = mp.SimpleQueue()
    ready_q = mp.SimpleQueue()

    procs = []
    for rank in range(args.num_workers):
        p = mp.Process(
            target=worker_loop,
            args=(rank, args, shared_model, optimizer, global_step, opt_lock, stats_q, err_q, ready_q),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    # wait init
    init_done = set()
    t0 = time.time()
    last_print = time.time()
    last_total = 0
    total_steps = 0
    last_msg = time.time()

    try:
        while (time.time() - t0) < float(args.init_timeout_sec) and len(init_done) < args.num_workers:
            while not err_q.empty():
                kind, r, pid, msg, tb = err_q.get()
                print(f"[ERR] worker{r} pid={pid} msg={msg}\n{tb}", flush=True)
                raise SystemExit(1)
            while not ready_q.empty():
                tag, r, pid, ts = ready_q.get()
                print(f"[READY] {tag} worker{r} pid={pid}", flush=True)
                last_msg = time.time()
                if tag == "init_done":
                    init_done.add(r)
            time.sleep(0.05)


        # ---------------------------
        # CSV logging (episode-level) + sliding window
        # ---------------------------
        log_dir = "runs_a3c"
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "train_log.csv")
        csv_fields = [
            "global_step",
            "lr",
            "episode_index",
            "episode_return",
            "episode_len",
            "episode_success",
            "best_sbbox",
            "episode_ever_visible",
            "episode_min_dist",
            "scene",
            "goal",
        ]
        csv_f = open(csv_path, "w", newline="", encoding="utf-8")
        csv_w = csv.DictWriter(csv_f, fieldnames=csv_fields)
        csv_w.writeheader()
        csv_f.flush()

        win = deque(maxlen=100)
        win = deque(maxlen=100)
        ep_total = 0
        last_sr_print_ep = 0
        last_sr_print_time = time.time()
        last_sr_print_steps = 0
        # monitor loop
        while True:
            while not err_q.empty():
                kind, r, pid, msg, tb = err_q.get()
                print(f"[ERR] worker{r} pid={pid} msg={msg}\n{tb}", flush=True)
                raise SystemExit(1)

            while not stats_q.empty():
                tag, r, val, ts = stats_q.get()
                last_msg = time.time()

                if tag == "steps":
                    total_steps += int(val)

                elif tag == "episode":
                    # val is a dict of episode metrics from worker
                    data = dict(val)
                    lr_now = optimizer.param_groups[0]["lr"]
                    data["lr"] = float(f"{lr_now:.8f}")
                    csv_w.writerow({k: data.get(k, "") for k in csv_fields})
                    csv_f.flush()

                    win.append(data)
                    ep_total += 1

                    # print SR/A3C only when we have collected another 100 episodes
                    if ep_total % 100 == 0 and ep_total > 0 and len(win) > 0:
                        now_ep = time.time()
                        avg_ret = sum(x["episode_return"] for x in win) / len(win)
                        sr100 = sum(x["episode_success"] for x in win) / len(win)
                        avg_len = sum(x["episode_len"] for x in win) / len(win)
                        avg_sbbox = sum(x["best_sbbox"] for x in win) / len(win)
                        gs_local = int(global_step.value)
                        # interval fps since last SR print (more stable than per-2s spam)
                        fps_int = (gs_local - last_sr_print_steps) / max(1e-9, (now_ep - last_sr_print_time))
                        last_sr_print_steps = gs_local
                        last_sr_print_time = now_ep
                        print(
                            f"[A3C] steps={gs_local}/{args.total_frames} fps~{fps_int:.1f}  "
                            f"[SR@{len(win)}] {sr100:.3f}  ep={ep_total}  ret={avg_ret:.3f}  len={avg_len:.1f}  sbbox={avg_sbbox:.3f}",
                            flush=True,
                        )

            gs = int(global_step.value)
            if gs > total_steps:
                total_steps = gs

            now = time.time()
            if now - last_print >= 2.0:
                # throttle progress printing; we only print with SR every 100 episodes
                last_total = total_steps
                last_print = now


                dead = [p for p in procs if p.exitcode is not None]
                if dead:
                    print("[A3C] workers exited:", [(p.pid, p.exitcode) for p in dead], flush=True)
                    raise SystemExit(1)

                if bool(getattr(args, "debug_workers", False)) and (now - last_msg) > max(10.0, float(args.heartbeat_sec) * 3) and total_steps == 0:
                    print("[A3C][DBG] no progress/no messages; possible init hang or deadlock", flush=True)
                    for i, p in enumerate(procs):
                        print(f"  worker{i} alive={p.is_alive()} exitcode={p.exitcode}", flush=True)

            if total_steps >= args.total_frames:
                break

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Interrupted, terminating workers...", flush=True)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=5)

        try:
            csv_f.close()
            print(f"[A3C] wrote CSV log: {csv_path}", flush=True)
        except Exception:
            pass

        os.makedirs("runs_a3c", exist_ok=True)
        ckpt = os.path.join("runs_a3c", "model.pt")
        torch.save({"model": shared_model.state_dict(), "args": vars(args), "goal_vocab": goal_vocab}, ckpt)
        print(f"[A3C] saved checkpoint: {ckpt}", flush=True)


if __name__ == "__main__":
    # IMPORTANT: set start method before creating any mp objects
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)
    main()