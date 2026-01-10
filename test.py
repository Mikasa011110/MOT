#!/usr/bin/env python3
# test.py
# 训练吞吐/FPS 基准测试（多环境 sweep），与 train_a2c.py 的 env 构造保持一致
#
# 用法：
#   # 小规模headless sanity check（FloorPlan1-20 + GarbageCan）
#   python -u test.py --smalltest --headless --step=200 --w2v-path /home/ubuntu/gensim-data/word2vec-google-news-300/word2vec-google-news-300
    # 小规模有渲染
#   python -u test.py --smalltest --step=200 --w2v-path /home/ubuntu/gensim-data/word2vec-google-news-300/word2vec-google-news-300
#
#   # 论文训练设置（TRAIN_SCENES + TARGETS）
#   python -u test.py --headless --w2v-path /home/ubuntu/gensim-data/word2vec-google-news-300/word2vec-google-news-300
#
# 输出：
#   logs_fps/fps_env_sweep.csv
#   logs_fps/fps_env_sweep.png

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from typing import Callable, Optional, Dict, List

from configs import CFG
from envs.thor_objnav_env import ThorObjNavEnv
from models.resnet_encoder import ResNet50Encoder
from models.word2vec_embed import WordEmbed
from models.osm import OSM
from models.omt_transformer import OMTTransformer


KITCHEN20_SCENES = [f"FloorPlan{i}" for i in range(1, 21)]


def _full_paper_splits():
    # 延迟 import，避免 smalltest 模式下不需要也加载
    from envs.scene_split import TRAIN_SCENES, TARGETS
    return TRAIN_SCENES, TARGETS


def make_env(
    rank: int,
    seed: int,
    scenes,
    targets_by_room,
    backbone_device: str,
    w2v_path: str | None,
    embed_table: dict | None,
    debug: bool,
    headless: bool,
) -> Callable[[], ThorObjNavEnv]:
    """
    返回一个可调用对象，用于 SubprocVecEnv/DummyVecEnv 创建 env。
    重要：每个子进程必须独立创建 Controller + 模型实例。
    """
    def _init():
        device = backbone_device

        resnet = ResNet50Encoder(device=device)
        embed = WordEmbed(precomputed=embed_table) if embed_table is not None else WordEmbed(local_path=w2v_path)
        osm = OSM(hist_len=CFG.hist_len, grid_size=CFG.grid_size, device=device).to(device)
        omt = OMTTransformer(d_model=300, nhead=4, num_layers=1, device=device).to(device)

        env = ThorObjNavEnv(
            scenes=scenes,
            targets_by_room=targets_by_room,
            resnet=resnet,
            embed=embed,
            osm=osm,
            omt=omt,
            device=device,
            debug=debug,
            headless=headless,
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(n_envs: int, seed: int, scenes, targets_by_room, backbone_device: str, w2v_path: str | None, embed_table: dict | None, debug: bool, headless: bool):
    fns = [make_env(i, seed, scenes, targets_by_room, backbone_device, w2v_path, embed_table, debug, headless) for i in range(n_envs)]
    if n_envs <= 1:
        return DummyVecEnv([fns[0]])
    # spawn 更稳，避免 fork 复制 CUDA/Controller 状态
    return SubprocVecEnv(fns, start_method="spawn")


def run_fps_once(n_envs: int, *, seed: int, scenes, targets_by_room, backbone_device: str, w2v_path: str | None, embed_table: dict | None, steps: int, debug: bool, headless: bool):
    env = build_vec_env(n_envs, seed, scenes, targets_by_room, backbone_device, w2v_path, embed_table, debug, headless)

    # reset
    env.reset()
    # timed
    t0 = time.perf_counter()
    total_transitions = 0
    for _ in range(steps):
        actions = np.array([env.action_space.sample() for _ in range(n_envs)])
        env.step(actions)
        total_transitions += n_envs
    t1 = time.perf_counter()

    env.close()

    wall = max(1e-9, t1 - t0)
    fps = total_transitions / wall  # transitions/sec (env-steps per second)
    return fps, wall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smalltest", action="store_true",
                        help="True: FloorPlan1-20(Kitchen) + GarbageCan; False: 论文设置（scene_split TRAIN_SCENES+TARGETS）")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 4, 8, 16], help="要测试的 n_envs 列表")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200, help="计时阶段循环次数（每次 step 走 n_envs 个 transition）")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="resnet/osm/omt 的 device")
    parser.add_argument("--w2v-table", type=str, default=None, help="预先计算好的 w2v table (.npz)，推荐用于训练/测试")
    parser.add_argument("--w2v-path", type=str, default=None, help="原始 3.4GB word2vec 文件路径（仅离线建表时需要，不推荐多进程直接加载）")
    parser.add_argument("--out-dir", type=str, default="logs_fps")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Use AI2-THOR headless mode (CloudRendering).")
    args = parser.parse_args()

    embed_table = None
    if args.w2v_table is not None:
        embed_table = WordEmbed.load_table(args.w2v_table)
        print(f"[W2V] loaded table: {args.w2v_table} (size={len(embed_table)})", flush=True)
    else:
        print("[W2V] WARNING: --w2v-table not set. Subprocesses may try to load the 3.4GB model.", flush=True)

    # config (smalltest vs paper)
    if args.smalltest:
        scenes = KITCHEN20_SCENES
        # Kitchen 只训练 GarbageCan
        targets_by_room = {"Kitchen": ["GarbageCan"], "LivingRoom": [], "Bedroom": [], "Bathroom": []}
        mode = "smalltest_kitchen20_gc"
    else:
        scenes, targets_by_room = _full_paper_splits()
        mode = "paper_full"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"fps_env_sweep_{mode}.csv"
    png_path = out_dir / f"fps_env_sweep_{mode}.png"

    results = []
    print(f"[MODE] {mode}", flush=True)
    print(f"[CFG] device={args.device} headless={args.headless} steps={args.steps} envs={args.envs}", flush=True)

    for n_envs in args.envs:
        try:
            fps, wall = run_fps_once(
                n_envs,
                seed=args.seed,
                scenes=scenes,
                targets_by_room=targets_by_room,
                backbone_device=args.device,
                w2v_path=args.w2v_path,
                embed_table=embed_table,
                steps=args.steps,
                debug=args.debug,
                headless=args.headless,
            )
            results.append((n_envs, fps, wall))
            print(f"n_envs={n_envs:<2d}  FPS={fps:8.2f}  wall={wall:7.2f}s", flush=True)
        except Exception as e:
            # 记录失败但不中断 sweep
            results.append((n_envs, float("nan"), float("nan")))
            print(f"n_envs={n_envs:<2d}  FAILED: {type(e).__name__}: {e}", flush=True)

    # save csv
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_envs", "fps", "wall_s", "timed_steps", "device"])
        for n_envs, fps, wall in results:
            w.writerow([mode, n_envs, fps, wall, args.steps, args.device])

    # plot
    xs = [r[0] for r in results if not (np.isnan(r[1]))]
    ys = [r[1] for r in results if not (np.isnan(r[1]))]
    if len(xs) > 0:
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("n_envs")
        plt.ylabel("FPS (env-steps/sec)")
        plt.title(f"FPS vs n_envs ({mode})")
        plt.grid(True, alpha=0.3)
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"\nSaved CSV: {csv_path}", flush=True)
    if png_path.exists():
        print(f"Saved PNG: {png_path}", flush=True)

    # simple recommendation: best fps among successful runs
    valid = [(n, f) for n, f, _ in results if not np.isnan(f)]
    if valid:
        best_n, best_f = max(valid, key=lambda x: x[1])
        print(f"\n[Recommend] best throughput: n_envs={best_n}  FPS={best_f:.2f}", flush=True)


if __name__ == "__main__":
    main()
