# train_a2c.py
# A2C + 多环境并行（完善版，含 --smalltest）
#
# 用法：
#   # 小规模 sanity check：FloorPlan1-20 + GarbageCan
#   xvfb-run -a python -u train_a2c.py --smalltest --n-envs 4
#
#   # 论文训练设置：TRAIN_SCENES + TARGETS（来自 envs/scene_split.py）
#   xvfb-run -a python -u train_a2c.py --n-envs 8
#
# 说明：
# - 环境 reward/Done 语义保持 ThorObjNavEnv（论文 Eq.(7)）
# - 多环境用 SubprocVecEnv(start_method="spawn") 更稳
# - 支持学习率线性衰减（论文 Appendix：7e-4 线性到 0）

import argparse
from pathlib import Path
from typing import Callable, Dict, List

import torch
import os

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from configs import CFG
from callbacks_success import SuccessLoggerCallback

from envs.thor_objnav_env import ThorObjNavEnv
from envs.scene_split import TRAIN_SCENES, TARGETS

from models.resnet_encoder import ResNet50Encoder
from models.word2vec_embed import WordEmbed
from models.osm import OSM
from models.omt_transformer import OMTTransformer


KITCHEN20_SCENES = [f"FloorPlan{i}" for i in range(1, 21)]


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """SB3 schedule: progress_remaining 1->0."""
    def f(progress_remaining: float) -> float:
        return float(progress_remaining) * float(initial_value)
    return f


def make_env_fn(
    rank: int,
    seed: int,
    scenes: List[str],
    targets_by_room: Dict[str, List[str]],
    backbone_device: str,
    debug: bool,
) -> Callable[[], ThorObjNavEnv]:
    """Factory for VecEnv. Each process must create its own Controller/backbone/memory."""
    def _init() -> ThorObjNavEnv:
        env_seed = seed + rank

        resnet = ResNet50Encoder(device=backbone_device)
        embed = WordEmbed()

        osm = OSM(hist_len=CFG.hist_len, grid_size=CFG.grid_size, device=backbone_device).to(backbone_device)
        omt = OMTTransformer(d_model=300, nhead=4, num_layers=1, device=backbone_device).to(backbone_device)

        env = ThorObjNavEnv(
            scenes=scenes,
            targets_by_room=targets_by_room,
            resnet=resnet,
            embed=embed,
            osm=osm,
            omt=omt,
            device=backbone_device,
            debug=debug,
        )
        env.reset(seed=env_seed)
        return env
    return _init


def build_vec_env(
    n_envs: int,
    seed: int,
    scenes: List[str],
    targets_by_room: Dict[str, List[str]],
    backbone_device: str,
    debug: bool,
    use_subproc: bool = True,
):
    env_fns = [
        make_env_fn(
            rank=i,
            seed=seed,
            scenes=scenes,
            targets_by_room=targets_by_room,
            backbone_device=backbone_device,
            debug=debug,
        )
        for i in range(n_envs)
    ]
    if n_envs == 1 or not use_subproc:
        venv = DummyVecEnv(env_fns)
    else:
        venv = SubprocVecEnv(env_fns, start_method="spawn")
    return VecMonitor(venv)


def main():
    parser = argparse.ArgumentParser()

    # --- experiment scale ---
    parser.add_argument(
        "--smalltest",
        action="store_true",
        help="If set: FloorPlan1-20 (Kitchen) + only target GarbageCan. Else: paper setting TRAIN_SCENES + TARGETS.",
    )

    # --- runtime ---
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="SB3 policy device")
    parser.add_argument(
        "--backbone-device",
        type=str,
        default=None,
        choices=["cpu", "cuda", None],
        help="resnet/osm/omt device; default follows --device",
    )
    parser.add_argument("--debug", action="store_true")

    # --- training ---
    parser.add_argument("--total-timesteps", type=int, default=int(getattr(CFG, "total_timesteps", 2_000_000)))
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-steps", type=int, default=5, help="A2C rollout length; paper A3C commonly uses 5")
    parser.add_argument("--lr", type=float, default=7e-4, help="paper uses 7e-4")
    parser.add_argument("--lr-linear-decay", action="store_true", help="enable linear decay lr->0")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gae-lambda", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-5)

    # --- normalization / eval / save ---
    parser.add_argument("--vecnorm", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--ckpt-freq", type=int, default=200_000)

    # --- outputs ---
    parser.add_argument("--run-name", type=str, default="a2c_omt")
    parser.add_argument("--out-dir", type=str, default="runs_a2c")

    args = parser.parse_args()

    # 分别设置实验标签
    if args.smalltest:
        exp_tag = "smalltest_kitchen20_gc"
    else:
        exp_tag = "paper_full"

    # 根据实验类型设置输出目录
    if args.run_name is None:
        args.run_name = f"a2c_{exp_tag}"

    if args.out_dir is None:
        args.out_dir = os.path.join("runs_a2c", args.run_name)
    print(f"[Run] out_dir={args.out_dir}", flush=True)

    torch.set_num_threads(1)
    set_random_seed(args.seed)

    policy_device = args.device
    backbone_device = args.backbone_device or args.device

    # ----- scenes/targets config -----
    if args.smalltest:
        print("[Config] SMALLTEST: FloorPlan1-20 (Kitchen) + goal=GarbageCan", flush=True)
        scenes = KITCHEN20_SCENES
        targets_by_room = dict(TARGETS)
        targets_by_room["Kitchen"] = ["GarbageCan"]
    else:
        print("[Config] FULL (paper): TRAIN_SCENES + TARGETS", flush=True)
        scenes = TRAIN_SCENES
        targets_by_room = dict(TARGETS)

    # ----- dirs -----
    root = Path(args.out_dir) / args.run_name
    tb_dir = root / "tb"
    succ_dir = root / "success_logs"
    ckpt_dir = root / "checkpoints"
    best_dir = root / "best"
    final_dir = root / "final"
    for d in [tb_dir, succ_dir, ckpt_dir, best_dir, final_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ----- envs -----
    env = build_vec_env(
        n_envs=args.n_envs,
        seed=args.seed,
        scenes=scenes,
        targets_by_room=targets_by_room,
        backbone_device=backbone_device,
        debug=args.debug or getattr(CFG, "debug", False),
        use_subproc=True,
    )
    eval_env = build_vec_env(
        n_envs=1,
        seed=args.seed + 10_000,
        scenes=scenes,
        targets_by_room=targets_by_room,
        backbone_device=backbone_device,
        debug=False,
        use_subproc=False,
    )

    if args.vecnorm:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False
        eval_env.norm_reward = False

    lr = linear_schedule(args.lr) if args.lr_linear_decay else args.lr

    policy_kwargs = dict(net_arch=[256, 256])

    model = A2C(
        policy="MlpPolicy",
        env=env,
        device=policy_device,
        n_steps=args.n_steps,
        gamma=args.gamma,
        learning_rate=lr,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        rms_prop_eps=args.rmsprop_eps,
        use_rms_prop=True,
        normalize_advantage=True,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        verbose=1,
    )

    cb_success = SuccessLoggerCallback(out_dir=str(succ_dir), window=100, verbose=1)

    cb_ckpt = CheckpointCallback(
        save_freq=max(1, args.ckpt_freq // max(1, args.n_envs)),
        save_path=str(ckpt_dir),
        name_prefix=f"a2c_ne{args.n_envs}",
        save_replay_buffer=False,
        save_vecnormalize=args.vecnorm,
    )

    cb_eval = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(best_dir / "eval"),
        eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[cb_success, cb_ckpt, cb_eval],
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    model.save(str(final_dir / "a2c_final"))
    if args.vecnorm:
        env.save(str(final_dir / "vecnormalize.pkl"))

    try:
        env.close()
    except Exception:
        pass
    try:
        eval_env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
