# train_ppo.py
# 无渲染模式： xvfb-run -a /home/hekanwei/miniconda3/envs/thor/bin/python -u train_ppo.py

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from callbacks_success import SuccessLoggerCallback

from configs import CFG
from envs.scene_split import TRAIN_SCENES, TARGETS
from envs.thor_objnav_env import ThorObjNavEnv
from models.resnet_encoder import ResNet50Encoder
from models.word2vec_embed import WordEmbed
from models.osm import OSM
from models.omt_transformer import OMTTransformer

from stable_baselines3.common.callbacks import CheckpointCallback

KITCHEN_SCENES_1_21 = [f"FloorPlan{i}" for i in range(1, 21)]

def make_env():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = ResNet50Encoder(device=device)
    print("resnet encoder device attr:", getattr(resnet, "device", None), flush=True)

    embed = WordEmbed()
    osm = OSM(hist_len=CFG.hist_len, grid_size=CFG.grid_size, device=device).to(device)
    omt = OMTTransformer(d_model=300, nhead=4, num_layers=1, device=device).to(device)

    # 只训练 GarbageCan：把 Kitchen 的候选目标强制设为仅 GarbageCan
    targets_gc = dict(TARGETS)
    targets_gc["Kitchen"] = ["GarbageCan"]

    return ThorObjNavEnv(
        # scenes=TRAIN_SCENES, # 使用全部场景训练
        # targets_by_room=TARGETS, # 使用全部目标训练
        scenes=KITCHEN_SCENES_1_21, # 只用20个厨房场景训练
        targets_by_room= targets_gc, # 只训练 GarbageCan 目标
        
        resnet=resnet,
        embed=embed,
        osm=osm,
        omt=omt,
        device=device,

        debug = CFG.debug
    )

if __name__ == "__main__":
    env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        device="cpu",
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        verbose=1,
    )

    cb = SuccessLoggerCallback(out_dir="logs", window=100, verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=2000,                 # 每 2000 step保存一次checkpoint
        save_path="checkpoints",         
        name_prefix="ppo_omt4_thor",     
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    model.learn(total_timesteps=CFG.total_timesteps, callback=[cb, checkpoint_cb])
    model.save("ppo_omt4_thor")
