import random
import time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from configs import CFG
from envs.thor_objnav_env import ThorObjNavEnv
from envs.scene_split import TARGETS
from models.resnet_encoder import ResNet50Encoder
from models.word2vec_embed import WordEmbed
from models.osm import OSM
from models.omt_transformer import OMTTransformer


KITCHEN_SCENES = [f"FloorPlan{i}" for i in range(1, 21)]


def make_demo_env():
    """
    Single-environment, visualized (headless=False) demo env.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # feature modules (same as training)
    resnet = ResNet50Encoder(device=device)
    embed = WordEmbed()
    osm = OSM(hist_len=CFG.hist_len, grid_size=CFG.grid_size, device=device).to(device)
    omt = OMTTransformer(d_model=300, nhead=4, num_layers=1, device=device).to(device)

    # target: GarbageCan only
    targets_gc = dict(TARGETS)
    targets_gc["Kitchen"] = ["GarbageCan"]

    env = ThorObjNavEnv(
        scenes=KITCHEN_SCENES,
        targets_by_room=targets_gc,
        resnet=resnet,
        embed=embed,
        osm=osm,
        omt=omt,
        device=device,
    )

    return env


def main():
    # ---- choose random kitchen ----
    scene = random.choice(KITCHEN_SCENES)
    print(f"[DEMO] Scene = {scene}, Goal = GarbageCan")

    # ---- override config for demo ----
    CFG.headless = False   # 👈 必须开画面
    CFG.max_steps = 300

    env = DummyVecEnv([make_demo_env])

    # ---- load trained policy ----
    model = PPO.load(
        "ppo_omt4_thor.zip",   # 👈 改成你的模型路径
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    obs = env.reset()

    step = 0
    done = False

    print("[DEMO] Starting rollout...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        step += 1
        done = dones[0]
        info = infos[0]

        print(
            f"[STEP {step:03d}] action={int(action[0])} "
            f"reward={reward[0]:+.3f} "
            f"done={done} "
            f"visible={info.get('episode_ever_visible', False)} "
            f"min_dist={info.get('episode_min_dist', float('nan')):.3f}"
        )

        # 放慢一点，方便人眼观察
        time.sleep(0.2)

        if step >= CFG.max_steps:
            print("[DEMO] Max steps reached.")
            break

    print("[DEMO] Episode finished.")
    env.close()


if __name__ == "__main__":
    main()
