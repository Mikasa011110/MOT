# configs.py
from dataclasses import dataclass

@dataclass
class OMTConfig:
    # Observation
    width: int = 400
    height: int = 300
    grid_size: int = 16

    # Discrete motion (paper)
    move_step: float = 0.5          # meters
    rotate_step: int = 45           # degrees
    horizon_step: int = 30          # degrees (look up/down)

    # Episode
    max_steps: int = 300
    visible_distance: float = 1.5   # meters (paper)

    # Memory
    hist_len: int = 4               # OMT-4 first

    # Reward
    success_reward: float = 5.0
    step_penalty: float = -0.02
    success_distance = 1.5 
    sbbox_coef = 1.0

    total_timesteps: int = 20_000  # 训练总步数

    debug: bool = False            # 是否打印调试信息


CFG = OMTConfig()
