# configs.py
from dataclasses import dataclass

@dataclass
class OMTConfig:
    # Observation
    # width: int = 400
    # height: int = 300
    width: int = 224
    height: int = 224
    grid_size: int = 16

    # Discrete motion (paper)
    move_step: float = 0.5          # meters
    rotate_step: int = 45           # degrees
    horizon_step: int = 30          # degrees (look up/down)

    # Episode
    max_steps: int = 300
    visible_distance: float = 1.5   # meters (paper)

    # Memory
    hist_len: int = 32               # 论文中OMT使用的 memory 长度为4或者32
    nhead: int = 4                  # multi-head attention 的并行子空间数（论文中为5）

    # Reward
    success_reward: float = 5.0
    step_penalty: float = -0.02
    success_distance = 1.5 
    sbbox_coef = 1.0

    total_timesteps: int = 20_000  # 默认的训练总步数

    debug: bool = False            # 是否打印调试信息


CFG = OMTConfig()
