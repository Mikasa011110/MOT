# models/osm.py
import numpy as np
import torch
import torch.nn as nn


class OSM(nn.Module):
    """
    Object-Scene Memory (OSM)

    论文中的 OSM 用于：
    - 存储最近 T 个时间步的“场景外观 + 目标相关物体语义”
    - 将每个时间步的信息融合成一个 300 维 memory token
    - 输出一个 (T, 300) 的 memory 序列，供 Object Memory Transformer (OMT) 使用

    对应论文公式：
        m_t = f_m( f_v(v_t), f_o(o_t) )
    """

    def __init__(self, hist_len=4, grid_size=16, device="cpu"):
        """
        Args:
            hist_len (int): 记忆长度 T（即 Transformer 看到的历史步数）
            grid_size (int): object grid 的边长（论文中为 16）
            device (str): 运行设备 cpu / cuda
        """
        super().__init__()
        self.hist_len = hist_len
        self.device = device

        # ========= 论文中的三个子网络 =========
        # f_v : 场景外观编码
        #   输入：ResNet-50 提取的 2048 维视觉特征
        #   输出：300 维场景嵌入
        self.fv = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 300)
        )

        # f_o : 目标相关物体语义编码
        #   输入：16x16 的 object context grid（展平后 256 维）
        #   输出：300 维物体语义嵌入
        self.fo = nn.Sequential(
            nn.Linear(grid_size * grid_size, 512),
            nn.ReLU(),
            nn.Linear(512, 300)
        )

        # f_m : 融合网络
        #   输入：concat([scene_embed, object_embed]) → 600 维
        #   输出：最终 memory token m_t（300 维）
        self.fm = nn.Sequential(
            nn.Linear(600, 300),
            nn.ReLU()
        )

        # 初始化 memory
        self.reset()

    def reset(self):
        """
        在 episode 开始时调用
        清空 Object-Scene Memory

        memory 中每个元素是：
            (v2048, grid256)
        """
        self.mem = []  # list[(Tensor[2048], Tensor[256])]

    @torch.no_grad()
    def push(self, v2048: torch.Tensor, grid: np.ndarray):
        """
        向 OSM 中写入当前时间步的信息（写入阶段不参与反向传播）

        Args:
            v2048 (Tensor): ResNet 提取的视觉特征，shape = (2048,)
            grid (np.ndarray): 目标相关 object grid，shape = (16,16)

        说明：
        - OSM 是一个 ring buffer，只保留最近 hist_len 个时间步
        - 超过 hist_len 时，最早的记忆会被丢弃
        """
        # 将 grid 展平成 256 维向量
        g = torch.from_numpy(grid.reshape(-1)).float()

        # 如果 memory 满了，弹出最早的一步
        if len(self.mem) >= self.hist_len:
            self.mem.pop(0)

        # 存储在 CPU 上，forward 时再搬到 device
        self.mem.append((v2048.float().cpu(), g))

    def forward(self):
        """
        从 OSM 中读取 memory，生成 Transformer 使用的 memory 序列

        Returns:
            Tensor:
                shape = (T, 300)
                T = hist_len

        注意：
        - 若当前 episode 步数 < T，会在前面进行 zero padding
        - padding token 理论上应在 Transformer 中被 mask 掉
        """
        T = self.hist_len
        out = []

        # 对 memory 中的每个时间步做特征融合
        for (v, g) in self.mem:
            # 场景外观编码
            vv = self.fv(v.to(self.device))    # (300,)

            # 目标相关物体语义编码
            oo = self.fo(g.to(self.device))    # (300,)

            # 融合得到 memory token m_t
            m = self.fm(torch.cat([vv, oo], dim=-1))  # (300,)

            out.append(m)

        # 如果 memory 长度不足 T，在前面补零
        if len(out) < T:
            pad = [
                torch.zeros(300, device=self.device)
                for _ in range(T - len(out))
            ]
            out = pad + out

        # 返回 shape = (T, 300)
        return torch.stack(out, dim=0)
