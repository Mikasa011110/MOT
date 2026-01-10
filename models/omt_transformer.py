# models/omt_transformer.py
import math
import torch
import torch.nn as nn


def sinusoidal_pos_encoding(T, d, device):
    """
    生成标准的正弦位置编码（sinusoidal positional encoding）
    用于给记忆序列中的每一个时间步注入“顺序信息”。

    Args:
        T (int): 记忆长度（时间步数）
        d (int): 特征维度（这里是 300）
        device: 张量所在设备

    Returns:
        pe (Tensor): (T, d) 的位置编码
    """
    pe = torch.zeros(T, d, device=device)
    for pos in range(T):
        for i in range(0, d, 2):
            div = 10000 ** (i / d)
            pe[pos, i] = math.sin(pos / div)
            if i + 1 < d:
                pe[pos, i + 1] = math.cos(pos / div)
    return pe


class OMTTransformer(nn.Module):
    """
    目标条件化 Object Memory Transformer（OMT）

    该模块对应论文中的：
        x = f_dec(w_g, f_enc(M))

    功能说明：
    1) encoder：对 Object Semantic Memory 中的历史记忆序列进行自注意力建模
    2) decoder（cross-attention）：使用目标词向量 w_g 作为 query，
       从记忆序列中检索与当前目标最相关的信息
    3) 输出一个 300 维目标相关特征，用于策略网络（PPO）决策
    """

    def __init__(self, d_model=300, nhead=5, num_layers=1, device="cpu"):
        super().__init__()
        self.device = device

        # -------------------------------
        # 1. Transformer Encoder
        # -------------------------------
        # 对历史记忆序列 mem = {m_1, ..., m_T} 做 self-attention
        # 用于建模不同时间步之间的关系
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers
        )

        # -------------------------------
        # 2. Cross-Attention（Decoder 核心）
        # -------------------------------
        # 使用目标词嵌入 w_g 作为 Query
        # 使用编码后的记忆 mem_enc 作为 Key / Value
        # 实现“目标条件化地读取记忆”
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )

        # -------------------------------
        # 3. FFN + LayerNorm
        # -------------------------------
        # 与标准 Transformer block 一致，用于稳定训练
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

        # -------------------------------
        # 4. 输出映射
        # -------------------------------
        # 将 Transformer 输出映射为最终用于 PPO 的状态特征
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

    def forward(self, mem_TxD: torch.Tensor, wg_D: torch.Tensor):
        """
        前向传播

        Args:
            mem_TxD (Tensor):
                Object Semantic Memory 输出的记忆序列
                形状为 (T, 300)，T 为时间步数

            wg_D (Tensor):
                目标类别的词向量（Word2Vec）
                形状为 (300,)

        Returns:
            feat (Tensor):
                目标条件化后的状态表征
                形状为 (300,)，供 PPO policy / value network 使用
        """

        # -------------------------------
        # Step 1：对记忆序列加入位置编码
        # -------------------------------
        T, D = mem_TxD.shape
        pe = sinusoidal_pos_encoding(T, D, mem_TxD.device)
        mem = (mem_TxD + pe).unsqueeze(0)   # (1, T, 300)

        # -------------------------------
        # Step 2：Encoder（自注意力）
        # -------------------------------
        # 对历史记忆进行时间维度上的 self-attention 建模
        mem_enc = self.encoder(mem)         # (1, T, 300)

        # -------------------------------
        # Step 3：构造目标查询向量 w_g
        # -------------------------------
        # 将目标词向量作为 decoder 的 query
        if wg_D.dim() == 1:
            wg = wg_D.unsqueeze(0).unsqueeze(1)  # (1, 1, 300)
        else:
            # 如果已经是 batch 形式 (B,300)
            wg = wg_D.unsqueeze(1)               # (B, 1, 300)

        # -------------------------------
        # Step 4：Cross-Attention（核心）
        # -------------------------------
        # Query  = w_g（目标）
        # Key    = mem_enc（记忆）
        # Value  = mem_enc（记忆）
        # 输出表示“与目标最相关的记忆摘要”
        attn_out, _ = self.cross_attn(
            query=wg,
            key=mem_enc,
            value=mem_enc
        )                                         # (1, 1, 300)

        # -------------------------------
        # Step 5：残差连接 + FFN
        # -------------------------------
        x = self.ln1(wg + attn_out)
        x2 = self.ffn(x)
        x = self.ln2(x + x2)

        # -------------------------------
        # Step 6：输出最终状态特征
        # -------------------------------
        feat = x.squeeze(1)                      # (1, 300)
        return self.pool(feat).squeeze(0)        # (300,)
