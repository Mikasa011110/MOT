# models/omt_transformer.py
import math
import torch
import torch.nn as nn


class SinusoidalTemporalEncoding(nn.Module):
    """
    Cached sinusoidal temporal (positional) encoding for memory tokens.
    Avoids rebuilding encoding every forward (important for CPU A3C).
    """

    def __init__(self, d_model: int = 300, max_len: int = 64):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / self.d_model)
        )  # (d/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)  # (L,d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,d) or (T,d)
        returns: same shape, with temporal encoding added
        """
        if x.dim() == 2:
            T = x.size(0)
            assert T <= self.max_len, f"T={T} exceeds max_len={self.max_len}"
            return x + self.pe[:T].to(x.dtype).to(x.device)
        elif x.dim() == 3:
            B, T, d = x.shape
            assert d == self.d_model
            assert T <= self.max_len, f"T={T} exceeds max_len={self.max_len}"
            return x + self.pe[:T].to(x.dtype).to(x.device).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")


class OMTTransformer(nn.Module):
    """
    Object Memory Transformer (OMT) core:
      - input: memory tokens M (T,300) or (B,T,300)
      - query: goal embedding w_g (300,) or (B,300)
      - output: goal-conditioned readout x (B,300)

    Implementation:
      1) add sinusoidal temporal encoding to M
      2) TransformerEncoder over M
      3) Cross-attention: query=w_g, key/value=encoded memory
      4) small FFN
    """

    def __init__(
        self,
        d_model: int = 300,
        nhead: int = 5,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        max_len: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.device = device

        self.temporal = SinusoidalTemporalEncoding(d_model=d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Cross-attention (decoder query attends to encoded memory)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.to(device)

    def forward(self, mem: torch.Tensor, wg: torch.Tensor) -> torch.Tensor:
        """
        mem: (T,300) or (B,T,300)
        wg : (300,) or (B,300)
        returns: (B,300)
        """
        # --- normalize shapes ---
        if mem.dim() == 2:
            mem = mem.unsqueeze(0)  # (1,T,D)
        elif mem.dim() != 3:
            raise ValueError(f"mem must be (T,D) or (B,T,D), got {tuple(mem.shape)}")

        B, T, D = mem.shape
        if wg.dim() == 1:
            wg = wg.view(1, 1, -1)  # (1,1,D)
        elif wg.dim() == 2:
            wg = wg.unsqueeze(1)    # (B,1,D)
        else:
            raise ValueError(f"wg must be (D,) or (B,D), got {tuple(wg.shape)}")

        if wg.size(-1) != D:
            raise ValueError(f"wg dim {wg.size(-1)} != mem dim {D}")

        # If wg has batch B but mem is only 1, repeat mem for batch
        if mem.size(0) == 1 and wg.size(0) > 1:
            mem = mem.repeat(wg.size(0), 1, 1)
            B = wg.size(0)

                # --- build padding mask BEFORE temporal encoding ---
        # Empty slots in OSM are padded with all-zero vectors (paper: masked attention for empty memory slots).
        # True values in key_padding_mask indicate positions that should be ignored by attention.
        pad_mask = (mem.abs().sum(dim=-1) == 0)  # (B,T) bool

        mem = mem.to(self.device)
        wg = wg.to(self.device)
        pad_mask = pad_mask.to(mem.device)

        # --- encode memory (masked self-attention over non-empty slots) ---
        mem = self.temporal(mem)  # (B,T,D)
        mem_enc = self.encoder(mem, src_key_padding_mask=pad_mask)  # (B,T,D)

        # --- cross attention: query=wg, key/value=mem_enc (mask empty slots) ---
        attn_out, _ = self.cross_attn(
            query=wg, key=mem_enc, value=mem_enc,
            key_padding_mask=pad_mask,
            need_weights=False,
        )  # (B,1,D)

        x = attn_out.squeeze(1)            # (B,D)
        x = x + self.ffn(x)                # residual-ish
        return x
