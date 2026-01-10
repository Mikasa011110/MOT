# models/word2vec_embed.py
#
# Word2Vec embedding wrapper.
#
# Two modes:
#   1) Heavy mode (offline preprocessing): load GoogleNews Word2Vec (3.4GB)
#   2) Table mode (recommended for train/test): load a small precomputed table (npz)
#
# Table file format (npz):
#   - tokens: (N,) array of strings
#   - vectors: (N, D) float32, already unit-normalized

import re
from typing import Dict, Optional

import numpy as np
import torch


class WordEmbed:
    """Word2Vec 词向量封装类，用于计算目标类别与物体类别之间的语义相似度。"""

    def __init__(
        self,
        model_name: str = "word2vec-google-news-300",
        local_path: Optional[str] = None,
        table_path: Optional[str] = None,
        precomputed: Optional[Dict[str, np.ndarray]] = None,
        oov_mode: str = "zero",  # 'zero' or 'rand'
    ):
        """Initialize.

        Args:
            model_name: used only for gensim-downloader (online) fallback.
            local_path: path to heavy word2vec file (offline preprocessing only).
            table_path: path to precomputed npz table. (recommended for train/test)
            precomputed: dict[token -> unit vector] passed directly.
            oov_mode: what to return when token is not found in table.
                - 'zero': all-zero vector (stable, safest)
                - 'rand': deterministic random unit vector (keeps some signal but adds noise)
        """
        self.model_name = model_name
        self.dim = 300
        self.model = None
        self.table: Optional[Dict[str, np.ndarray]] = None
        self.cache: Dict[str, np.ndarray] = {}
        self.oov_mode = oov_mode

        if precomputed is not None:
            # Table mode via in-memory dict
            self.table = {k: self._unit(np.asarray(v, dtype=np.float32)) for k, v in precomputed.items()}
            any_vec = next(iter(self.table.values()), None)
            if any_vec is not None:
                self.dim = int(any_vec.shape[0])
            return

        if table_path is not None:
            self.table = self.load_table(table_path)
            any_vec = next(iter(self.table.values()), None)
            if any_vec is not None:
                self.dim = int(any_vec.shape[0])
            return

        # Heavy mode (gensim) — only use this for offline preprocessing scripts.
        try:
            from gensim.models import KeyedVectors
            import gensim.downloader as api
        except Exception as e:
            raise ImportError(
                "需要安装 gensim 才能加载 Word2Vec 原始大模型。训练/测试推荐使用 table_path/precomputed。\n"
                "pip install gensim"
            ) from e

        if local_path is not None:
            # Prefer mmap for .kv; for word2vec_format default to binary=True for GoogleNews.
            if local_path.endswith(".kv"):
                self.model = KeyedVectors.load(local_path, mmap="r")
            else:
                # GoogleNews 解压后的文件通常没有 .bin 后缀，但仍然是 binary 格式
                self.model = KeyedVectors.load_word2vec_format(local_path, binary=True)
            self.dim = int(self.model.vector_size)
        else:
            # Online downloader fallback (often not available for GoogleNews)
            self.model = api.load(model_name)
            self.dim = int(self.model.vector_size)

    # -------------------------
    # Table utilities
    # -------------------------

    @staticmethod
    def load_table(table_path: str) -> Dict[str, np.ndarray]:
        data = np.load(table_path, allow_pickle=True)
        tokens = data["tokens"]
        vectors = data["vectors"]
        table: Dict[str, np.ndarray] = {}
        for t, v in zip(tokens, vectors):
            table[str(t)] = np.asarray(v, dtype=np.float32)
            # Helpful aliases to improve hit-rate
            t2 = str(t)
            table.setdefault(t2.lower(), table[t2])
            table.setdefault(t2.replace(" ", ""), table[t2])
        return table

    # -------------------------
    # Token normalization
    # -------------------------

    def _split_camel(self, s: str) -> list[str]:
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s)
        return [p.lower() for p in parts if p.strip()]

    def _normalize(self, s: str) -> list[str]:
        s = s.replace("_", " ").replace("-", " ").strip()

        if " " in s:
            toks = [t.lower() for t in s.split() if t.strip()]
        else:
            toks = self._split_camel(s) if any(c.isupper() for c in s) else [s.lower()]

        mapping = {
            "tv": "television",
            "tissue": "paper",
        }
        toks = [mapping.get(t, t) for t in toks]
        return toks

    # -------------------------
    # Vector utils
    # -------------------------

    def _unit(self, v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-8
        return (v / n).astype(np.float32)

    def _rand_oov(self, key: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        v = rng.normal(size=(self.dim,)).astype(np.float32)
        return self._unit(v)

    # -------------------------
    # Public API
    # -------------------------

    def get_vec(self, token: str) -> np.ndarray:
        """Return unit vector for token.

        In table mode, we try direct lookup, then simple normalized variants.
        In heavy mode, we use gensim KeyedVectors with sub-token averaging.
        """
        if token in self.cache:
            return self.cache[token]

        # --- table mode ---
        if self.table is not None:
            # Direct and common aliases
            key_candidates = [token, token.lower(), token.replace(" ", "")]
            for k in key_candidates:
                v = self.table.get(k)
                if v is not None:
                    self.cache[token] = v
                    return v

            # Try sub-token averaging if the table contains subtokens
            toks = self._normalize(token)
            vecs = []
            for t in toks:
                v = self.table.get(t)
                if v is not None:
                    vecs.append(v)
            if vecs:
                v = self._unit(np.mean(np.stack(vecs, axis=0), axis=0))
                self.cache[token] = v
                return v

            # OOV fallback
            if self.oov_mode == "rand":
                v = self._rand_oov(token)
            else:
                v = np.zeros((self.dim,), dtype=np.float32)
            self.cache[token] = v
            return v

        # --- heavy mode (gensim) ---
        toks = self._normalize(token)
        vecs = []
        for t in toks:
            if t in self.model:
                vecs.append(self.model[t])
            else:
                t2 = t.replace(" ", "")
                if t2 in self.model:
                    vecs.append(self.model[t2])

        if vecs:
            v = np.mean(np.stack(vecs, axis=0), axis=0)
            v = self._unit(np.asarray(v, dtype=np.float32))
        else:
            v = self._rand_oov(token) if self.oov_mode == "rand" else np.zeros((self.dim,), dtype=np.float32)

        self.cache[token] = v
        return v

    def cosine(self, a: str, b: str) -> float:
        va = self.get_vec(a)
        vb = self.get_vec(b)
        return float(np.dot(va, vb))

    def __call__(self, token: str) -> torch.Tensor:
        v = self.get_vec(token)
        return torch.from_numpy(v).float()
