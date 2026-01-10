# models/word2vec_embed.py
import re
import numpy as np
import torch

class WordEmbed:
    """
    Word2Vec 词向量封装类，用于计算目标类别与物体类别之间的语义相似度。

    设计目的：
    - 为 Object Navigation / ObjectGrid 提供稳定的语义先验
    - 与 Wortsman et al. (CVPR 2019) 及 OMT 论文中的设定保持一致
    - 支持 AI2-THOR 中常见的 CamelCase 物体名称
    - 对 OOV（词表外词）具有鲁棒处理能力

    特点：
    - 使用固定（不训练）的 300 维词向量
    - 支持本地加载 GoogleNews Word2Vec
    - 对拆分后的子词进行平均
    - 完全 OOV 时使用确定性的随机向量作为回退
    """

    def __init__(
        self,
        # model_name: str = "GoogleNews-vectors-negative300",
        model_name: str = "word2vec-google-news-300",
        local_path: str | None = None
    ):
        """
        初始化 Word2Vec 词向量模型。

        Args:
            model_name (str):
                词向量模型名称（主要用于记录，与论文对齐）
            local_path (str | None):
                本地词向量文件路径（推荐使用 GoogleNews-vectors-negative300.bin）
        """
        self.dim = 300
        self.model = None
        self.cache = {}  # 缓存：token -> 向量，避免重复计算

        try:
            import gensim
            from gensim.models import KeyedVectors
            import gensim.downloader as api
        except Exception as e:
            raise ImportError(
                "需要安装 gensim 才能使用 Word2Vec / GloVe 词向量。"
                "请执行：pip install gensim"
            ) from e

        if local_path is not None:
            # -------------------------------
            # 从本地加载词向量（推荐做法）
            # -------------------------------
            # 支持两种常见格式：
            # 1) gensim 原生 .kv
            # 2) word2vec binary / text 格式（如 GoogleNews .bin）
            from gensim.models import KeyedVectors

            if local_path.endswith(".kv"):
                self.model = KeyedVectors.load(local_path, mmap="r")
            else:
                # 默认按 word2vec 格式加载
                self.model = KeyedVectors.load_word2vec_format(
                    local_path,
                    binary=local_path.endswith(".bin")
                )

            self.dim = int(self.model.vector_size)

        else:
            # -------------------------------
            # 使用 gensim downloader 自动下载（需要联网）
            # 注意：GoogleNews 通常无法自动下载
            # -------------------------------
            import gensim.downloader as api
            self.model = api.load(model_name)
            self.dim = int(self.model.vector_size)

    # ============================================================
    # 文本预处理相关函数
    # ============================================================

    def _split_camel(self, s: str) -> list[str]:
        """
        拆分 CamelCase 形式的字符串。
        例如：
            "CoffeeMachine" -> ["coffee", "machine"]
        """
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s)
        return [p.lower() for p in parts if p.strip()]

    def _normalize(self, s: str) -> list[str]:
        """
        对物体 / 目标名称进行规范化和分词。

        处理逻辑：
        - 替换下划线和连字符
        - 优先按空格分词
        - 否则尝试 CamelCase 拆分
        - 对 AI2-THOR 中常见名称做轻微修正
        """
        s = s.replace("_", " ").replace("-", " ").strip()

        if " " in s:
            toks = [t.lower() for t in s.split() if t.strip()]
        else:
            toks = self._split_camel(s) if any(c.isupper() for c in s) else [s.lower()]

        # 针对 THOR 中常见名称的简单映射修正
        mapping = {
            "tv": "television",
            "tissue": "paper",
        }
        toks = [mapping.get(t, t) for t in toks]
        return toks

    # ============================================================
    # 向量工具函数
    # ============================================================

    def _unit(self, v: np.ndarray) -> np.ndarray:
        """
        向量单位化（L2 norm = 1），用于稳定 cosine similarity。
        """
        n = np.linalg.norm(v) + 1e-8
        return (v / n).astype(np.float32)

    def _rand_oov(self, key: str) -> np.ndarray:
        """
        对完全 OOV 的 token 生成一个确定性的随机向量。

        说明：
        - 使用 token 的 hash 作为随机种子
        - 保证同一个 token 在不同时间返回相同向量
        - 避免 RL 训练中引入不稳定噪声
        """
        rng = np.random.default_rng(abs(hash(key)) % (2**32))
        v = rng.normal(size=(self.dim,)).astype(np.float32)
        return self._unit(v)

    # ============================================================
    # 对外接口
    # ============================================================

    def get_vec(self, token: str) -> np.ndarray:
        """
        获取某个物体 / 目标名称的词向量表示。

        Args:
            token (str):
                物体或目标类别名称，如 "CoffeeMachine"

        Returns:
            v (ndarray):
                单位化后的 300 维词向量
        """
        if token in self.cache:
            return self.cache[token]

        toks = self._normalize(token)
        vecs = []

        for t in toks:
            if t in self.model:
                vecs.append(self.model[t])
            else:
                # 尝试去掉空格的变体
                t2 = t.replace(" ", "")
                if t2 in self.model:
                    vecs.append(self.model[t2])

        if vecs:
            # 多个子词向量取平均
            v = np.mean(np.stack(vecs, axis=0), axis=0)
            v = self._unit(np.asarray(v, dtype=np.float32))
        else:
            # 完全 OOV：使用确定性随机向量
            v = self._rand_oov(token)

        self.cache[token] = v
        return v

    def cosine(self, a: str, b: str) -> float:
        """
        计算两个类别名称之间的 cosine 相似度。

        Args:
            a (str): 物体类别
            b (str): 目标类别

        Returns:
            float: cosine similarity（[-1, 1]）
        """
        va = self.get_vec(a)
        vb = self.get_vec(b)
        return float(np.dot(va, vb))
    
    def __call__(self, token: str) -> torch.Tensor:
        """
        让 WordEmbed 实例可直接被调用：embed(token) -> torch.Tensor

        Args:
            token (str): 物体类别 / 目标类别名称，如 "CoffeeMachine"

        Returns:
            torch.Tensor: 单位化后的词向量，形状 (dim,)
        """
        v = self.get_vec(token)                 # numpy (dim,)
        return torch.from_numpy(v).float()      # torch (dim,)

