#!/usr/bin/env python3
"""Build a small word2vec lookup table for AI2-THOR objects.

Why
----
GoogleNews Word2Vec is ~3.4GB. Loading it inside each SubprocVecEnv worker will explode memory.
This script loads the big model ONCE, collects object tokens from AI2-THOR scenes, computes unit-normalized
300-d vectors, and saves a small .npz table.

After that, train/test use --w2v-table table.npz and never touch the 3.4GB file.

Output
------
A .npz file containing:
  - tokens: (N,) strings
  - vectors: (N, 300) float32 unit vectors

Notes
-----
- OOV tokens are stored as all-zero vectors (stable, avoids random noise).
- We store a few aliases (lowercase, no-space) at runtime in WordEmbed.load_table.
"""

import argparse
import os
import re
from typing import Iterable, List, Set

import numpy as np


# Same mapping as models/word2vec_embed.py
_TOKEN_MAPPING = {
    "tv": "television",
    "tissue": "paper",
}


def split_camel(s: str) -> List[str]:
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s)
    return [p.lower() for p in parts if p.strip()]


def normalize_token(s: str) -> List[str]:
    s = s.replace("_", " ").replace("-", " ").strip()
    if " " in s:
        toks = [t.lower() for t in s.split() if t.strip()]
    else:
        toks = split_camel(s) if any(c.isupper() for c in s) else [s.lower()]
    toks = [_TOKEN_MAPPING.get(t, t) for t in toks]
    return toks


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-8)
    return (v / n).astype(np.float32)


def thor_base_token(obj_type: str) -> str:
    """Match models/object_grid.py parsing.

    Example inputs:
      "Window|-01.04|+01.40|+00.02" -> "Window"
      "FP410:Walls|-3.83|0.29|3.13" -> "Walls"
      "Standard Wall" -> "StandardWall"
    """
    base = obj_type
    if ":" in base:
        base = base.split(":")[-1]
    if "|" in base:
        base = base.split("|")[0]
    tok = base.strip().replace(" ", "")
    tok_low = tok.lower()
    # plural -> singular (same heuristic as object_grid.py)
    tok_sing = tok[:-1] if tok_low.endswith("s") and len(tok) > 3 else tok
    return tok_sing


def collect_tokens_from_targets() -> Set[str]:
    try:
        from envs.scene_split import TARGETS
    except Exception:
        return set()

    vocab: Set[str] = set()
    for room, lst in dict(TARGETS).items():
        for t in lst:
            vocab.add(str(t))
    return vocab


def collect_tokens_from_scenes(scenes: Iterable[str], headless: bool) -> Set[str]:
    from ai2thor.controller import Controller

    controller = Controller(
        headless=headless,
        renderInstanceSegmentation=False,
        renderDepthImage=False,
        width=300,
        height=300,
    )

    vocab: Set[str] = set()

    try:
        for i, scene in enumerate(scenes, 1):
            if i % 10 == 0 or i == 1 or i == len(list(scenes)):
                print(f"[SCENE] scanning {i}/{len(list(scenes))}: {scene}")

            ev = controller.reset(scene)
            objs = ev.metadata.get("objects", [])
            for o in objs:
                ot = o.get("objectType")
                if ot:
                    vocab.add(thor_base_token(str(ot)))
    finally:
        controller.stop()

    return vocab


def default_scene_list(which: str) -> List[str]:
    which = which.lower()
    if which == "kitchens20":
        return [f"FloorPlan{i}" for i in range(1, 21)]
    if which == "paper120":
        # Common AI2-THOR split: kitchens 1-30, living 201-230, bedrooms 301-330, bathrooms 401-430
        scenes: List[str] = []
        scenes += [f"FloorPlan{i}" for i in range(1, 31)]
        scenes += [f"FloorPlan{i}" for i in range(201, 231)]
        scenes += [f"FloorPlan{i}" for i in range(301, 331)]
        scenes += [f"FloorPlan{i}" for i in range(401, 431)]
        return scenes
    # fallback: user provides a comma-separated list
    return [s.strip() for s in which.split(",") if s.strip()]


def build_table(vocab: List[str], w2v_path: str, oov_zero: bool = True) -> np.ndarray:
    from gensim.models import KeyedVectors

    print(f"[W2V] loading heavy model: {w2v_path}")
    kv = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    dim = int(kv.vector_size)
    assert dim == 300, f"Expected 300d vectors, got {dim}"

    vectors = np.zeros((len(vocab), dim), dtype=np.float32)
    N = len(vocab)

    for i, token in enumerate(vocab):
        if i % 20 == 0 or i == N - 1:
            print(f"[W2V] computing {i+1}/{N}: {token}")
        toks = normalize_token(token)
        vecs = []
        for t in toks:
            if t in kv:
                vecs.append(kv[t])
            else:
                t2 = t.replace(" ", "")
                if t2 in kv:
                    vecs.append(kv[t2])
        if vecs:
            v = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
            vectors[i] = unit(v)
        else:
            if oov_zero:
                vectors[i] = np.zeros((dim,), dtype=np.float32)
            else:
                rng = np.random.default_rng(abs(hash(token)) % (2**32))
                vectors[i] = unit(rng.normal(size=(dim,)).astype(np.float32))

    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w2v-path", type=str, required=True, help="Path to GoogleNews word2vec (3.4GB) file")
    parser.add_argument("--out", type=str, required=True, help="Output .npz table path")
    parser.add_argument(
        "--scenes",
        type=str,
        default="paper120",
        help="Which scenes to scan: kitchens20 | paper120 | or comma-separated list",
    )
    parser.add_argument("--headless", action="store_true", help="Use AI2-THOR headless controller")
    parser.add_argument("--include-targets", action="store_true", help="Also include TARGETS (paper goals)")
    args = parser.parse_args()

    import os
    assert os.path.isfile(args.w2v_path), f"W2V file not found: {args.w2v_path}"
    print(f"[OK] Using W2V from {args.w2v_path}")

    scenes = default_scene_list(args.scenes)
    print(f"[SCENES] count={len(scenes)} sample={scenes[:5]}")

    vocab: Set[str] = set()

    if args.include_targets:
        tset = collect_tokens_from_targets()
        vocab |= tset
        print(f"[VOCAB] added targets: {len(tset)}")

    sset = collect_tokens_from_scenes(scenes, headless=args.headless)
    vocab |= sset
    print(f"[VOCAB] added scene objects: {len(sset)}")

    vocab_list = sorted(vocab)
    print(f"[VOCAB] total unique tokens={len(vocab_list)}")

    vectors = build_table(vocab_list, args.w2v_path, oov_zero=True)

    np.savez_compressed(args.out, tokens=np.array(vocab_list, dtype=object), vectors=vectors)
    print(f"[SAVE] wrote {args.out} (tokens={len(vocab_list)}, dim={vectors.shape[1]})")


if __name__ == "__main__":
    main()
