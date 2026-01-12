#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3C logging utilities.

This module centralizes:
- Episode-level CSV logging
- Sliding-window statistics (SR, return, episode length, sbbox)
- Periodic console printing with approximate FPS

Designed to be lightweight and multiprocessing-safe when used only
from the main process (workers send episode summaries through a queue).
"""

from __future__ import annotations

import os
import csv
import time
from collections import deque
from typing import Dict, Any, List, Optional


class A3CLogger:
    def __init__(
        self,
        log_dir: str = "runs_a3c",
        window_size: int = 100,
        print_every_episodes: int = 100,
        total_frames: int | None = None,
        csv_name: str = "train_log.csv",
        flush_every: int = 1,
        fields: Optional[List[str]] = None,
    ):
        self.log_dir = log_dir
        self.window_size = int(window_size)
        self.print_every_episodes = int(print_every_episodes)
        self.total_frames = None if total_frames is None else int(total_frames)
        self.csv_name = csv_name
        self.flush_every = int(flush_every)

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, self.csv_name)

        self.fields = fields or [
            "global_step",
            "lr"
            "episode_index",
            "episode_return",
            "episode_len",
            "episode_success",
            "best_sbbox",
            "episode_ever_visible",
            "episode_min_dist",
            "scene",
            "goal",
        ]

        self._csv_f = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=self.fields)
        self._csv_w.writeheader()
        self._csv_f.flush()

        self.win = deque(maxlen=self.window_size)
        self.ep_total = 0

        self._last_print_time = time.time()
        self._last_print_steps = 0
        self._since_flush = 0

    def log_episode(self, data: Dict[str, Any], global_step: int | None = None) -> None:
        """Log one finished episode.

        Args:
            data: dict containing episode summary (keys align with self.fields).
            global_step: optionally override data['global_step'] for FPS computation.
        """
        if global_step is not None:
            data = dict(data)
            data["global_step"] = int(global_step)

        # CSV
        row = {k: data.get(k, "") for k in self.fields}
        self._csv_w.writerow(row)
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._csv_f.flush()
            self._since_flush = 0

        # Window stats
        self.win.append(data)
        self.ep_total += 1

        # Console print
        if self.print_every_episodes > 0 and (self.ep_total % self.print_every_episodes == 0) and len(self.win) > 0:
            self._print_window()

    def _print_window(self) -> None:
        now = time.time()
        last = self._last_print_time
        self._last_print_time = now

        # Use latest global_step for FPS
        gs = int(self.win[-1].get("global_step", 0))
        fps = (gs - self._last_print_steps) / max(1e-9, (now - last))
        self._last_print_steps = gs

        avg_ret = sum(float(x.get("episode_return", 0.0)) for x in self.win) / len(self.win)
        sr = sum(int(bool(x.get("episode_success", 0))) for x in self.win) / len(self.win)
        avg_len = sum(float(x.get("episode_len", 0.0)) for x in self.win) / len(self.win)
        avg_sbbox = sum(float(x.get("best_sbbox", 0.0)) for x in self.win) / len(self.win)

        denom = f"{gs}/{self.total_frames}" if self.total_frames is not None else f"{gs}"
        print(
            f"[A3C] steps={denom} fps~{fps:.1f}  "
            f"[SR@{len(self.win)}] {sr:.3f}  ep={self.ep_total}  "
            f"ret={avg_ret:.3f}  len={avg_len:.1f}  sbbox={avg_sbbox:.3f}",
            flush=True,
        )

    def close(self) -> None:
        try:
            self._csv_f.flush()
        except Exception:
            pass
        try:
            self._csv_f.close()
        except Exception:
            pass
        print(f"[CSV] wrote CSV log: {self.csv_path}", flush=True)

