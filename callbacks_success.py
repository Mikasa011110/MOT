import csv
from collections import deque
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback

'''SR: Success Rate 最近100个episode的成功率'''


class SuccessLoggerCallback(BaseCallback):
    """
    Log episode-level success to CSV and print rolling success rate.
    Expects env to put `episode_success` (0/1) into info when done=True.
    """

    def __init__(self, out_dir="logs", window=100, verbose=1):
        super().__init__(verbose)
        self.out_dir = Path(out_dir)
        self.window = int(window)

        self._succ_window = deque(maxlen=self.window)

        self._csv_path = None
        self._csv_file = None
        self._writer = None

        self.ep_idx = 0
        self.last_print_ep = 0

    def _on_training_start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.out_dir / "episode_metrics.csv"
        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "episode",
                "timesteps",
                "success",
                "episode_len",
                "min_dist",
                "ever_visible",
                "rolling_sr",
            ],
        )
        self._writer.writeheader()
        self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
    
        # ✅ 兼容 Gymnasium：terminated/truncated
        if dones is None:
            terminated = self.locals.get("terminated", [])
            truncated = self.locals.get("truncated", [])
            dones = [bool(t or tr) for t, tr in zip(terminated, truncated)]
    
        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue
            if "episode_success" not in info:
                continue
            
            self.ep_idx += 1
            success = int(info.get("episode_success", 0))
            ep_len = int(info.get("episode_len", 0))
            min_dist = float(info.get("episode_min_dist", float("nan")))
            ever_vis = int(info.get("episode_ever_visible", 0))
    
            self._succ_window.append(success)
            rolling_sr = sum(self._succ_window) / len(self._succ_window)
    
            self._writer.writerow({
                "episode": self.ep_idx,
                "timesteps": int(self.num_timesteps),
                "success": success,
                "episode_len": ep_len,
                "min_dist": min_dist,
                "ever_visible": ever_vis,
                "rolling_sr": rolling_sr,
            })
    
            # ✅ 更稳：每条都 flush（你也可以改成每 5 条/10 条）
            self._csv_file.flush()
    
            # ✅ 打印：每 20 个 episode 打一次
            if self.verbose and (self.ep_idx % 20 == 0):
                print(f"[Rolling] SR@{len(self._succ_window)}={rolling_sr:.2f} (eps={self.ep_idx})", flush=True)
    
        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
