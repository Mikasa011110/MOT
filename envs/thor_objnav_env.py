# envs/thor_objnav_env.py
import gymnasium as gym
import numpy as np
from ai2thor.controller import Controller

from configs import CFG
from models.object_grid import build_object_grid

from ai2thor.controller import Controller

from gymnasium import spaces

class ThorObjNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scenes, targets_by_room, embed, device="cpu", debug=False, headless=False):
        super().__init__()
        self.scenes = scenes # 可用的 THOR 场景列表， reset() 时会随机选一个
        self.targets_by_room = targets_by_room # 每个房间允许的目标物体集合

        self.embed = embed  # Word2Vec 封装，用于构建 object grid（env 端轻量计算）

        # ---- goal vocabulary (string -> id) ----
        all_goals = []
        for _, lst in self.targets_by_room.items():
            all_goals.extend([str(x) for x in lst])
        self.goal_vocab = sorted(list(set(all_goals)))
        self.goal_to_id = {g: i for i, g in enumerate(self.goal_vocab)}
        self.goal_id = 0  # will be set in reset()

        self.device = device

        self.last_success = False # 记录上一个 episode 是否成功
        self.success_count = 0 # 累计成功的 episode 数
        self.episode_count = 0 # 累计的 episode 数

        self.step_count = 0 # 当前 episode 的步数
        self.best_sbbox = 0.0 # 当前 episode 中最大的 S_bbox

        self.scene = None # 当前场景
        self.goal = None # 当前目标物体

        self.headless = headless # 是否使用无头模式

        # 机器人的9个离散动作 𝒜 = {Move Forward, Move Backward, Move Right, Move Left, Rotate Right, Rotate Left, Look Up, Look Down, Done}
        self.action_space = gym.spaces.Discrete(9)
        # Transformers的输出作为观测空间
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(CFG.height, CFG.width, 3), dtype=np.uint8),
            "grid": spaces.Box(low=-1.0, high=1.0, shape=(CFG.grid_size, CFG.grid_size), dtype=np.float32),
            "goal_id": spaces.Discrete(len(self.goal_vocab)),
        })

        self.controller = None 
        self.debug = debug # 是否打印调试信息

    # 用于启动或重启 AI2-THOR 仿真世界
    def _init_controller(self, scene):
        if self.controller is None:
            if self.headless:
                # 启动新的 controller(无头模式)
                self.controller = Controller(
                    scene=scene, # 场景名称
                    width=CFG.width, # 渲染图像宽度
                    height=CFG.height, # 渲染图像高度
                    renderInstanceSegmentation=True, # 需要每一步返回实例分割图, 用于计算目标物体的 bbox
                    renderDepthImage=False, # 不需要深度图
                    # platform=CloudRendering, # 使用云渲染平台，无头模式
                )
            else:
                # 启动新的 controller
                self.controller = Controller(
                    scene=scene, # 场景名称
                    width=CFG.width, # 渲染图像宽度
                    height=CFG.height, # 渲染图像高度
                    renderInstanceSegmentation=True, # 需要每一步返回实例分割图, 用于计算目标物体的 bbox
                    renderDepthImage=False, # 不需要深度图
                )
        else:
            # 重置到新场景
            self.controller.reset(scene=scene)
    
    # 重置环境，开始一个新的 episode
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0 # 重置步数
        self.best_sbbox = 0.0 # 重置最大 S_bbox
        self.last_any_visible = False # 上一步的目标是否可见
        self.last_found = False  # 上一步的目标是否存在
        self.last_best_dist = float("inf") # 上一步agent和goal最短距离
        self.last_success = False # 每个 episode 开始都清零
        self.last_grid_max = 0.0 # 记录上一步的 grid max

        # sample scene + goal (you can refine mapping room type -> targets)
        scene = self.np_random.choice(self.scenes)
        self.scene = scene
        self._init_controller(scene)

        # pick goal based on scene room type
        def _room_type_from_scene(scene_name: str) -> str:
            # FloorPlan1-30: Kitchen
            # FloorPlan201-230: LivingRoom
            # FloorPlan301-330: Bedroom
            # FloorPlan401-430: Bathroom
            n = int(scene_name.replace("FloorPlan", ""))
            if 1 <= n <= 30:
                return "Kitchen"
            if 201 <= n <= 230:
                return "LivingRoom"
            if 301 <= n <= 330:
                return "Bedroom"
            if 401 <= n <= 430:
                return "Bathroom"
            # fallback
            return "Kitchen"
        room_type = _room_type_from_scene(scene)
        candidates = self.targets_by_room[room_type]
        self.goal = str(self.np_random.choice(candidates))

        self.goal_id = int(self.goal_to_id[self.goal])
        # ---- randomize start position (version-stable) ----
        ev = self.controller.step(action="GetReachablePositions")
        if ev.metadata.get("lastActionSuccess") is False:
            raise RuntimeError(f"GetReachablePositions failed: {ev.metadata.get('errorMessage')}")

        positions = ev.metadata["actionReturn"]
        pos = self.np_random.choice(positions)

        # random yaw in multiples of 45 degrees (paper rotation step is 45°)
        yaw = int(self.np_random.choice([0, 45, 90, 135, 180, 225, 270, 315]))

        ev = self.controller.step(
            action="Teleport",
            position=pos,
            rotation={"x": 0, "y": yaw, "z": 0},
        )
        if ev.metadata.get("lastActionSuccess") is False:
            raise RuntimeError(f"Teleport failed: {ev.metadata.get('errorMessage')}")
        # -----------------------------------------------

        # after Teleport, make sure goal exists in this scene
        objects = self.controller.last_event.metadata["objects"]

        def goal_exists(goal: str) -> bool:
            return any(o["objectType"] == goal for o in objects)

        tries = 0
        while (not goal_exists(self.goal)) and tries < 10:
            self.goal = str(self.np_random.choice(candidates))
            self.goal_id = int(self.goal_to_id[self.goal])
            tries += 1

        obs = self._get_obs()
        return obs, {}
    
    def step(self, action_idx):
        """Run one environment step.

        Paper alignment:
        - Action space includes a dedicated Done action.
        - Episode success is ONLY when agent executes Done while the goal is visible and within
          the success distance (paper uses 1.5m).
        - Reward follows Eq.(7):
            +success_reward          if success (Done & visible & close)
            +sbbox_reward            if S_bbox is the highest in the episode so far
            +step_penalty            otherwise
          No extra distance/grid shaping is used here.
        """
        self.step_count += 1
        thor_action = self._map_action(action_idx)
        ev = self.controller.step(action=thor_action["action"], **thor_action.get("kwargs", {}))

        # --- visibility / distance bookkeeping (for metrics, not reward shaping) ---
        success_now, any_visible_now, best_dist_now, found_now = self._is_goal_visible(ev)
        self.last_any_visible = self.last_any_visible or any_visible_now
        self.last_found = self.last_found or found_now
        self.last_best_dist = min(self.last_best_dist, best_dist_now)

        # --- S_bbox (target mask area ratio) ---
        sbbox = self._target_sbbox(ev)
        is_new_sbbox_max = (sbbox > self.best_sbbox + 1e-6)

        # --- reward & termination (paper Eq.(7) + Done semantics) ---
        terminated = False

        if thor_action["action"] == "Done":
            # Done always ends the episode; success only if goal is visible & close at Done time.
            terminated = True
            if success_now:
                reward = float(CFG.success_reward)
                self.last_success = True
            else:
                reward = float(CFG.step_penalty)
        else:
            # Not Done: keep exploring. Reward is either new S_bbox max or step penalty.
            if is_new_sbbox_max:
                # Eq.(7): reward = S_bbox when it's the highest in the episode
                self.best_sbbox = sbbox
                reward = CFG.sbbox_coef * float(sbbox)
            else:
                reward = float(CFG.step_penalty)

        truncated = (self.step_count >= CFG.max_steps)

        # observation after the action
        obs = self._get_obs(ev)

        # info (every step)
        info = {
            "goal_id": np.int64(self.goal_id),
            "sbbox": float(sbbox),
            "best_sbbox": float(self.best_sbbox),
            "any_visible": int(self.last_any_visible),
            "found": int(self.last_found),
            "best_dist": float(self.last_best_dist),
        }

        # episode-end summary fields
        if terminated or truncated:
            info["episode_success"] = int(self.last_success)
            info["episode_len"] = int(self.step_count)
            info["episode_min_dist"] = float(self.last_best_dist)
            info["episode_ever_visible"] = int(self.last_any_visible)

            if self.debug:
                print(
                    f"[END] scene={self.scene} goal={self.goal} "
                    f"len={self.step_count} terminated={terminated} truncated={truncated} "
                    f"best_sbbox={self.best_sbbox:.4f} success={self.last_success}",
                    flush=True,
                )
                print(
                    f"[Debug] any_visible={self.last_any_visible} found={self.last_found} "
                    f"best_dist={self.last_best_dist:.3f}",
                    flush=True,
                )
                print("", flush=True)

        return obs, reward, terminated, truncated, info

    # 从 2D 实例检测结果中提取边界框
    def _bboxes_from_instance_detections2D(self, ev):
        det = getattr(ev, "instance_detections2D", None)
        if det is None:
            # 有的版本放在 ev.metadata
            det = ev.metadata.get("instance_detections2D", None)

        bboxes = {}
        if not det:
            return bboxes

        # det: dict[objId -> np.array([x1,y1,x2,y2])]
        for obj_id, arr in det.items():
            x1, y1, x2, y2 = [float(v) for v in arr]
            # clip 到图像范围，避免越界
            x1 = max(0.0, min(x1, CFG.width - 1))
            y1 = max(0.0, min(y1, CFG.height - 1))
            x2 = max(0.0, min(x2, CFG.width - 1))
            y2 = max(0.0, min(y2, CFG.height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)  # bbox 面积（像素^2）
            bboxes[obj_id] = (x1, y1, x2, y2, float(area))
        return bboxes
    
    # 获取当前观测
    def _get_obs(self, ev=None):
        if ev is None:
            ev = self.controller.last_event

        rgb = ev.frame  # HxWx3 uint8

        # 构建 object_id -> object_type 映射
        object_id_to_type = {o["objectId"]: o["objectType"] for o in ev.metadata["objects"]}

        # 提取 2D 边界框
        bboxes = self._bboxes_from_instance_detections2D(ev)

        # 构建object_grid
        grid, kept, ignored = build_object_grid(
            bboxes=bboxes,
            object_id_to_type=object_id_to_type,
            goal_type=self.goal,
            embed=self.embed,
            img_w=CFG.width,
            img_h=CFG.height,
            grid_size=CFG.grid_size,
        )

        obs = {
            "rgb": rgb,
            "grid": grid.astype(np.float32),
            "goal_id": np.int64(self.goal_id),
        }

        # debug info
        if self.debug and self.step_count % 50 == 0:
            nz = int((grid > 0).sum())
            mx = float(grid.max())
            print(f"[BBOX] n_bboxes={len(bboxes)} kept={kept} ignored={ignored}   [GRID] nonzero={nz} max={mx:.3f} goal={self.goal}")

        return obs

    def _map_action(self, idx):
        # You must implement 9 actions with paper step sizes :contentReference[oaicite:5]{index=5}
        mapping = {
            0: ("MoveAhead", {"moveMagnitude": CFG.move_step}),
            1: ("MoveBack", {"moveMagnitude": CFG.move_step}),
            2: ("MoveRight", {"moveMagnitude": CFG.move_step}),
            3: ("MoveLeft", {"moveMagnitude": CFG.move_step}),
            4: ("RotateRight", {"degrees": CFG.rotate_step}),
            5: ("RotateLeft", {"degrees": CFG.rotate_step}),
            6: ("LookUp", {"degrees": CFG.horizon_step}),
            7: ("LookDown", {"degrees": CFG.horizon_step}),
            8: ("Done", {}),
        }
        a, kw = mapping[int(idx)]
        return {"action": a, "kwargs": kw}

    def _is_goal_visible(self, ev):
        """Check whether the goal object is currently visible and close enough.

        Returns:
            success (bool): True iff (visible == True) and (distance <= success_distance)
            any_visible (bool): True iff any instance of goal is visible (regardless of distance)
            best_dist (float): minimum distance to any instance of goal (visible or not, if available)
            found (bool): True iff any instance of goal exists in metadata
        """
        best_dist = float("inf")
        any_visible = False
        found = False

        # Paper uses 1.5m as the success distance for Done success.
        success_dist = float(getattr(CFG, "success_distance", getattr(CFG, "visible_distance", 1.5)))

        for o in ev.metadata.get("objects", []):
            if o.get("objectType") != self.goal:
                continue

            found = True
            vis = bool(o.get("visible", False))
            dist = float(o.get("distance", float("inf")))

            if dist < best_dist:
                best_dist = dist
            any_visible = any_visible or vis

            if vis and dist <= success_dist:
                return True, True, best_dist, found

        if best_dist == float("inf"):
            best_dist = 999.0

        return False, any_visible, best_dist, found
    
    # 
    def _target_sbbox(self, ev):
        # compute target bbox area ratio from instance segmentation mask
        inst = ev.instance_segmentation_frame
        # quick way: count pixels belonging to any target object (by objectId)
        target_ids = [o["objectId"] for o in ev.metadata["objects"] if o["objectType"] == self.goal]
        if not target_ids:
            return 0.0
        # build set of colors for target ids
        id_to_color = ev.object_id_to_color  # mapping objectId -> "r,g,b" or tuple-like depending on version
        colors = []
        for oid in target_ids:
            c = id_to_color.get(oid, None)
            if c is None:
                continue
            if isinstance(c, str):
                parts = [int(x) for x in c.split(",")]
                colors.append(parts)
            else:
                colors.append(list(c))
        if not colors:
            return 0.0
        colors = np.array(colors, dtype=np.uint8)  # (K,3)
        flat = inst.reshape(-1, 3)
        # membership check
        mask = np.zeros((flat.shape[0],), dtype=bool)
        for c in colors:
            mask |= np.all(flat == c, axis=1)
        area = mask.sum()
        return float(area) / float(CFG.width * CFG.height)

    def close(self):
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
