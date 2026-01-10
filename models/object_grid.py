# models/object_grid.py
'''
目标：
    把当前帧检测到的物体 bbox（2D 框）映射成一个 grid_size × grid_size 的二维网格（16×16）。
    每个检测到的物体会落在网格的某个 cell（根据 bbox 中心点）   
    cell 的值不是 1/0，而是一个相似度分数 sim（通常来自“物体类别”和“目标类别 goal_type”的文本相似度，比如 CLIP embedding cosine）
    如果一个 cell 里有多个物体，就取最大相似度（max pooling）
    同时过滤掉 THOR 里常见的结构类物体（墙、地板、门……），避免它们污染网格

输出：
    grid: shape (grid_size, grid_size)，float32
    kept: 参与建图的物体数
    ignored: 被过滤掉的结构物体数
'''
import numpy as np

def build_object_grid(
    bboxes: dict,
    object_id_to_type: dict,
    goal_type: str,
    embed, 
    img_w: int, 
    img_h: int, # 图像宽高（用于归一化坐标）
    grid_size: int = 16, # 网格分辨率
):
    """
    bboxes: dict[obj_id -> (x1,y1,x2,y2, area)]
    object_id_to_type: dict[obj_id -> objectType]
    Returns:
        grid: (grid_size, grid_size) float32
        kept: number of objects kept for grid
        ignored: number of objects filtered out
    """
    # 过滤列表
    IGNORE_TYPES = {
        "Floor", "Ceiling", "Walls", "StandardWallSize",
        "Door", "DoorFrame", "Window", "Blinds",
    }
    IGNORE_LOW = {t.lower() for t in IGNORE_TYPES} # 小写形式

    kept = 0
    ignored = 0
    grid = np.zeros((grid_size, grid_size), dtype=np.float32) # 所有网格初始为0，表示没看到任何有用物体

    for obj_id, (x1, y1, x2, y2, _) in bboxes.items():
        obj_type = object_id_to_type.get(obj_id, None)
        if obj_type is None:
            continue
        
        # objectType 一般为 Window|-01.04|+01.40|+00.02
        # 把 THOR 的 objectType / name 里那些数字和符号去掉，留下纯名称部分
        base = obj_type
        if ":" in base:
            base = base.split(":")[-1]
        if "|" in base:
            base = base.split("|")[0]

        # 把空格去掉
        tok = base.strip()
        tok = tok.replace(" ", "")  # e.g. "Standard Wall" -> "StandardWall"
        tok_low = tok.lower()

        # 把复数变为单数形式
        tok_sing = tok[:-1] if tok_low.endswith("s") and len(tok) > 3 else tok
        tok_sing_low = tok_sing.lower()


        if tok_low in IGNORE_LOW or tok_sing_low in IGNORE_LOW:
            ignored += 1 # 统计 ignored
            continue
        kept += 1 # 统计 kept

        # -----------------------------------------------

        cx = (x1 + x2) / 2.0 # bbox 中心点（像素坐标）
        cy = (y1 + y2) / 2.0 # bbox 中心点（像素坐标）
        gi = int(np.clip(cy / max(img_h, 1) * grid_size, 0, grid_size - 1)) # 映射到 grid cell 坐标
        gj = int(np.clip(cx / max(img_w, 1) * grid_size, 0, grid_size - 1)) # 映射到 grid cell 坐标

        sim = float(embed.cosine(tok_sing, goal_type)) # 计算目标类别和物体类别的文本相似度， sim 越大，说明“这个格子里出现的东西越像目标”
        # sim = max(0.0, sim)           # 把负相似度 clamp 到 0 (原论文没这么做)
        grid[gi, gj] = max(grid[gi, gj], sim)

    return grid, kept, ignored        
