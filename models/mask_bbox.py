import numpy as np

def mask_to_bboxes(instance_mask: np.ndarray, color_to_object_id: dict):
    """
    用于计算当前图像中每个物体的边界框（bounding box）和像素面积。

    instance_mask: 当前图像的彩色图像蒙版，shape=(H,W,3)，每个像素的RGB值对应一个物体ID
    color_to_object_id: AI2-THOR 提供的映射：颜色 → objectId

    输出: dict[objectId -> (x1,y1,x2,y2, area_pixels)]
        其中 (x1,y1) 是边界框左上角坐标，(x2,y2) 是右下角坐标，area_pixels 是该物体在图像中的像素面积
    """
    h, w, _ = instance_mask.shape # 拿到图像高度和宽度
    flat = instance_mask.reshape(-1, 3)  # 拉平成像素列表 (H*W,3)

    # 找出图像中出现过的所有颜色 uniq为颜色列表, inv为每个像素对应的颜色索引
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)

    bboxes = {} # 用于存储结果
    inv2d = inv.reshape(h, w)  # (H,W)

    # 遍历每种颜色，计算对应物体的边界框
    for idx, color in enumerate(uniq):
        # 跳过背景（黑色）
        if int(color[0]) == 0 and int(color[1]) == 0 and int(color[2]) == 0:
            continue
   
        r, g, b = int(color[0]), int(color[1]), int(color[2]) # 提取RGB值

        # 尝试多种颜色格式以匹配 color_to_object_id
        candidates = [
            f"{r},{g},{b}",
            f"{r} {g} {b}",
            (r, g, b),
            "#{:02x}{:02x}{:02x}".format(r, g, b),
            "#{:02X}{:02X}{:02X}".format(r, g, b),
        ]

        # 找到对应的 objectId
        obj_id = None
        for k in candidates:
            if k in color_to_object_id:
                obj_id = color_to_object_id[k]
                break
        if obj_id is None:
            continue

        # 计算该物体的边界框
        mask = (inv2d == idx)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        
        # 计算边界框坐标和面积
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        area = int(mask.sum())
        bboxes[obj_id] = (int(x1), int(y1), int(x2), int(y2), area)

        '''
        示例输出格式:
        {
            "Apple|+1.23|+0.45|−2.34": (120, 80, 200, 150, 1320),
            "Table|-0.5|0.0|1.2":     (30, 200, 600, 420, 42000),
        }'''

    return bboxes
