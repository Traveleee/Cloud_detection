import re
import os
import numpy as np
from PIL import Image


# 提取指定目录下所有预测结果文件的唯一场景ID
def extract_unique_sceneids(preds_dir_root):
    # 构造完整路径
    path_4landtype = preds_dir_root

    # 获取目录下所有文件项（过滤隐藏文件和非TIF文件）
    folders = [f.name for f in os.scandir(path_4landtype)
               if f.is_file() and f.name.upper().endswith(".TIF")]

    scene_ids = []
    for filename in folders:
        # 去除文件扩展名（兼容大小写）
        clean_name = re.sub(r'\.tif$', '', filename, flags=re.IGNORECASE)

        # 使用正则定位LC起始位置
        lc_match = re.search(r'LC', clean_name)

        # 提取场景ID部分（从LC到字符串结尾）
        scene_id = clean_name[lc_match.start():]
        scene_ids.append(scene_id)

    # set()去重,并转换为PyTorch张量
    unique_ids = sorted(list(set(scene_ids)))
    return unique_ids


def stitch_patches(pred_root, scene_id, patch_size, resize=True):
    """拼接预测补丁"""
    # 初始化完整掩膜
    max_row = max_col = 0
    patches = []

    # 遍历所有补丁文件
    for f in os.listdir(pred_root):
        if scene_id in f and f.endswith('.TIF'):
            # 解析行列号
            row, col = extract_rowcol_each_patch(f)
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            patches.append((row, col, f))

    # 统一初始化 full_mask 的逻辑
    if resize:
        target_size = (384, 384)  # 硬编码目标尺寸
        full_mask = np.zeros((max_row * target_size[0], max_col * target_size[1]))
    else:
        full_mask = np.zeros((max_row * patch_size[0], max_col * patch_size[1]))

    # 填充补丁到 full_mask
    for row, col, fname in patches:
        patch = Image.open(os.path.join(pred_root, fname))

        if resize:
            # 调整补丁尺寸到 384x384
            patch = patch.resize((384, 384), Image.Resampling.NEAREST)
            y_start = (row - 1) * 384
            x_start = (col - 1) * 384
            patch_size_used = 384  # 当前使用的补丁尺寸
        else:
            # 使用原始尺寸
            y_start = (row - 1) * patch_size[0]
            x_start = (col - 1) * patch_size[1]
            patch_size_used = patch_size[0]  # 假设 patch_size 是正方形

        patch = np.array(patch)
        full_mask[y_start:y_start + patch_size_used, x_start:x_start + patch_size_used] = patch

    return full_mask


def remove_padding(pred_mask, gt_shape):
    """去除零填充"""
    y_center = (pred_mask.shape[0] - gt_shape[0]) // 2
    x_center = (pred_mask.shape[1] - gt_shape[1]) // 2
    return pred_mask[y_center:y_center + gt_shape[0], x_center:x_center + gt_shape[1]]


def extract_rowcol_each_patch(name):
    # 移除文件扩展名（兼容大小写）
    clean_name = name.replace('.TIF', '').replace('.tif', '')

    # 使用正则表达式定位关键标记
    lc_match = re.search(r'LC', clean_name)
    h_match = re.search(r'h_', clean_name)

    # 提取中间段（兼容MATLAB索引差异）
    patchbad = clean_name[h_match.end():lc_match.start() - 1]

    # 分割数字部分（增强容错性）
    numbers = list(map(int, re.findall(r'\d+', patchbad)))

    row = numbers[1]  # 第二个数字
    col = numbers[2]  # 第三个数字

    return row, col
