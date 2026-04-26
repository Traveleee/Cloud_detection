import os
import numpy as np
import tifffile
import torch
import pandas as pd
from PIL import Image
from scripts.calculate_metrics import calculate_metrics
from scripts.get_patches import extract_unique_sceneids,\
    stitch_patches, remove_padding


# 配置参数
gt_folder = "95-Cloud dataset/Test/Entire_scene_gts"
pred_folder_root = "95-Cloud dataset/Predictions/U-Net_train_96_test_192/U-Net_best"  # 预测文件父级目录
pred_folder_save = "95-Cloud dataset/Predictions/U-Net_train_96_test_192/U-Net_complete_mask"
max_pix = 255
pr_patch_size = 192, 192
classes = [0, 1]  # 0：北京，1：云


def main():
    # 加载补丁场景ID列表
    all_uniq_sceneid = extract_unique_sceneids(pred_folder_root)
    # print("提取到的场景ID列表:", all_uniq_sceneid)  # 检查是否包含多个不同ID
    results = []

    for scene_id in all_uniq_sceneid:
        # 加载GT
        gt_file = os.path.join(gt_folder, f'edited_corrected_gts_{scene_id}.TIF')
        print("当前场景ID:", scene_id)
        # print("GT文件路径:", gt_file)  # 检查路径是否随scene_id变化
        # print("预测补丁路径:", os.path.join(pred_folder_root, f"scene_{scene_id}"))  # 示例路径
        gt = np.array(Image.open(gt_file))
        gt = gt.astype(np.uint8)  # 确保GT为整数类型

        # 生成完整掩膜
        pred_mask = stitch_patches(pred_folder_root, scene_id, pr_patch_size, resize=True)
        # 对齐尺寸
        pred_mask = remove_padding(pred_mask, gt.shape)

        ######################################################
        # 关键修复：对预测掩膜进行二值化
        # 假设模型输出的是 logits（含负数）
        pred_mask_sigmoid = 1 / (1 + np.exp(-pred_mask))  # Sigmoid转概率
        pred_mask_b = (pred_mask_sigmoid > 0.5).astype(np.uint8)  # 二值化
        ######################################################

        # # 检查形状是否一致
        # print(f"pred_mask shape: {pred_mask.shape}, gt shape: {gt.shape}")  # 必须完全相同
        #
        # # 检查像素唯一值
        # print("预测掩膜唯一值:", np.unique(pred_mask))
        # print("真实标签唯一值:", np.unique(gt))
        #
        # # 检查是否有 NaN
        # print("预测掩膜 NaN 数量:", np.isnan(pred_mask).sum())
        # print("真实标签 NaN 数量:", np.isnan(gt).sum())

        # 计算评估指标
        metrics = calculate_metrics(pred_mask_b, gt)

        # 保存完整掩膜
        output_dir = os.path.join(pred_folder_save, "complete_masks")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"LC{scene_id}.tif")
        # 将概率图转换为0-max_pix的uint8格式
        pred_mask_unit8 = (pred_mask_sigmoid * max_pix).astype(np.uint8)
        tifffile.imwrite(output_path, pred_mask_unit8)

        # 记录结果
        results.append([f"LC{scene_id}"] + metrics)
        print(f"处理完成：Precision={metrics[1]:.2f}%, Recall={metrics[2]:.2f}%")

    # 生成报告
    if results:
        df = pd.DataFrame(results, columns=["SceneID", "Accuracy", "Precision",
                                            "Recall", "F1-Score", "iou"])

        # 保存详细结果
        csv_path = os.path.join(pred_folder_save, "U-Net_metrics_results.csv")
        df.to_csv(csv_path, index=False)

        # 生成统计摘要
        summary = df.mean(numeric_only=True)
        txt_content = f"""U-Net_定量评估报告 ({pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')})
------------------------------------------------
平均指标：
- Overall Accuracy: {summary['Accuracy']:.2f}%
- Precision: {summary['Precision']:.2f}%
- Recall:    {summary['Recall']:.2f}%
- F1-Score: {summary['F1-Score']:.2f}%
- IoU Index: {summary['iou']:.2f}%

评估参数：
- 处理场景数: {len(results)}/{len(all_uniq_sceneid)}
- 补丁尺寸: {pr_patch_size[0]}x{pr_patch_size[1]}
"""
        with open(os.path.join(pred_folder_save, "U-Net_summary_report.txt"), "w") as f:
            f.write(txt_content)

        print("\n处理完成，结果已保存至：")
        print(f"- 定量结果: {csv_path}")
        print(f"- 汇总报告: {pred_folder_save}/U-Net_summary_report.txt")


if __name__ == "__main__":
    main()