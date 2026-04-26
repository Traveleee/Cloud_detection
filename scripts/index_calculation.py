import os
import numpy as np
import tifffile as tiff


# 读取路径、名称
GLOBAL_PATH = '../95-Cloud dataset'
TEST_FOLDER = os.path.join(GLOBAL_PATH, 'Test')
TRAIN_FOLDER = os.path.join(GLOBAL_PATH, 'Training')
PRED_FOLDER = os.path.join(GLOBAL_PATH, 'Predictions')
error = 1e-7
predictions_path = os.path.join(PRED_FOLDER,
                           'Cloud-Detection_outputs_of_prediction/'
                           'patch_75_4_by_15_LC08_L1TP_002054_20160520_20170324_01_T1.TIF')

targets_path = os.path.join(TRAIN_FOLDER,
                            'train_gts/gt_patch_75_4_by_15_LC08_L1TP_002054_20160520_20170324_01_T1.TIF')


# 读取预测结果和真实标签
predictions = tiff.imread(predictions_path)
targets = tiff.imread(targets_path)
predictions = (predictions > 0).astype(int)
targets = (targets > 0).astype(int)
# # 检查数据中类别标签的分布
# print("Prediction unique values:", np.unique(predictions))
# print("Target unique values:", np.unique(targets))


def calculate_metrics(predictions, targets):
    # 初始化混淆矩阵的元素
    tp = fp = tn = fn = 0

    for i in range(predictions.shape[0]):
        pred = predictions[i].flatten().astype(int)  # 确保为整数
        target = targets[i].flatten().astype(int)    # 确保为整数

        # 计算混淆矩阵的元素
        tp += np.sum((pred == 1) & (target == 1))
        tn += np.sum((pred == 0) & (target == 0))
        fp += np.sum((pred == 1) & (target == 0))
        fn += np.sum((pred == 0) & (target == 1))

    # 计算总体精度
    overall_accuracy = (tp + tn) / (tp + tn + fp + fn )
    # 计算精确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # 计算召回率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # 计算F1-Score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # 计算平均交并比（Mean IoU）
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    mean_iou = iou  # 只有云这一个待检测类别

    return overall_accuracy, precision, recall, f1_score, mean_iou


# 调用计算指标函数
overall_accuracy, precision, recall, f1_score, mean_iou = calculate_metrics(predictions, targets)

print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
