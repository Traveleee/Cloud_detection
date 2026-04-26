import numpy as np


def calculate_metrics(pred, gt):
    """计算评估指标"""
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1_score = (2 * accuracy * recall) / (accuracy + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)

    return [accuracy * 100, precision * 100, recall * 100, f1_score * 100, iou * 100]
