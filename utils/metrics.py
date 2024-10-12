import torch

def iou(preds, targets, threshold=0.5):
    """
    计算IoU（交并比）。

    参数:
    - preds (torch.Tensor): 预测值张量。
    - targets (torch.Tensor): 真实标签张量。
    - threshold (float): 用于二值化预测值的阈值。

    返回:
    - float: IoU值。
    """
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    iou = intersection / union
    return iou.item()

def f1_score(preds, targets, threshold=0.5):
    """
    计算F1-Score。

    参数:
    - preds (torch.Tensor): 预测值张量。
    - targets (torch.Tensor): 真实标签张量。
    - threshold (float): 用于二值化预测值的阈值。

    返回:
    - float: F1-Score值。
    """
    preds = (preds > threshold).float()
    tp = (preds * targets).sum()
    precision = tp / (preds.sum() + 1e-7)
    recall = tp / (targets.sum() + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    return f1.item()
