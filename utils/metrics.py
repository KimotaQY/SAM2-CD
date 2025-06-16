from scipy import stats


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


def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input > expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input < expected_dims:
        np_output = np_input.unsqueeze(0)
    assert len(np_output.shape) == expected_dims
    return np_output


def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = pred >= 0.5
    label = label >= 0.5

    TP = float((pred * label).sum())
    FP = float((pred * (1 - label)).sum())
    FN = float(((1 - pred) * (label)).sum())
    TN = float(((1 - pred) * (1 - label)).sum())
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    IoU = TP / (TP + FP + FN + 1e-10)
    acc = (TP + TN) / (TP + FP + FN + TN)
    F1 = 0
    if acc > 0.999 and TP == 0:
        precision = 1
        recall = 1
        IoU = 1
    if precision > 0 and recall > 0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU
