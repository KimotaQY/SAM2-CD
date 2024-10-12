import torch
import torch.nn as nn

from torch.nn.functional import threshold, normalize
import torch.nn.functional as F

# Binary Cross-Entropy Loss
bce_loss = nn.BCELoss()

# Dice Loss
def dice_loss(pred, target, smooth=1.0):
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
 
    return 1 - ((2. * intersection + smooth) /
              (pred_flat.sum() + target_flat.sum() + smooth))

# 组合损失函数
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = dice_loss(pred, target)
        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCELoss(reduction='none')  # 不要先平均

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def iou_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    total = (pred + target).sum(dim=2).sum(dim=2)
    union = total - intersection    
    IoU = (intersection + smooth) / (union + smooth)    
    return 1 - IoU.mean()


def mean_iou(preds, labels, eps=1e-6):
    # Assuming preds shape is [batch_size, 1, H, W] and labels is [batch_size, H, W]
    preds = preds.squeeze(1)  # Remove the channel dimension if it's 1
    # preds = torch.sigmoid(preds)  # Ensure preds are in the range [0, 1]
    
    pred_cls = (preds >= 0.5).float()  # Thresholding at 0.5 to get binary predictions
    label_cls = (labels == 1).float()

    intersection = (pred_cls * label_cls).sum((1, 2))  # Sum over H and W dimensions
    union = pred_cls.sum((1, 2)) + label_cls.sum((1, 2)) - intersection  # Union calculation

    ious = (intersection + eps) / (union + eps)  # Avoid division by zero
 
    return ious


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2, alpha=1, gamma=2):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        pred = F.sigmoid(pred).squeeze(1).to(dtype = torch.float64)
        bce = self.bce_loss(pred, target)
        dice = dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        loss = self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal
        return loss
