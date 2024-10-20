import csv
import json
import os
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from datetime import datetime
import yaml
from models.build_sam import build_sam2
from datasets.CustomDataset import build_dataloader

import torch.nn.functional as F
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter
from utils.heatmap import grad_cam


# 读取配置
with open('./configs/config_test.yaml', "r", encoding='utf-8') as file:
    config_data = yaml.safe_load(file)


DATA_TYPE = config_data["data"]["type"]
NET_NAME = "SAM2_" + DATA_TYPE
TASK_TYPE = 'test'


# import matplotlib.pyplot as plt
def visualize_batch(images_a, images_b, masks):
    input_A_np = images_a.cpu().numpy().transpose(0, 2, 3, 1)
    input_B_np = images_b.cpu().numpy().transpose(0, 2, 3, 1)
    mask_np = masks.cpu().numpy().transpose(0, 2, 3, 1)

    # 可视化前后时相及其对应的mask
    for i in range(min(4, images_a.size(0))):  # 打印前4个图像样本
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow((input_A_np[i] * 0.5) + 0.5)  # 反归一化
        axs[0].set_title('Image A (T1)')
        axs[0].axis('off')

        axs[1].imshow((input_B_np[i] * 0.5) + 0.5)  # 反归一化
        axs[1].set_title('Image B (T2)')
        axs[1].axis('off')

        axs[2].imshow(mask_np[i].squeeze(), cmap='gray')
        axs[2].set_title('Mask')
        axs[2].axis('off')

        plt.show()


def FocalLoss(inputs, targets, alpha=0.25, gamma=2):
    # inputs = F.sigmoid(inputs)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE
    return focal_loss.mean()


def BCEDiceLoss(inputs, targets):
    # print(inputs.shape, targets.shape)
    inputs = F.sigmoid(inputs)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    # focal = FocalLoss(inputs, targets)  # BCEDiceFocalLoss
    return bce + 1 - dice

def set_seed(seed):
    random.seed(seed)  # 设置 Python 内部的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子（CPU）
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的随机种子（单 GPU）
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU 设置所有 GPU 的随机种子

    # 确保 CuDNN 的确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    train_opt = config_data["training"]

    SEED = train_opt['seed']
    # set_seed(SEED)

    # 构建模型
    model_opt = config_data["model"]
    # checkpoint_path = model_opt["checkpoint_path"]
    checkpoint_path = model_opt["checkpoint_path"]
    model_cfg = model_opt["config"]
    model = build_sam2(model_cfg, checkpoint_path)
    
    # print("可训练参数:")
    # for name, param in sam2.named_parameters():
    #     # if param.requires_grad:
    #     #     print(f"参数名: {name}, 尺寸: {param.size()}")
    #     if any(keyword in name for keyword in ['down_channel', 'soft_ffn', 'mask_decoder', 'kan', 'dynamic_map_gen']):
    #         param.requires_grad = True
    #         # print(f"参数名: {name}, 尺寸: {param.requires_grad}")
    #     print(f"参数名: {name}, 尺寸: {param.requires_grad}")

    file_path = config_data["data"][DATA_TYPE]
    global TASK_TYPE
    TASK_TYPE = 'test' if 'test' in file_path else 'train'

    # dataloaders
    batch_size = train_opt["batch_size"]
    dataloaders = build_dataloader(file_path, batch_size, train_opt['num_workers'])
    val_loader = dataloaders['test']
    
    val_F, val_acc, val_IoU, val_loss, val_pre, val_rec = validate(val_loader, model)
    # test(val_loader, model)

    # 查找模型中所有的子模块
    # for name, module in model.named_modules():
    #     print(name, module)


def validate(val_loader, model):
    model.eval()
    torch.cuda.empty_cache()

    train_opt = config_data["training"]

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    Pre_meter = AverageMeter()
    Rec_meter = AverageMeter()

    iterations = tqdm(val_loader)
    for valid_data in iterations:
        valid_input_A = valid_data['image_A'].to(torch.device('cuda', int(train_opt['dev_id']))).float()
        valid_input_B = valid_data['image_B'].to(torch.device('cuda', int(train_opt['dev_id']))).float()
        labels = valid_data['mask'].to(torch.device('cuda', int(train_opt['dev_id']))).float()

        # 可视化前后时相及其对应的mask
        # visualize_batch(valid_input_A, valid_input_B, labels)

        valid_input = torch.cat((valid_input_A, valid_input_B), dim=0)
        with torch.no_grad():
            # outputs = model(valid_input)
            # # 上采样输出到标签尺寸
            # outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            # loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.tensor([10]).to(torch.device('cuda', int(train_opt['dev_id']))))
            outputs, outputs_2, outputs_3 = model(valid_input)
            loss = BCEDiceLoss(outputs, labels) + BCEDiceLoss(outputs_2, labels) + BCEDiceLoss(outputs_3, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        preds = F.sigmoid(outputs).numpy()
        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
            Pre_meter.update(precision)
            Rec_meter.update(recall)

        pbar_desc = "Model valid loss --- "
        pbar_desc += f"loss: {val_loss.average():.5f}"
        pbar_desc += f", F1: {F1_meter.avg * 100:.2f}"
        pbar_desc += f", mIOU: {IoU_meter.avg * 100:.2f}"
        pbar_desc += f", Acc: {Acc_meter.avg * 100:.2f}"
        pbar_desc += f", Pre: {Pre_meter.avg * 100:.2f}"
        pbar_desc += f", Rec: {Rec_meter.avg * 100:.2f}"
        iterations.set_description(pbar_desc)

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg, Pre_meter.avg, Rec_meter.avg

import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, \
                            ScoreCAM, \
                            GradCAMPlusPlus, \
                            AblationCAM, \
                            XGradCAM, \
                            EigenCAM, \
                            EigenGradCAM, \
                            LayerCAM, \
                            FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def test(val_loader, model):
    train_opt = config_data["training"]

    iterations = tqdm(val_loader)
    for valid_data in iterations:
        valid_input_A = valid_data['image_A'].to(torch.device('cuda', int(train_opt['dev_id']))).float()
        valid_input_B = valid_data['image_B'].to(torch.device('cuda', int(train_opt['dev_id']))).float()
        labels = valid_data['mask'].to(torch.device('cuda', int(train_opt['dev_id']))).float()
        valid_input = torch.cat((valid_input_A, valid_input_B), dim=0)

        

        # attention_map = grad_cam(model, valid_input, model.mask_decoder.frm)
        
        # # 使用 OpenCV 将注意力图转换为颜色热力图
        # attention_colormap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

        # # 转换为 RGB 格式（可选）
        # attention_colormap = cv2.cvtColor(attention_colormap, cv2.COLOR_BGR2RGB)

        # 显示热力图
        plt.imshow(attention_colormap)
        plt.show()


if __name__ == '__main__':
    main()
