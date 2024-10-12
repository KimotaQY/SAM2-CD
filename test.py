import argparse
import csv
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import yaml
from models.sam2_cd.build_sam import build_sam2
from utils.losses import mean_iou, CombinedLoss
from datasets.CustomDataset import CustomDataset

import torch.nn.functional as F
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter

###################### Data and Model ########################
NET_NAME = 'SAM2'
TASK_TYPE = 'test'
# from datasets import Levir_CD as RS
# DATA_NAME = 'LevirCD'
#from datasets import WHU_CD as RS
#DATA_NAME = 'WHU_CD'
###################### Data and Model ########################
######################## Parameters ########################
args = {
    'gpu': True,
    'dev_id': 0,
    'multi_gpu': None,  #"0,1,2,3",
    'weight_decay': 1e-2,
    'momentum': 0.9,
    'print_freq': 100,
    'predict_step': 5,
    'num_workers': 10,
    'config_path': './configs/config.yaml'}


# 数据加载
def build_dataloader(data_dir, batch_size, num_workers):
    dataloaders = {
        key: DataLoader(
            CustomDataset(data_dir, key),
            batch_size=batch_size,
            shuffle=True if key == 'train' else False,
            num_workers=num_workers,
            # pin_memory=True,
            # persistent_workers=False  # 增加 persistent_workers 参数，避免频繁的加载与释放
        ) for key in ['train', 'val', 'test']
    }

    return dataloaders

import torchvision
# import matplotlib.pyplot as plt
def visualize_batch(images_a, images_b, masks, preds, loss, f1_score, iou, acc):
    # input_A_np = images_a.cpu().numpy().transpose(0, 2, 3, 1)
    # input_B_np = images_b.cpu().numpy().transpose(0, 2, 3, 1)
    # mask_np = masks.cpu().numpy().transpose(0, 2, 3, 1)
    # preds_np = preds.cpu().numpy().transpose(0, 2, 3, 1)
    input_A_np = images_a.cpu().numpy().transpose(1, 2, 0)
    input_B_np = images_b.cpu().numpy().transpose(1, 2, 0)
    mask_np = masks.transpose(1, 2, 0)
    preds_np = preds.transpose(1, 2, 0)
    
    # 可视化前后时相及其对应的mask
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow((input_A_np * 0.5) + 0.5)  # 反归一化
    axs[0].set_title('Image A (T1)')
    axs[0].axis('off')
    
    axs[1].imshow((input_B_np * 0.5) + 0.5)  # 反归一化
    axs[1].set_title('Image B (T2)')
    axs[1].axis('off')
    
    axs[2].imshow(mask_np.squeeze(), cmap='gray')
    axs[2].set_title('Ground Truth Mask')
    axs[2].axis('off')
    
    axs[3].imshow(preds_np.squeeze(), cmap='gray')
    axs[3].set_title('Predicted Mask')
    axs[3].axis('off')

    # 在每个图像窗口上显示评价指标
    fig.suptitle(f'Loss: {loss:.4f}, F1 Score: {f1_score:.4f}, IoU: {iou:.4f}, Acc: {acc:.4f}', fontsize=16)
    
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
    focal = FocalLoss(inputs, targets)  # BCEDiceFocalLoss
    return bce + 1 - dice + focal

def main():
    # 读取配置
    with open(args['config_path'], "r", encoding='utf-8') as file:
        config_data = yaml.safe_load(file)

    train_opt = config_data["training"]

    SEED = train_opt['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    epochs = train_opt['num_epochs']
    

    # # 定义文件目录
    # directory = 'D:\SAM\change_detection_project\outputs\models\kan_sam2_hiera_l\model_100_1_61'

    # # 使用 os 模块读取目录下所有文件并筛选
    # all_files = os.listdir(directory)
    # pth_files = [f for f in all_files if f.endswith('.pth')]
    # # print("方法1:", pth_files)

    # best_F = 0
    # best_acc = 0
    # best_IoU = 0
    # best_loss = 0
    # best_pth = ''
    # for pth_file in pth_files:
    #     modeling_cfg = config_data["model"]
    #     checkpoint_path = f'{directory}\{pth_file}'
    #     print(checkpoint_path)
    #     model_cfg = modeling_cfg["config"]
    #     sam2 = build_sam2(model_cfg, checkpoint_path)

    #     file_path = config_data["data"]["root"]
    #     dataloaders = build_dataloader(file_path, 1, opt.num_workers)

    #     val_F, val_acc, val_IoU, val_loss = validate(dataloaders['test'], sam2)
    #     if val_F > best_F:
    #         best_F = val_F
    #         best_acc = val_acc
    #         best_IoU = val_IoU
    #         best_loss = val_loss
    #         best_pth = pth_file
    #     print('[epoch %d/%d %.1fs] Best rec: Val %.2f, F1 score: %.2f IoU %.2f Loss %.2f pth %s' \
    #                 % (0, epochs, 0, best_acc * 100, best_F * 100,
    #                     best_IoU * 100, best_loss, best_pth))

    # 单例测试
    modeling_cfg = config_data["model"]
    # checkpoint_path = modeling_cfg["checkpoint_path"]
    checkpoint_path = "D:/SAM/SAM2-CD/outputs/models/kan_sam2_hiera_l/model_100_1_0/SAM2_LEVIR_e99_OA99.17_F91.55_IoU85.07.pth"
    model_cfg = modeling_cfg["config"]
    model = build_sam2(model_cfg, checkpoint_path)
    
    file_path = config_data["data"]["root"]
    global TASK_TYPE 
    TASK_TYPE = 'test' if 'test' in file_path else 'train'
    dataloaders = build_dataloader(file_path, 1, args['num_workers'])

    val_F, val_acc, val_IoU, val_loss, test_pre, test_rec = validate(dataloaders['test'], model)
    print('[epoch %d/%d %.1fs] Best rec: Val %.2f, F1 score: %.2f IoU %.2f Loss %.2f Pre %.2f Rec %.2f' \
                % (0, epochs, 0 * 100, val_acc * 100, val_F * 100,
                    val_IoU * 100, val_loss, test_pre*100, test_rec*100))

    
def validate(val_loader, model):
    model.eval()
    torch.cuda.empty_cache()

    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    Pre_meter = AverageMeter()
    Rec_meter = AverageMeter()

    iterations = tqdm(val_loader)
    for valid_data in iterations:
        valid_input_A = valid_data['image_A'].to(torch.device('cuda', int(args['dev_id']))).float()
        valid_input_B = valid_data['image_B'].to(torch.device('cuda', int(args['dev_id']))).float()
        labels = valid_data['mask'].to(torch.device('cuda', int(args['dev_id']))).float()
        
        valid_input = torch.cat((valid_input_A, valid_input_B), dim=0)
        with torch.no_grad():
            # outputs = model(valid_input)
            # # 上采样输出到标签尺寸
            # outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            # loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.tensor([10]).to(torch.device('cuda', int(args['dev_id']))))
            outputs, outputs_2, outputs_3 = model(valid_input)
            loss = BCEDiceLoss(outputs, labels) + BCEDiceLoss(outputs_2, labels) + BCEDiceLoss(outputs_3, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        preds = F.sigmoid(outputs).numpy()
        for (pred, label, input_A, input_B) in zip(preds, labels, valid_input_A, valid_input_B):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
            Pre_meter.update(precision)
            Rec_meter.update(recall)
            # 可视化前后时相及其对应的mask
            # visualize_batch(input_A, input_B, label, pred, loss.cpu().detach().numpy(), F1, IoU, acc)

        pbar_desc = "Model valid loss --- "
        pbar_desc += f"Total loss: {val_loss.average():.5f}"
        pbar_desc += f", total mIOU: {IoU_meter.avg:.5f}"
        iterations.set_description(pbar_desc)
    
    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg, Pre_meter.avg, Rec_meter.avg



if __name__ == '__main__':
    main()