import argparse
import csv
import json
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from datetime import datetime
import yaml
from models.build_sam import build_sam2
from utils.losses import mean_iou, CombinedLoss
from datasets.CustomDataset import build_dataloader
# from datasets.SYSU_CD import CustomDataset

import torch.nn.functional as F
from utils.utils import binary_accuracy as accuracy
from utils.utils import AverageMeter


# 读取配置
with open('./configs/config.yaml', "r", encoding='utf-8') as file:
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


def main():
    train_opt = config_data["training"]

    SEED = train_opt['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # 新建保存文件夹
    date_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # 获取当前的年月日和时间
    output_model_path = config_data["logging"]["save_dir"] + config_data["data"]["type"]
    epochs = train_opt['num_epochs']
    batch_size = train_opt["batch_size"]
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    save_path = os.path.join(output_model_path, f"model_{epochs}_{batch_size}_{date_time}")
    os.makedirs(save_path)
    os.makedirs(save_path, exist_ok=True)

    # 配置文件写入txt
    data_str = json.dumps(config_data, indent=4)
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:  # 保存配置文件
        f.write(data_str)

    # 构建模型
    model_opt = config_data["model"]
    checkpoint_path = model_opt["checkpoint_path"]
    model_cfg = model_opt["config"]
    sam2 = build_sam2(model_cfg, checkpoint_path)
    
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
    dataloaders = build_dataloader(file_path, batch_size, train_opt['num_workers'])
    train_loader = dataloaders['train']
    val_loader = dataloaders['test']

    # 定义优化器、调度器
    lr = train_opt['learning_rate']
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, sam2.parameters()), lr=lr,
                            weight_decay=train_opt['weight_decay'])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr*2,
        steps_per_epoch=len(train_loader),  # 每个epoch的总步数（一般等于训练样本数除以batch_size）
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        # div_factor=10,
        # final_div_factor=100
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)

    # 加载优化器、调度器状态
    if 'optimizer' in checkpoint:
        print("——————加载优化器状态——————")
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint:
        print("——————加载调度器状态——————")    
        scheduler.load_state_dict(checkpoint['scheduler'])
    if 'epoch' in checkpoint:
        print("——————加载epoch——————")
        epoch = checkpoint['epoch']
        print("当前epoch: ", epoch)
    else:
        epoch = 0
    
    train(train_loader, sam2, optimizer, scheduler, val_loader, save_path, epoch)


def train(train_loader, model, optimizer, scheduler, val_loader, save_path, curr_epoch=0):
    global TASK_TYPE
    
    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    bestloss = 1.0
    bestaccT = 0.0

    train_opt = config_data["training"]
    epochs = train_opt['num_epochs'] - curr_epoch
    batch_size = train_opt["batch_size"]
    lr = train_opt['learning_rate']
    begin_time = time.time()

    if epochs <= 0:
        raise ValueError("——————No epochs left to train——————")

    # 创建CSV文件并写入表头
    with open(save_path + '/training_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['epoch', 'train_loss', 'train_acc', 'train_f1', 'train_iou', 'val_loss', 'val_acc', 'val_f1', 'val_iou'])

        for epoch in range(epochs):
            torch.cuda.empty_cache()
            model.train()
            acc_meter = AverageMeter()
            train_loss = AverageMeter()
            iou_meter = AverageMeter()
            f1_meter = AverageMeter()
            loss1_meter = AverageMeter()
            loss2_meter = AverageMeter()
            loss3_meter = AverageMeter()
            start = time.time()

            iterations = tqdm(train_loader)
            for train_data in iterations:
                train_input_A = train_data['image_A'].to(torch.device('cuda', int(train_opt['dev_id']))).float()
                train_input_B = train_data['image_B'].to(torch.device('cuda', int(train_opt['dev_id']))).float()
                labels = train_data['mask'].to(torch.device('cuda', int(train_opt['dev_id']))).float()

                # # 可视化前后时相及其对应的mask
                # visualize_batch(train_input_A, train_input_B, labels)

                optimizer.zero_grad()
                train_input = torch.cat((train_input_A, train_input_B), dim=0)
                # outputs = model(train_input)
                # # 上采样输出到标签尺寸
                # outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                outputs, outputs_2, outputs_3 = model(train_input)

                loss1 = BCEDiceLoss(outputs, labels)
                loss2 = BCEDiceLoss(outputs_2, labels)
                loss3 = BCEDiceLoss(outputs_3, labels)
                loss = loss1 + loss2 + loss3
                loss.backward()
                optimizer.step()
                scheduler.step()

                labels = labels.cpu().detach().numpy()
                outputs = outputs.cpu().detach()
                preds = F.sigmoid(outputs).numpy()
                acc_curr_meter = AverageMeter()
                for (pred, label) in zip(preds, labels):
                    acc, precision, recall, F1, IoU = accuracy(pred, label)
                    acc_curr_meter.update(acc)
                    iou_meter.update(IoU)
                    f1_meter.update(F1)
                acc_meter.update(acc_curr_meter.avg)
                train_loss.update(loss.cpu().detach().numpy())
                loss1_meter.update(loss1.data.item())
                loss2_meter.update(loss2.data.item())
                loss3_meter.update(loss3.data.item())

                pbar_desc = "Model train loss --- "
                pbar_desc += f"loss: {train_loss.avg:.5f}"
                pbar_desc += f", f1: {f1_meter.avg:.5f}"
                pbar_desc += f", iou: {iou_meter.avg:.5f}"
                pbar_desc += f", lr: {scheduler.get_last_lr()}"
                # pbar_desc += f", l1: {loss1.data.item():.5f}"
                # pbar_desc += f", l2: {loss2.data.item():.5f}"
                # pbar_desc += f", l3: {loss3.data.item():.5f}"
                iterations.set_description(pbar_desc)

            val_F, val_acc, val_IoU, val_loss, val_pre, val_rec = validate(val_loader, model)
            writer.writerow(
                [epoch + curr_epoch + 1, train_loss.avg, acc_meter.avg * 100, f1_meter.avg * 100, iou_meter.avg * 100, val_loss,
                 val_acc * 100, val_F * 100, val_IoU * 100])
            if val_F > bestF or val_IoU > bestIoU:
                bestF = val_F
                bestacc = val_acc
                bestIoU = val_IoU
                bestPre = val_pre
                bestRec = val_rec
                if TASK_TYPE != 'test':
                    torch.save({
                        'model': model.state_dict()
                    }, os.path.join(save_path, NET_NAME + '_e%d_OA%.2f_F%.2f_IoU%.2f.pth' % (
                        epoch + curr_epoch + 1, val_acc * 100, val_F * 100, val_IoU * 100)))
                # 记录best_model评分
                with open(save_path + '/best_models_score.txt', 'a') as file:
                    file.write('e%d OA_%.2f F1_%.2f Iou_%.2f Pre_%.2f Rec_%.2f \n' % (epoch + curr_epoch + 1, val_acc * 100, val_F * 100, val_IoU * 100, val_pre * 100, val_rec * 100))
            if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
            print('[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1: %.2f IoU: %.2f, Pre: %.2f, Rec: %.2f L1 %.2f L2 %.2f L3 %.2f' \
                  % (epoch + curr_epoch + 1, epochs + curr_epoch, time.time() - begin_time, bestaccT * 100, bestacc * 100, bestF * 100,
                     bestIoU * 100, bestPre * 100, bestRec * 100, loss1_meter.avg, loss2_meter.avg, loss3_meter.avg))

            # scheduler.step()
            # 根据验证损失更新学习率
            # if TASK_TYPE != 'test':
            #     scheduler.step(val_loss)
            # else:
            #     scheduler.step(train_loss.avg)

            # 保存检查点
            model_path = save_path + "/" + NET_NAME + '_checkpoint.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + curr_epoch + 1,
                
            }, model_path)


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


if __name__ == '__main__':
    main()
