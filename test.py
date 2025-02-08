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
from PIL import Image

# 读取配置
with open("./configs/config_test.yaml", "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)


DATA_TYPE = config_data["data"]["type"]
NET_NAME = "SAM2_" + DATA_TYPE
TASK_TYPE = "test"


# import matplotlib.pyplot as plt
def visualize_batch(images_a, images_b, masks, pred, **kwargs):
    input_A_np = images_a.cpu().numpy().transpose(0, 2, 3, 1)
    input_B_np = images_b.cpu().numpy().transpose(0, 2, 3, 1)
    # mask_np = masks.cpu().numpy().transpose(0, 2, 3, 1)
    # pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
    mask_np = masks.transpose(0, 2, 3, 1)
    pred_np = pred.transpose(0, 2, 3, 1)

    plt.title(f"F1: {kwargs['F1']}, IoU: {kwargs['IoU']}")

    # 可视化前后时相及其对应的mask
    for i in range(min(4, images_a.size(0))):  # 打印前4个图像样本
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].imshow((input_A_np[i] * 0.5) + 0.3)  # 反归一化
        axs[0, 0].set_title("Image A (T1)")
        axs[0, 0].axis("off")

        axs[0, 1].imshow((input_B_np[i] * 0.5) + 0.3)  # 反归一化
        axs[0, 1].set_title("Image B (T2)")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(mask_np[i].squeeze(), cmap="gray")
        axs[1, 0].set_title("Mask")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(pred_np[i].squeeze(), cmap="gray")
        axs[1, 1].set_title("Pred")
        axs[1, 1].axis("off")

        plt.show()


def FocalLoss(inputs, targets, alpha=0.25, gamma=2):
    # inputs = F.sigmoid(inputs)
    BCE = F.binary_cross_entropy(inputs, targets, reduction="none")
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

    SEED = train_opt["seed"]
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
    TASK_TYPE = "test" if "test" in file_path else "train"

    # dataloaders
    batch_size = train_opt["batch_size"]
    dataloaders = build_dataloader(file_path, batch_size, train_opt["num_workers"])
    val_loader = dataloaders["test"]

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

    # 新建CSV文件并写入表头
    vis_outpath = "./vis_outputs/LEVIR-CD-VIS-baseline-0.2-train"
    csv_file_path = vis_outpath + "/output.csv"
    with open(csv_file_path, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["filename", "f1", "IoU"])

        for valid_data in iterations:
            valid_input_A = (
                valid_data["image_A"]
                .to(torch.device("cuda", int(train_opt["dev_id"])))
                .float()
            )
            valid_input_B = (
                valid_data["image_B"]
                .to(torch.device("cuda", int(train_opt["dev_id"])))
                .float()
            )
            labels = (
                valid_data["mask"]
                .to(torch.device("cuda", int(train_opt["dev_id"])))
                .float()
            )

            valid_input = torch.cat((valid_input_A, valid_input_B), dim=0)
            with torch.no_grad():
                # outputs = model(valid_input)
                # # 上采样输出到标签尺寸
                # outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
                # loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.tensor([10]).to(torch.device('cuda', int(train_opt['dev_id']))))
                outputs, outputs_2, outputs_3 = model(valid_input)
                loss = (
                    BCEDiceLoss(outputs, labels)
                    + BCEDiceLoss(outputs_2, labels)
                    + BCEDiceLoss(outputs_3, labels)
                )
            val_loss.update(loss.cpu().detach().numpy())

            outputs = outputs.cpu().detach()
            labels = labels.cpu().detach().numpy()
            preds = F.sigmoid(outputs).numpy()
            filenames = valid_data["filename"]

            for pred, label, filename in zip(preds, labels, filenames):
                acc, precision, recall, F1, IoU = accuracy(pred, label)
                F1_meter.update(F1)
                Acc_meter.update(acc)
                IoU_meter.update(IoU)
                Pre_meter.update(precision)
                Rec_meter.update(recall)

                # 将文件名和其他信息写入CSV
                csv_writer.writerow([filename, round(F1 * 100, 2), round(IoU * 100, 2)])
                # if IoU > 0.95:
                #     visualize_batch(valid_input_A, valid_input_B, labels, preds, F1=F1, IoU=IoU)

                # ##################### 输出预测结果 #####################
                change_map = (pred >= 0.5).astype(int).squeeze(0)
                # ground_truth = label.squeeze(0)
                # # Step 1: 计算 FP 和 FN 区域
                # fp = (change_map == 1) & (ground_truth == 0)  # 假阳性区域
                # fn = (change_map == 0) & (ground_truth == 1)  # 假阴性区域

                # Step 2: 创建一个 RGB 彩色图像用于标记结果
                result_img = np.zeros(
                    (change_map.shape[0], change_map.shape[1], 3), dtype=np.uint8
                )

                # 将变化检测的结果转换成白色区域
                result_img[change_map == 1] = [255, 255, 255]  # 白色表示检测到变化

                # # Step 3: 在结果图上叠加 FP 和 FN 区域
                # result_img[fp] = [255, 0, 0]   # 红色标记假阳性
                # result_img[fn] = [0, 255, 0]   # 绿色标记假阴性

                # Step 4: 保存结果图
                output_image = Image.fromarray(result_img)
                output_image.save(f"{vis_outpath}/{filename}")
                # ##################### 输出预测结果 End #####################

            pbar_desc = "Model valid loss --- "
            pbar_desc += f"loss: {val_loss.average():.5f}"
            pbar_desc += f", F1: {F1_meter.avg * 100:.2f}"
            pbar_desc += f", mIOU: {IoU_meter.avg * 100:.2f}"
            pbar_desc += f", Acc: {Acc_meter.avg * 100:.2f}"
            pbar_desc += f", Pre: {Pre_meter.avg * 100:.2f}"
            pbar_desc += f", Rec: {Rec_meter.avg * 100:.2f}"
            iterations.set_description(pbar_desc)

    return (
        F1_meter.avg,
        Acc_meter.avg,
        IoU_meter.avg,
        val_loss.avg,
        Pre_meter.avg,
        Rec_meter.avg,
    )


import cv2
import matplotlib.pyplot as plt


def test(val_loader, model):
    train_opt = config_data["training"]

    # 用于存储目标层的激活和梯度
    activations = []
    gradients = []

    # 定义钩子函数来保存激活值
    def forward_hook(module, input, output):
        # 如果输出是元组，选择你想要的输出
        if isinstance(output, tuple):
            activations.append(output[3])  # 选择第一个输出
        else:
            activations.append(output)

    # 定义钩子函数来保存梯度
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # 选择模型中要监控的层
    target_layer = (
        model.mask_decoder.decoder.channel_attention
    )  # 根据模型的实际结构修改
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    # 1. 创建 GradCAM 对象
    # cam = GradCAM(model=model, target_layers=[target_layer])

    iterations = tqdm(val_loader)
    for valid_data in iterations:
        valid_input_A = (
            valid_data["image_A"]
            .to(torch.device("cuda", int(train_opt["dev_id"])))
            .float()
        )
        valid_input_B = (
            valid_data["image_B"]
            .to(torch.device("cuda", int(train_opt["dev_id"])))
            .float()
        )
        labels = (
            valid_data["mask"]
            .to(torch.device("cuda", int(train_opt["dev_id"])))
            .float()
        )

        valid_input = torch.cat((valid_input_A, valid_input_B), dim=0)

        # 模型的前向传播
        outputs, outputs_2, outputs_3 = model(valid_input)

        # 计算损失
        loss1 = BCEDiceLoss(outputs, labels)
        loss2 = BCEDiceLoss(outputs_2, labels)
        loss3 = BCEDiceLoss(outputs_3, labels)
        loss = loss1 + loss2 + loss3

        # 反向传播
        loss.backward()

        # 梯度已经计算，获取激活值和梯度用于 GradCAM
        gradients = gradients[0].cpu().detach().numpy()
        activations = activations[0].cpu().detach().numpy()

        # 生成 GradCAM 热力图
        weights = np.mean(gradients, axis=(2, 3))  # 计算全局平均池化，获得权重
        cam = np.zeros(activations.shape[2:], dtype=np.float32)  # 初始化热力图

        for i in range(weights.shape[1]):  # 遍历每个 channel
            cam += weights[0, i] * activations[0, i, :, :]  # 加权激活

        cam = np.maximum(cam, 0)  # ReLU 操作
        cam = cv2.resize(cam, (labels.shape[-2], labels.shape[-1]))  # 调整到标签大小
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化到 [0, 1]

        # 可视化 GradCAM 热力图
        visualize_cam(valid_input_B[0], cam)  # 需要实现 visualize_cam 来显示


def visualize_cam(img, cam):
    """
    将 GradCAM 热力图叠加到原始图像上进行可视化
    img: 输入图像 (tensor)
    cam: 生成的 GradCAM (numpy array)
    """
    img = img.cpu().numpy().transpose(1, 2, 0)  # 转换到 HWC 格式
    img = (img - img.min()) / (img.max() - img.min())  # 归一化图像

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # 反转颜色
    heatmap = cv2.bitwise_not(heatmap)

    heatmap = np.float32(heatmap)

    superimposed_img = heatmap + np.float32(img)
    superimposed_img = superimposed_img / np.max(superimposed_img)

    plt.imshow(superimposed_img)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
