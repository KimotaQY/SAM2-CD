import argparse
import csv
import json
import os
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
import yaml
from models.build_sam import build_sam2

# from datasets.CustomDataset import build_dataloader
from datasets.CD import build_dataloader

from utils.metrics import binary_accuracy as accuracy
from utils.AverageMeter import AverageMeter
from utils.visualization import visualize_batch
from utils.losses import BCEDiceLoss


# 读取配置
with open("./configs/config.yaml", "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)


DATA_TYPE = config_data["data"]["type"]
NET_NAME = "SAM2_" + DATA_TYPE
TASK_TYPE = "test"


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
    set_seed(SEED)

    # 新建保存文件夹
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前的年月日和时间
    output_model_path = (
        config_data["logging"]["save_dir"]
        + config_data["data"]["type"]
        + "/"
        + config_data["model"]["model_type"]
    )
    epochs = train_opt["num_epochs"]
    batch_size = train_opt["batch_size"]
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    save_path = os.path.join(
        output_model_path, f"model_{epochs}_{batch_size}_{date_time}"
    )
    os.makedirs(save_path)
    os.makedirs(save_path, exist_ok=True)

    # 配置文件写入txt
    data_str = json.dumps(config_data, indent=4)
    with open(os.path.join(save_path, "config.txt"), "w") as f:  # 保存配置文件
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
    #     if any(
    #         keyword in name
    #         for keyword in [
    #             "down_channel",
    #             "soft_ffn",
    #             "mask_decoder",
    #             "dynamic_map_gen",
    #         ]
    #     ):
    #         param.requires_grad = True
    #         print(f"参数名: {name}, 尺寸: {param.requires_grad}")
    #     print(f"参数名: {name}, 尺寸: {param.requires_grad}")

    file_path = config_data["data"][DATA_TYPE]
    global TASK_TYPE
    TASK_TYPE = "test" if "test" in file_path else "train"

    # dataloaders
    dataloaders = build_dataloader(file_path, batch_size, train_opt["num_workers"])
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # 定义优化器、调度器
    lr = train_opt["learning_rate"]
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, sam2.parameters()),
        lr=lr,
        weight_decay=train_opt["weight_decay"],
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr * 2,
        steps_per_epoch=len(
            train_loader
        ),  # 每个epoch的总步数（一般等于训练样本数除以batch_size）
        epochs=epochs,
        pct_start=0.1,
        anneal_strategy="cos",
        # div_factor=10,
        # final_div_factor=100
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)

    # 加载优化器、调度器状态
    if "optimizer" in checkpoint:
        print("——————加载优化器状态——————")
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        print("——————加载调度器状态——————")
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "epoch" in checkpoint:
        print("——————加载epoch——————")
        epoch = checkpoint["epoch"]
        print("当前epoch: ", epoch)
    else:
        epoch = 0

    train(train_loader, sam2, optimizer, scheduler, val_loader, save_path, epoch)


def train(
    train_loader, model, optimizer, scheduler, val_loader, save_path, curr_epoch=0
):
    global TASK_TYPE

    bestF = 0.0
    bestacc = 0.0
    bestIoU = 0.0
    # bestloss = 1.0
    bestaccT = 0.0

    train_opt = config_data["training"]
    epochs = train_opt["num_epochs"] - curr_epoch
    begin_time = time.time()
    validate_every = config_data["validation"]["validate_every"]

    if epochs <= 0:
        raise ValueError("——————No epochs left to train——————")

    scaler = GradScaler()  # 初始化 GradScaler

    # 创建CSV文件并写入表头
    with open(save_path + "/training_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "train_f1",
                "train_iou",
                "val_loss",
                "val_acc",
                "val_f1",
                "val_iou",
            ]
        )

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
            # start = time.time()

            iterations = tqdm(train_loader)
            for train_data in iterations:
                train_input_A = (
                    train_data["image_A"]
                    .to(torch.device("cuda", int(train_opt["dev_id"])))
                    .float()
                )
                train_input_B = (
                    train_data["image_B"]
                    .to(torch.device("cuda", int(train_opt["dev_id"])))
                    .float()
                )
                labels = (
                    train_data["mask"]
                    .to(torch.device("cuda", int(train_opt["dev_id"])))
                    .float()
                )

                # # 可视化前后时相及其对应的mask
                # visualize_batch(train_input_A, train_input_B, labels)

                optimizer.zero_grad()

                # 使用 autocast 混合精度训练
                with autocast(device_type="cuda"):
                    train_input = torch.cat((train_input_A, train_input_B), dim=0)
                    outputs, outputs_2, outputs_3 = model(train_input)

                    loss1 = BCEDiceLoss(outputs, labels)
                    loss2 = BCEDiceLoss(outputs_2, labels)
                    loss3 = BCEDiceLoss(outputs_3, labels)
                    loss = loss1 + loss2 + loss3

                # 使用 scaler 进行梯度缩放和反向传播
                scaler.scale(loss).backward()
                # loss.backward()

                # 使用 scaler 进行优化器 step 操作
                scaler.step(optimizer)
                # optimizer.step()

                # 更新 scaler
                scaler.update()

                scheduler.step()

                labels = labels.cpu().detach().numpy()
                outputs = outputs.cpu().detach()
                preds = F.sigmoid(outputs).numpy()
                acc_curr_meter = AverageMeter()
                for pred, label in zip(preds, labels):
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

            if (epoch + curr_epoch + 1) % validate_every == 0:
                val_F, val_acc, val_IoU, val_loss, val_pre, val_rec = validate(
                    val_loader, model
                )
                writer.writerow(
                    [
                        epoch + curr_epoch + 1,
                        train_loss.avg,
                        acc_meter.avg * 100,
                        f1_meter.avg * 100,
                        iou_meter.avg * 100,
                        val_loss,
                        val_acc * 100,
                        val_F * 100,
                        val_IoU * 100,
                    ]
                )
                if val_F > bestF or val_IoU > bestIoU:
                    bestF = val_F
                    bestacc = val_acc
                    bestIoU = val_IoU
                    bestPre = val_pre
                    bestRec = val_rec
                    if TASK_TYPE != "test":
                        torch.save(
                            {"model": model.state_dict()},
                            os.path.join(
                                save_path,
                                NET_NAME
                                + "_e%d_OA%.2f_F%.2f_IoU%.2f.pth"
                                % (
                                    epoch + curr_epoch + 1,
                                    val_acc * 100,
                                    val_F * 100,
                                    val_IoU * 100,
                                ),
                            ),
                        )
                    # 清理多余的 .pth 文件
                    model_files = [
                        f for f in os.listdir(save_path) if f.endswith(".pth")
                    ]
                    if len(model_files) > 5:  # 设置最大保留的模型数量
                        # 按文件创建时间排序，保留最新的 5 个模型
                        model_files.sort(
                            key=lambda x: os.path.getmtime(os.path.join(save_path, x))
                        )
                        for file_name in model_files[:-5]:
                            os.remove(os.path.join(save_path, file_name))
                            print(f"Deleted old model: {file_name}")

                    # 记录best_model评分
                    with open(save_path + "/best_models_score.txt", "a") as file:
                        file.write(
                            "e%d OA_%.2f F1_%.2f Iou_%.2f Pre_%.2f Rec_%.2f \n"
                            % (
                                epoch + curr_epoch + 1,
                                val_acc * 100,
                                val_F * 100,
                                val_IoU * 100,
                                val_pre * 100,
                                val_rec * 100,
                            )
                        )
                if acc_meter.avg > bestaccT:
                    bestaccT = acc_meter.avg
                print(
                    "[epoch %d/%d %.1fs] Best rec: Train %.2f, Val %.2f, F1: %.2f IoU: %.2f, Pre: %.2f, Rec: %.2f L1 %.2f L2 %.2f L3 %.2f"
                    % (
                        epoch + curr_epoch + 1,
                        epochs + curr_epoch,
                        time.time() - begin_time,
                        bestaccT * 100,
                        bestacc * 100,
                        bestF * 100,
                        bestIoU * 100,
                        bestPre * 100,
                        bestRec * 100,
                        loss1_meter.avg,
                        loss2_meter.avg,
                        loss3_meter.avg,
                    )
                )

            # scheduler.step()
            # 根据验证损失更新学习率
            # if TASK_TYPE != 'test':
            #     scheduler.step(val_loss)
            # else:
            #     scheduler.step(train_loss.avg)

            # 保存检查点
            model_path = save_path + "/" + NET_NAME + "_checkpoint.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + curr_epoch + 1,
                },
                model_path,
            )


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

        # 可视化前后时相及其对应的mask
        # visualize_batch(valid_input_A, valid_input_B, labels)

        valid_input = torch.cat((valid_input_A, valid_input_B), dim=0)
        with torch.no_grad():
            # outputs = model(valid_input)
            # # 上采样输出到标签尺寸
            # outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            # loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=torch.tensor([10]).to(torch.device('cuda', int(train_opt['dev_id']))))
            outputs, outputs_2, outputs_3 = model(valid_input)
            loss_1 = BCEDiceLoss(outputs, labels)
            loss_2 = BCEDiceLoss(outputs_2, labels)
            loss_3 = BCEDiceLoss(outputs_3, labels)
            loss = loss_1 + loss_2 + loss_3
        val_loss.update(loss.cpu().detach().numpy())

        # L1 = loss_1.data.item()
        # L2 = loss_2.data.item()
        # L3 = loss_3.data.item()
        # min_loss = min(L1, L2, L3)
        # if min_loss == L1:
        #     outputs = outputs.cpu().detach()
        # elif min_loss == L2:
        #     outputs = outputs_2.cpu().detach()
        # elif min_loss == L3:
        #     outputs = outputs_3.cpu().detach()

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        preds = F.sigmoid(outputs).numpy()
        for pred, label in zip(preds, labels):
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

    return (
        F1_meter.avg,
        Acc_meter.avg,
        IoU_meter.avg,
        val_loss.avg,
        Pre_meter.avg,
        Rec_meter.avg,
    )


if __name__ == "__main__":
    main()
