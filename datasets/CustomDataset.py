import os
import random
from typing import List, Type
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader


# 定义训练集的pipeline
train_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL image for transforms
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(p=0.5),  # Randomly flip the image vertically
    transforms.RandomRotation(degrees=30),  # Randomly rotate the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change the brightness, contrast, saturation, and hue
    transforms.ToTensor()  # Convert PIL image back to tensor
])

# 定义验证和测试集的pipeline（通常不需要进行数据增强）
val_test_transform = transforms.Compose([
    # transforms.ToPILImage(),  # Convert tensor to PIL image for transforms
    transforms.ToTensor()  # Convert PIL image to tensor
])

class RandomTransform:
    def __init__(self, size=(1024, 1024)):
        self.size = size
        self.resize = transforms.Resize(size)

        # 定义光照和颜色变换
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,  # 随机改变亮度
            contrast=0.2,    # 随机改变对比度
            saturation=0.2,  # 随机改变饱和度
            hue=0.1          # 随机改变色调
        )

    def __call__(self, img_a, img_b, mask):
        # cutout
        if random.random() > 0.5:
            img_a, img_b, mask = cutout(img_a, img_b, mask, self.size)

        # cutmix
        # if random.random() > 0.5:
        #     img_a, img_b = cutmix(img_a, img_b, self.size)

        # 随机水平翻转
        if random.random() > 0.5:
            img_a = transforms.functional.hflip(img_a)
            img_b = transforms.functional.hflip(img_b)
            mask = transforms.functional.hflip(mask)

        # 随机垂直翻转
        if random.random() > 0.5:
            img_a = transforms.functional.vflip(img_a)
            img_b = transforms.functional.vflip(img_b)
            mask = transforms.functional.vflip(mask)

        # 随机旋转
        angle = random.randint(-30, 30)
        img_a = transforms.functional.rotate(img_a, angle)
        img_b = transforms.functional.rotate(img_b, angle)
        mask = transforms.functional.rotate(mask, angle)

        # 强-数据增强
        # 光照和颜色变换
        if random.random() < 0.8:
            img_a = self.color_jitter(img_a)
            img_b = self.color_jitter(img_b)

        # 模糊
        if random.random() < 0.5:
            sigma = np.random.uniform(0.1, 2.0)
            img_a = img_a.filter(ImageFilter.GaussianBlur(radius=sigma))
            img_b = img_b.filter(ImageFilter.GaussianBlur(radius=sigma))

        # 调整大小
        img_a = self.resize(img_a)
        img_b = self.resize(img_b)
        mask = self.resize(mask)

        return img_a, img_b, mask
    

def cutout(image_A, image_B, mask, output_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
    original_width, original_height = image_A.size
    area = original_width * original_height

    # 随机选择裁剪区域的面积
    target_area = random.uniform(scale[0], scale[1]) * area

    # 随机选择裁剪区域的宽高比
    aspect_ratio = random.uniform(ratio[0], ratio[1])

    # 计算裁剪区域的宽度和高度
    w = int(round((target_area * aspect_ratio) ** 0.5))
    h = int(round((target_area / aspect_ratio) ** 0.5))

    # 确保裁剪区域不超过原始图像的大小
    if w > original_width or h > original_height:
        return cutout(image_A, image_B, mask, output_size, scale, ratio)

    # 随机选择裁剪区域的位置
    x1 = random.randint(0, original_width - w)
    y1 = random.randint(0, original_height - h)

    # 裁剪图像
    cropped_image_A = image_A.crop((x1, y1, x1 + w, y1 + h))
    cropped_image_B = image_B.crop((x1, y1, x1 + w, y1 + h))
    cropped_mask = mask.crop((x1, y1, x1 + w, y1 + h))

    # 调整大小
    resized_image_A = cropped_image_A.resize(output_size, Image.BILINEAR)
    resized_image_B = cropped_image_B.resize(output_size, Image.BILINEAR)
    resized_mask = cropped_mask.resize(output_size, Image.BILINEAR)

    return resized_image_A, resized_image_B, resized_mask

def cutmix(image_A, image_B, output_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)):
    original_width, original_height = image_A.size
    area = original_width * original_height

    # 随机选择裁剪区域的面积
    target_area = random.uniform(scale[0], scale[1]) * area

    # 随机选择裁剪区域的宽高比
    aspect_ratio = random.uniform(ratio[0], ratio[1])

    # 计算裁剪区域的宽度和高度
    w = int(round((target_area * aspect_ratio) ** 0.5))
    h = int(round((target_area / aspect_ratio) ** 0.5))

    # 确保裁剪区域不超过原始图像的大小
    if w > original_width or h > original_height:
        return cutmix(image_A, image_B, output_size, scale, ratio)

    # 随机选择裁剪区域的位置
    x1 = random.randint(0, original_width - w)
    y1 = random.randint(0, original_height - h)

    # 创建一个与原始图像大小相同的空白图像
    cutmix_image_A = image_A.copy()
    cutmix_image_B = image_B.copy()

    # 将image_A的裁剪区域替换为image_B的相应区域
    cutmix_image_A.paste(image_B.crop((x1, y1, x1 + w, y1 + h)), (x1, y1))
    # 将image_B的裁剪区域替换为image_A的相应区域
    cutmix_image_B.paste(image_A.crop((x1, y1, x1 + w, y1 + h)), (x1, y1))

    # 调整大小
    resized_image_A = cutmix_image_A.resize(output_size, Image.BILINEAR)
    resized_image_B = cutmix_image_B.resize(output_size, Image.BILINEAR)

    return resized_image_A, resized_image_B

def preprocess(
        x: torch.Tensor,
        img_size: int = 1024,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375]
    ) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
        pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
        x = (x - pixel_mean) / pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

class CustomDataset(Dataset):
    def __init__(self, data_dir, data_type="train"):
        # 根据 data_type 选择相应的文件夹
        self.a_dir = os.path.join(data_dir, data_type, 'A')
        self.b_dir = os.path.join(data_dir, data_type, 'B')
        self.label_dir = os.path.join(data_dir, data_type, 'label')

        # 获取文件名列表
        self.names = sorted(os.listdir(self.a_dir))

        # 根据数据类型选择不同的transform
        if data_type == "train":
            self.transform = RandomTransform(size=(1024, 1024))
        else:
            self.transform = val_test_transform

        # self.preprocess = preprocess
        # self.img_size = 1024
        # self.resize = transforms.Resize((256, 256))
        self.data_type = data_type

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        filename = self.names[idx]

        # 构建图像和掩码的路径
        a_path = os.path.join(self.a_dir, filename)
        b_path = os.path.join(self.b_dir, filename)
        mask_path = os.path.join(self.label_dir, filename)

        # 读取图像
        # img_a = cv2.imread(a_path)
        # img_b = cv2.imread(b_path)
        # img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
        # img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_a = Image.open(a_path).convert('RGB')
        img_b = Image.open(b_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        # 对图像应用变换
        if self.transform:
            if isinstance(self.transform, RandomTransform):
                img_a, img_b, mask = self.transform(img_a, img_b, mask)
                # 将图像和掩码转换为张量
                img_a = transforms.ToTensor()(img_a)
                img_b = transforms.ToTensor()(img_b)
                mask = transforms.ToTensor()(mask)
            else:
                img_a = self.transform(img_a)
                img_b = self.transform(img_b)
                mask = self.transform(mask)

        mask = (mask > 0).float()  # 二值化mask

        # 将 numpy 转换为 tensor
        # img_a = torch.as_tensor(img_a).permute(2, 0, 1).float()
        # img_b = torch.as_tensor(img_b).permute(2, 0, 1).float()

        # 预处理图像
        # img_a = self.preprocess(img_a)
        # img_b = self.preprocess(img_b)

        # # 读取掩码
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # # mask = self.transform.apply_image(mask) if self.transform else self.resize(mask)
        # mask = torch.as_tensor(mask).unsqueeze(0).float()

        # # 对掩码进行填充和调整大小
        # h, w = mask.shape[-2:]
        # padh = self.img_size - h
        # padw = self.img_size - w
        # mask = F.pad(mask, (0, padw, 0, padh))
        # # mask = self.resize(mask).squeeze(0)
        # mask = (mask != 0) * 1

        # 返回字典格式的数据
        data = {
            'image_A': img_a,
            'image_B': img_b,
            'mask': mask,
            'filename': filename
        }

        return data
    
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