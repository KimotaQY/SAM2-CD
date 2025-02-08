from datasets.transform import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import random


class CustomDataset(Dataset):
    def __init__(self, data_dir, data_type="train"):
        # 根据 data_type 选择相应的文件夹
        self.a_dir = os.path.join(data_dir, data_type, "A")
        self.b_dir = os.path.join(data_dir, data_type, "B")
        self.label_dir = os.path.join(data_dir, data_type, "label")

        # 获取文件名列表
        self.names = sorted(os.listdir(self.a_dir))

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
        img_a = Image.open(a_path).convert("RGB")
        img_b = Image.open(b_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 对图像应用变换
        if "train" not in self.data_type:
            img_a, img_b = normalize(img_a, img_b)
        else:
            # 弱增强
            img_a, img_b, mask = resize(img_a, img_b, mask, (0.5, 2.0))
            img_a, img_b, mask = crop(img_a, img_b, mask, 1024)
            img_a, img_b, mask = hflip(img_a, img_b, mask, p=0.5)
            img_a, img_b, mask = vflip(img_a, img_b, mask, p=0.5)
            img_a, img_b, mask = rotate(img_a, img_b, mask, p=0.5)

            img_a, img_b = color_jitter(img_a, img_b, p=0.8)
            img_a, img_b = grayscale(img_a, img_b, p=0.2)
            img_a, img_b = blur(img_a, img_b, p=0.5)
            # cutmix or cutout
            if random.random() < 0.5:
                img_a, img_b = cutmix(img_a, img_b, (1024, 1024))
                # img_a, img_b, mask = cutout(img_a, img_b, mask, (1024, 1024))

            img_a, img_b = normalize(img_a, img_b)

        mask = transforms.ToTensor()(mask)
        mask = (mask > 0).float()

        # 返回字典格式的数据
        data = {"image_A": img_a, "image_B": img_b, "mask": mask, "filename": filename}

        return data


def build_dataloader(data_dir, batch_size, num_workers):
    dataloaders = {
        key: DataLoader(
            CustomDataset(data_dir, key),
            batch_size=batch_size,
            shuffle=True if key == "train" else False,
            num_workers=num_workers,
            # pin_memory=True,
            # persistent_workers=False  # 增加 persistent_workers 参数，避免频繁的加载与释放
        )
        for key in ["train", "val", "test"]
    }

    return dataloaders
