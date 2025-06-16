from PIL import Image
import os

# 源、目标文件夹名称
src_folder = "E:/CD_datasets/LEVIR-CD/val/B"
dst_folder = "E:/CD_datasets/LEVIR-CD/256/val/B"

# 如果目标文件夹不存在则创建
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)

for img_file in os.listdir(src_folder):
    if img_file.endswith(".png"):
        img_path = os.path.join(src_folder, img_file)
        print(f"正在处理：{img_file}")
        img = Image.open(img_path)
        # 开始分割图片
        for i in range(0, img.width, 256):
            for j in range(0, img.height, 256):
                box = (i, j, i + 256, j + 256)
                cropped_img = img.crop(box)
                filename, ext = os.path.splitext(img_file)
                new_file = (
                    filename + "_{0}_{1}".format(int(i / 256), int(j / 256)) + ext
                )
                save_path = os.path.join(dst_folder, new_file)
                cropped_img.save(save_path)
