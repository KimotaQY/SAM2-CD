import shutil
import os

txt_file = "D:/BaiduNetdiskDownload/WHU-CD/train.txt"
images_folder = "D:/BaiduNetdiskDownload/WHU-CD/B"
new_folder = "D:/BaiduNetdiskDownload/WHU-CD/train/B"

# 从txt文件中读取图片名称列表
with open(txt_file, 'r') as file:
    images = [line.strip() for line in file]

# 遍历名称列表
for image in images:
    # 建立源文件和目标文件的路径
    src = os.path.join(images_folder, image)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    dst = os.path.join(new_folder, image)
    
    # 尝试复制文件
    try:
        shutil.copy2(src, dst)
    except IOError as e:
        print(f'无法复制{image}。原因: {e}')

print('复制完成。')