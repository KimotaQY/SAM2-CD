import os
import shutil
import math

# 定义文件夹路径
train_folder = 'E:/CD_datasets/LEVIR-CD/train'
folder_a = os.path.join(train_folder, 'A')
folder_b = os.path.join(train_folder, 'B')
label_folder = os.path.join(train_folder, 'label')

# 定义新的文件夹路径
new_folder = 'E:/CD_datasets/LEVIR-CD/train_0.4'
new_folder_a = os.path.join(new_folder, 'A')
new_folder_b = os.path.join(new_folder, 'B')
new_folder_label = os.path.join(new_folder, 'label')

# 创建新的文件夹
os.makedirs(new_folder_a, exist_ok=True)
os.makedirs(new_folder_b, exist_ok=True)
os.makedirs(new_folder_label, exist_ok=True)

# 获取文件名列表
file_names = os.listdir(folder_a)  # 假设文件名在A文件夹中一致

# 计算需要复制的文件数量
total_files = len(file_names)
num_to_copy = math.ceil(total_files * 0.4)  # 取x%的图片
interval = total_files // num_to_copy  # 计算间隔

# 创建一个列表来保存复制的文件名
copied_files = []

# 复制文件并保存文件名
for i in range(num_to_copy):
    index = i * interval
    # 复制A文件夹中的文件
    src_a = os.path.join(folder_a, file_names[index])
    dst_a = os.path.join(new_folder_a, file_names[index])
    shutil.copy(src_a, dst_a)
    copied_files.append(f'{file_names[index]}')

    # 复制B文件夹中的文件
    src_b = os.path.join(folder_b, file_names[index])
    dst_b = os.path.join(new_folder_b, file_names[index])
    shutil.copy(src_b, dst_b)
    # copied_files.append(f'B/{file_names[index]}')

    # 复制label文件夹中的文件
    src_label = os.path.join(label_folder, file_names[index])
    dst_label = os.path.join(new_folder_label, file_names[index])
    shutil.copy(src_label, dst_label)
    # copied_files.append(f'label/{file_names[index]}')

# 将复制的文件名写入txt文件
with open(os.path.join(new_folder, 'copied_files.txt'), 'w') as f:
    for file_name in copied_files:
        f.write(f"{file_name}\n")

print(f'成功复制 {num_to_copy} 张图片到 {new_folder} 文件夹中，并记录在 copied_files.txt 文件中。')