import os
import numpy as np
from osgeo import gdal
from PIL import Image

def tif_to_png_tiles(tif_path, output_dir, tile_size=1024):
    """
    将TIF文件分割成指定大小的PNG图像并保存到输出目录。

    :param tif_path: 输入的TIF文件路径
    :param output_dir: 保存分割后PNG文件的目录
    :param tile_size: 分割块的大小，默认为1024
    """
    # 打开TIF文件
    dataset = gdal.Open(tif_path)
    if not dataset:
        raise FileNotFoundError(f"无法打开文件: {tif_path}")
    
    # 获取TIF文件的尺寸
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount

    # 读取影像的所有波段数据
    img_data = dataset.ReadAsArray(0, 0, width, height)

    # 如果影像有多个波段，则需要对波段进行处理
    if bands > 1:
        img_data = np.transpose(img_data, (1, 2, 0))

    # 生成分割后的PNG文件
    tile_count = 0
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            # 定义切片区域
            tile = img_data[i:i+tile_size, j:j+tile_size]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                # 跳过不完整的块（如果图像不能被整除）
                continue
            
            # 将NumPy数组转换为Pillow Image对象
            if bands > 1:
                tile_img = Image.fromarray(tile)
            else:
                tile_img = Image.fromarray(tile, mode='L')  # 单波段处理

            # 定义保存路径
            tile_filename = os.path.join(output_dir, f"tile_{i}_{j}.png")
            
            # 保存为PNG文件
            tile_img.save(tile_filename)
            tile_count += 1

    print(f"总共生成了 {tile_count} 个PNG图像。")

def tif_to_png_tiles_label(tif_path, output_dir, tile_size=1024):
    """
    将标签TIF文件（黑白二值）分割成指定大小的PNG图像并保存到输出目录。

    :param tif_path: 输入的TIF标签文件路径
    :param output_dir: 保存分割后PNG文件的目录
    :param tile_size: 分割块的大小，默认为1024
    """
    # 打开TIF文件
    dataset = gdal.Open(tif_path)
    if not dataset:
        raise FileNotFoundError(f"无法打开文件: {tif_path}")
    
    # 获取TIF文件的尺寸
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 读取影像的单波段数据（假设是单波段标签图像）
    img_data = dataset.ReadAsArray(0, 0, width, height)

    # 检查数据值并确保是二值图像 (0 和 1)
    if np.unique(img_data).tolist() not in [[0], [1], [0, 1]]:
        raise ValueError("标签图像应该是二值图像（0 和 1）。")
    
    # 将1映射为255，使其在保存为PNG时是黑白图像
    img_data = img_data * 255

    # 生成分割后的PNG文件
    tile_count = 0
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            # 定义切片区域
            tile = img_data[i:i+tile_size, j:j+tile_size]
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                # 跳过不完整的块（如果图像不能被整除）
                continue

            # 将NumPy数组转换为Pillow Image对象 (灰度图像 'L' 模式)
            tile_img = Image.fromarray(tile.astype(np.uint8), mode='L')

            # 定义保存路径
            tile_filename = os.path.join(output_dir, f"tile_{i}_{j}.png")
            
            # 保存为PNG文件
            tile_img.save(tile_filename)
            tile_count += 1

    print(f"总共生成了 {tile_count} 个PNG标签图像。")

# 使用方法
# tif_path = "D:/BaiduNetdiskDownload/change_label.tif"  # 输入TIF文件路径
# output_dir = "D:/BaiduNetdiskDownload/WHU-CD/label"  # 输出PNG文件保存目录
# os.makedirs(output_dir, exist_ok=True)

# tif_to_png_tiles(tif_path, output_dir)

tif_path = "D:/BaiduNetdiskDownload/change_label.tif"  # 输入TIF文件路径
output_dir = "D:/BaiduNetdiskDownload/WHU-CD/label"  # 输出PNG文件保存目录
os.makedirs(output_dir, exist_ok=True)

tif_to_png_tiles_label(tif_path, output_dir)
