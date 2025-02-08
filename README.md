# 📖 项目概述

**论文标题**  
《SAM2-CD: Remote Sensing Image Change Detection with SAM2》  

**作者**  
Your Name, Co-Author Name  

**发表信息**  
- 会议/期刊：  
- 发表日期： Yuan Qin,Chaoting Wang,Yuanyuan Fan,Chanling Pan 
- DOI: [10.xxxx/xxxx](链接)  
- 论文链接: [PDF](链接) | [arXiv](链接)  

**项目简介**  
本项目是论文《SAM2-CD: Remote Sensing Image Change Detection with SAM2》的官方开源实现，包含实验代码、数据集和复现指南。  
🔬 研究领域：Remote Sensing, Deep Learning, Change Detection  
💡 核心贡献：

---

## 🛠️ 项目状态

**当前状态**  
✅ **代码可用性**：本仓库包含论文中所有实验的完整实现
✅ **数据集**：提供实验所需的数据集或生成脚本
✅ **复现性**：已通过测试，可复现论文中的主要结果

**维护说明**  
🚧 本项目处于 **维护模式**，主要修复关键问题，不再添加新功能。  
📅 最后更新时间：2025-02-08  
📬 如有问题，请通过 Issues 或邮件联系作者。

---

## 📂 仓库结构

```plaintext
.
├── configs/             # 配置文件
├── datasets/            # 数据集加载方式
├── models/              # sam2核心算法实现
├── sam2_configs/        # sam2配置文件
├── outputs/             # 结果文件
├── utils/               # 部分核心代码
├── README.md            # 本文件
├── requirements.txt     # 依赖库

```

---

# 🚀 快速开始

## 环境配置

# 克隆仓库
git clone https://github.com/KimotaQY/SAM2-CD.git
cd SAM2-CD

# 安装依赖
pip install -r requirements.txt

## 运行示例
python train.py

## 数据集和检查点
1. LEVIR-CD：(链接)
2. WHU-CD：(链接)
3. Checkpoints: (链接)

---

# 📬 联系作者