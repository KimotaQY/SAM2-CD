import torch

checkpoint_path = "D:/Checkpoints/rename/SAM2_e68_OA99.16_F91.53_IoU85.04.pth"

model = build_sam2(model_cfg, checkpoint_path)

# 1. 加载预训练权重
pretrained_weights = torch.load(checkpoint_path)

# 2. 获取旧模型的 state_dict
pretrained_dict = pretrained_weights['model']  # 假设预训练权重保存在 'state_dict' 中

# 3. 获取新模型的 state_dict
model_dict = model.state_dict()

# 4. 映射旧的权重名称到新的权重名称
new_pretrained_dict = {}

for key in pretrained_dict:
    new_key = key.replace('conv_scale', 'conv_scales.conv_scale')  # 根据调整后的模型名称进行替换
    if new_key in model_dict:  # 确保映射后的 key 存在于新模型中
        new_pretrained_dict[new_key] = pretrained_dict[key]

# 5. 更新新模型的权重
model_dict.update(new_pretrained_dict)

# 6. 加载更新后的权重到模型中
model.load_state_dict(model_dict)

# 7. 保存新的权重文件
torch.save({'model': model.state_dict()}, 'path_to_save_new_model.pth')