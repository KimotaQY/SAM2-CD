from models.sam2_cd.build_sam import build_sam2, build_sam22



import torch

def compare_model_parameters(model1, model2):
    params1 = model1.state_dict()
    params2 = model2.state_dict()

    # 如果模型参数的keys不相同，先打印不同的keys
    keys1 = set(params1.keys())
    keys2 = set(params2.keys())

    if keys1 != keys2:
        print("The models have different parameters!")
        print("Parameters in model1 but not in model2:", keys1 - keys2)
        print("Parameters in model2 but not in model1:", keys2 - keys1)

    # 对比每个相同key的参数
    # for key in keys1 & keys2:
    #     param1 = params1[key]
    #     param2 = params2[key]
    #     if not torch.equal(param1, param2):
    #         print(f"Parameter '{key}' differs between the models.")
    #         print(f"Model1 parameter shape: {param1.shape}")
    #         print(f"Model2 parameter shape: {param2.shape}")

    #         # 打印具体的差异（可以根据实际需要选择打印多少内容）
    #         diff = param1 - param2
    #         print(f"Difference for '{key}':")
    #         print(diff)

# 示例用法
# model1 = Model1()
# model2 = Model2()
# compare_model_parameters(model1, model2)

model_cfg = "sam2_configs/kan_sam2_hiera_l.yaml"
model1 = build_sam2(model_cfg)
model2 = build_sam22(model_cfg)
compare_model_parameters(model1, model2)