import torch
import torch.nn.functional as F
from torch.autograd import Variable

def grad_cam(model, input_image, target_layer):
    # 前向传播
    model.eval()
    input_image = Variable(input_image, requires_grad=True)
    # output = model(input_image)

    feature_maps = []

    # 定义 hook 函数
    def hook_fn(module, input, output):
        feature_maps.append(output)

    # 注册 hook，假设你知道目标层在 model 中的具体路径
    # target_layer = model.mask_decoder.frm  # 这是一个例子，根据你的模型修改

    # 注册 hook 到目标层
    hook_handle = target_layer.register_forward_hook(hook_fn)

    # 前向传播，确保调用模型的 `forward` 函数来触发 hook
    output = model(input_image)[0]

    # 提取特征图
    feature_map = feature_maps[0][0]

    # 解除 hook（避免不必要的内存占用）
    hook_handle.remove()

    # 取目标层的特征图和梯度
    # feature_map = model.features[target_layer]
    feature_map.retain_grad()
    
    # 计算目标输出的梯度
    print(type(output))
    target_class = output.argmax(dim=1)
    output[:, target_class].backward()

    # 获取梯度和特征图
    gradients = feature_map.grad
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # 权重化特征图并生成热力图
    cam = F.relu(torch.sum(weights * feature_map, dim=1)).squeeze().cpu().data.numpy()

    # 归一化
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam