import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        # 使用1x1卷积来减少通道数
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        # 使用1x1卷积将通道数恢复到原来的数量
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        # 对输入进行全局平均池化
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # 通过fc1卷积层
        x1 = self.fc1(x1)
        # ReLU激活
        x1 = F.relu(x1, inplace=True)
        # 通过fc2卷积层
        x1 = self.fc2(x1)
        # Sigmoid激活，得到通道注意力权重
        x1 = torch.sigmoid(x1)
        
        # 对输入进行全局最大池化
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # 通过fc1卷积层
        x2 = self.fc1(x2)
        # ReLU激活
        x2 = F.relu(x2, inplace=True)
        # 通过fc2卷积层
        x2 = self.fc2(x2)
        # Sigmoid激活，得到通道注意力权重
        x2 = torch.sigmoid(x2)
        
        # 将平均池化和最大池化得到的权重相加
        x = x1 + x2
        # 将权重张量调整为与输入相同的形状
        x = x.view(-1, self.input_channels, 1, 1)
        return x

# 定义重复块模块
class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        # 初始化通道注意力模块
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        # 定义各种不同的深度可分离卷积
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        # 1x1卷积用于进一步处理
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()  # GELU激活函数

    def forward(self, inputs):
        # 通过1x1卷积和激活函数处理输入
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        # 计算通道注意力权重
        channel_att_vec = self.ca(inputs)
        # 应用通道注意力权重
        inputs = channel_att_vec * inputs

        # 对输入进行各种深度可分离卷积
        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        # 将所有卷积结果相加
        x = x_1 + x_2 + x_3 + x_init
        # 通过1x1卷积处理卷积结果
        spatial_att = self.conv(x)
        # 应用空间注意力权重
        out = spatial_att * inputs
        # 通过1x1卷积进一步处理
        out = self.conv(out)
        return out

# 测试模块
if __name__ == "__main__":
    # 创建一个随机输入张量，形状为 (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 64, 32, 32)  # batch_size=1, channels=64, height=32, width=32
    
    # 初始化一个 RepBlock 模块
    rep_block = RepBlock(in_channels=64, out_channels=64)
    
    # 将输入张量传递通过 RepBlock 模块
    output_tensor = rep_block(input_tensor)
    
    # 打印输出张量的形状
    print("Output tensor shape:", output_tensor.shape)
