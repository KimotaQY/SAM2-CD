import torch
import torch.nn as nn
import torch.nn.functional as F
# from .backbone.regnet import DenseNetWrapper
# import MobileNetV2
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from torch import Tensor
from torch.nn import Module, ModuleList, Sigmoid
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)
        
    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CCALayer, self).__init__()

        self.esa = EfficientSelfAttention(dim=64,heads=2,reduction_ratio=reduction)

        hidden_dim = channel * 8
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channel, 1),
        )
    def forward(self, x):#v3
        y = self.esa(x) 
        y = self.conv3(y)
        return y
        # return self.relu(y+x)


# class ESE(torch.nn.Module):
#     """This is adapted from the efficientnet Squeeze Excitation. The idea is to not
#     squeeze the number of channels to keep more information."""

#     def __init__(self, channel: int) -> None:
#         super().__init__()
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.fc = torch.nn.Conv2d(channel, channel, kernel_size=1)  # (Linear)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.avg_pool(x)
#         out = self.fc(out)
#         return torch.sigmoid(out) * x

class SIMDB(nn.Module):#OLD
    def __init__(self, in_channels, distillation_rate=0.25):
        super(SIMDB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        # self.remaining_channels = int(in_channels - self.distilled_channels)
        # self.c1 = Conv2dSWL(in_channels, in_channels, 2)
        # self.c2 = Conv2dSWR(self.remaining_channels, in_channels, 2)
        # self.c3 = Conv2dSWU(self.remaining_channels, in_channels, 2)
        # self.c4 = Conv2dSWD(self.remaining_channels, self.distilled_channels, 2)


        # self.c1=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.c2=nn.Conv2d(self.remaining_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.c3=nn.Conv2d(self.remaining_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # self.c4=nn.Conv2d(self.remaining_channels, self.distilled_channels, kernel_size=3, stride=1, padding=1)
        self.c1=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.c2=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.c3=nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.c4=nn.Conv2d(in_channels, self.distilled_channels, kernel_size=3, stride=1, padding=1)
        self.act = activation('lrelu', neg_slope=0.05)
        # self.c5 = conv_layer(in_channels, in_channels, 1)
        self.c6 = conv_layer(in_channels, self.distilled_channels, 1)
        # self.cca = CCALayer(self.distilled_channels * 4)
        # self.esa = EfficientSelfAttention(dim=64,heads=2,reduction_ratio=4)
        # self.se=SEModule(64)
        # self.cbam = CBAM(64)
        self.relu=nn.ReLU(inplace=True)
        # self.conv_aggregation= FeatureFusionModule(64,64, 64)
    def forward(self, input):#m:cca改成了semodule split改成普通的
        out_c1 = self.act(self.c1(input))
        # distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(out_c1))
        # distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(out_c2))
        # distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(out_c3)
        out = torch.cat((self.c6(out_c1), self.c6(out_c2), self.c6(out_c3), out_c4), dim=1)
        
        # out_fused = self.c5(self.cbam(out)) + input
        out_fused = self.relu(out + input)
        # out_fused = self.conv_aggregation(out , input)#p2
        return out_fused
# import MobileNetV2
# class SEModule(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16):
#         super(SEModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction_ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction_ratio, in_channels),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return y
class FPN(Module):
    def __init__(
        self,
        ch_in,  # 输入张量x和y的通道数
        ch_out: int,  # 输出张量的通道数
    ):
        super().__init__()
        if ch_in is None:
            ch_in = [16, 24, 32, 96, 320]
        self.ch_in = ch_in
        self.ch_out=ch_out
        self.ch3=64
        self.ch2=32
        self.ch1=16
        self.chfusein=144
        # self.ch3=32
        # self.ch2=32
        # self.ch1=32
        # self.chfusein=128
        # self.relu = nn.ReLU(inplace=True)
        # self.conv_c2=nn.Conv2d(self.ch_in[0], self.ch_out, kernel_size=1)
        # self.conv_c3=nn.Conv2d(self.ch_in[1], self.ch_out, kernel_size=1)
        # self.conv_c4=nn.Conv2d(self.ch_in[2], self.ch_out, kernel_size=1)
        # self.conv_c5=nn.Conv2d(self.ch_in[3], self.ch_out, kernel_size=1)


        self.convs1_1=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[0], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True)
        )
        self.convs1_2=nn.Sequential(
            nn.Conv2d(self.ch_in[1], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs1_3=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs1_4=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)  # 添加上采样层
        )

        self.convs2_1=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[0], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True)
        )
        self.convs2_2=nn.Sequential(
            nn.Conv2d(self.ch_in[1], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs2_3=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs2_4=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 添加上采样层
        )


        self.convs3_1=nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.ch_in[0], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True)
        )
        self.convs3_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[1], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs3_3=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs3_4=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )

        self.convs4_1=nn.Sequential(
            nn.MaxPool2d(kernel_size=8, stride=8),
            nn.Conv2d(self.ch_in[0], self.ch1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch1),
            nn.ReLU(inplace=True)
        )
        self.convs4_2=nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(self.ch_in[1], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs4_3=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[2], self.ch2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        self.convs4_4=nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.ch_in[3], self.ch3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.ch3),
            nn.ReLU(inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 添加上采样层
        )
        # self.conv_fuse = nn.Sequential(
        #     nn.Conv2d(self.chfusein, self.ch_out, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.ch_out),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.ch_out, self.ch_out, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.ch_out),
        #     nn.ReLU(inplace=True)
        # )

        self.conv_aggregation_s1= FeatureFusionModule(self.chfusein, self.ch_in[0], self.ch_out)
        self.conv_aggregation_s2= FeatureFusionModule(self.chfusein, self.ch_in[1], self.ch_out)
        self.conv_aggregation_s3= FeatureFusionModule(self.chfusein, self.ch_in[2], self.ch_out)
        self.conv_aggregation_s4= FeatureFusionModule(self.chfusein, self.ch_in[3], self.ch_out)

    def forward(self,c1,c2,c3,c4) -> Tensor:#v2.0
        s1_c1=self.convs1_1(c1)
        s1_c2=self.convs1_2(c2)
        s1_c3=self.convs1_3(c3)
        s1_c4=self.convs1_4(c4)
        s1 = self.conv_aggregation_s1(torch.cat([s1_c1, s1_c2,s1_c3,s1_c4], dim=1), c1)
        # s1=self.conv_fuse(torch.cat([s1_c1, s1_c2,s1_c3,s1_c4], dim=1))

        s2_c1=self.convs2_1(c1)
        s2_c2=self.convs2_2(c2)
        s2_c3=self.convs2_3(c3)
        s2_c4=self.convs2_4(c4)
        # s2=self.conv_fuse(torch.cat([s2_c1, s2_c2,s2_c3,s2_c4], dim=1))
        s2 = self.conv_aggregation_s2(torch.cat([s2_c1, s2_c2,s2_c3,s2_c4], dim=1), c2)
        
        s3_c1=self.convs3_1(c1)
        s3_c2=self.convs3_2(c2)
        s3_c3=self.convs3_3(c3)
        s3_c4=self.convs3_4(c4)
        # s3=self.conv_fuse(torch.cat([s3_c1, s3_c2,s3_c3,s3_c4], dim=1))
        s3 = self.conv_aggregation_s3(torch.cat([s3_c1, s3_c2,s3_c3,s3_c4], dim=1), c3)

        s4_c1=self.convs4_1(c1)
        s4_c2=self.convs4_2(c2)
        s4_c3=self.convs4_3(c3)
        s4_c4=self.convs4_4(c4)

        # s4=self.conv_fuse(torch.cat([s4_c1, s4_c2,s4_c3,s4_c4], dim=1))
        s4 = self.conv_aggregation_s4(torch.cat([s4_c1, s4_c2,s4_c3,s4_c4], dim=1), c4)

        return s1,s2,s3,s4



class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        # self.esa = EfficientSelfAttention(dim=64,heads=2,reduction_ratio=4)
        # self.se_module = SEModule(self.out_d, reduction_ratio=16)
        # self.cbam = CBAM(64)
    def forward(self, c_fuse, c):#1.0 
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse * self.conv_identity(c))
        # c_out = self.relu(c_fuse * self.se_module(self.conv_identity(c)))
        return c_out

    # def forward(self, c_fuse, c):#2.0 
    #     c_fuse = self.conv_fuse(c_fuse)
    #     c_out = self.relu(c_fuse * self.esa(self.conv_identity(c)))

    #     return c_out


class FFm(nn.Module):
    def __init__(self, in_d=64, out_d=64):
        super(FFm, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid=nn.Sigmoid()
        self.conv1=nn.Conv2d(16, self.in_d, kernel_size=1)


        self.msff = SIMDB(64)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # self.sigmoid=nn.Sigmoid()
        # self.bn=nn.BatchNorm2d(self.in_d)
    # def featureFuse0(self,x1,x2):#mul
    #     df=torch.abs(x1 - x2)
    #     df=self.conv1(df)
    #     c1=self.msff(df)
    #     return c1
    def featureFuse(self,x1,x2):#mul
        df=torch.abs(x1 - x2)
        c1=self.msff(df)
        return c1

    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V6p0
    #     # temporal fusion
    #     c2 = self.featureFuse(x1_2, x2_2)
    #     c3 = self.featureFuse(x1_3, x2_3)
    #     c4 = self.featureFuse(x1_4, x2_4)
    #     c5 = self.featureFuse(x1_5, x2_5)

    #     return c2, c3, c4, c5
    

    
    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#v2
    #     # temporal fusion
    #     c2 = self.featureFuse(x1_2, x2_2)
    #     c3 = self.featureFuse(x1_3, x2_3)*self.sigmoid(self.avg_pool(c2))
    #     c4 = self.featureFuse(x1_4, x2_4)*self.sigmoid(self.avg_pool(c3))
    #     c5 = self.featureFuse(x1_5, x2_5)*self.sigmoid(self.avg_pool(c4))

    #     return c2, c3, c4, c5
    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V3
    #     # temporal fusion
    #     c2 = self.featureFuse(x1_2, x2_2)
    #     c3 = self.conv3(self.featureFuse(x1_3, x2_3)+0.2*self.avg_pool(c2))
    #     c4 = self.conv3(self.featureFuse(x1_4, x2_4)+0.2*self.avg_pool(c3))
    #     c5 = self.conv3(self.featureFuse(x1_5, x2_5)+0.2*self.avg_pool(c4))

    #     return c2, c3, c4, c5
    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V4
    #     # temporal fusion
    #     c2 = self.featureFuse(x1_2, x2_2)
    #     c3 = self.featureFuse(x1_3, x2_3)+self.avg_pool(c2)
    #     c4 = self.featureFuse(x1_4, x2_4)+self.avg_pool(c3)
    #     c5 = self.featureFuse(x1_5, x2_5)+self.avg_pool(c4)

    #     return c2, c3, c4, c5
    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V5p1
    #     # temporal fusion
    #     c2=torch.abs(x1_2 - x2_2) #v5
    #     # c2 = self.featureFuse(x1_2, x2_2)#v5p1
    #     c3=torch.abs(x1_3 - x2_3)+self.avg_pool(self.msff(c2))
    #     c4=torch.abs(x1_4 - x2_4)+self.avg_pool(self.msff(c3))
    #     c5=torch.abs(x1_5 - x2_5)+self.avg_pool(self.msff(c4))
    #     return c2, c3, c4, c5
    
    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V6p1
    #     # temporal fusion
    #     # c2=torch.abs(x1_2 - x2_2)
    #     c2 = self.featureFuse(x1_2, x2_2)
    #     c3=torch.abs(x1_3 - x2_3)*self.sigmoid(self.avg_pool(c2))
    #     c4=torch.abs(x1_4 - x2_4)*self.sigmoid(self.avg_pool(self.msff(c3)))
    #     c5=torch.abs(x1_5 - x2_5)*self.sigmoid(self.avg_pool(self.msff(c4)))
    #     return c2, c3, c4, c5
    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V6p2
        # temporal fusion
        c2=torch.abs(x1_2 - x2_2)
        # c2 = self.featureFuse(x1_2, x2_2)
        c3=torch.abs(x1_3 - x2_3)*self.sigmoid(self.avg_pool(self.msff(c2)))
        c4=torch.abs(x1_4 - x2_4)*self.sigmoid(self.avg_pool(self.msff(c3)))
        c5=torch.abs(x1_5 - x2_5)*self.sigmoid(self.avg_pool(self.msff(c4)))
        return c2, c3, c4, c5
    # def forward(self,x1, x1_2, x1_3, x1_4, x1_5, x2,x2_2, x2_3, x2_4, x2_5):#V6p2
    #     # temporal fusion
    #     c1 = self.featureFuse0(x1, x2)
    #     c2=torch.abs(x1_2 - x2_2)*self.sigmoid(self.avg_pool(self.msff(c1)))
    #     # c2 = self.featureFuse(x1_2, x2_2)
    #     c3=torch.abs(x1_3 - x2_3)*self.sigmoid(self.avg_pool(self.msff(c2)))
    #     c4=torch.abs(x1_4 - x2_4)*self.sigmoid(self.avg_pool(self.msff(c3)))
    #     c5=torch.abs(x1_5 - x2_5)*self.sigmoid(self.avg_pool(self.msff(c4)))
    #     return c2, c3, c4, c5
    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V7
    #     # temporal fusion
    #     c2=torch.abs(x1_2 - x2_2)
    #     # c2 = self.featureFuse(x1_2, x2_2)
    #     c3=(torch.abs(x1_3 - x2_3)+self.avg_pool(self.msff(c2)))*self.sigmoid(self.avg_pool(self.msff(c2)))
    #     c4=(torch.abs(x1_4 - x2_4)+self.avg_pool(self.msff(c3)))*self.sigmoid(self.avg_pool(self.msff(c3)))
    #     c5=(torch.abs(x1_5 - x2_5)+self.avg_pool(self.msff(c4)))*self.sigmoid(self.avg_pool(self.msff(c4)))

    #     return c2, c3, c4, c5
    # def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):#V8
    #     # temporal fusion
    #     c2=torch.abs(x1_2 - x2_2)
    #     # c2 = self.featureFuse(x1_2, x2_2)
    #     c3=torch.abs(x1_3 - x2_3)*self.sigmoid(self.avg_pool(self.msff(c2)))+self.avg_pool(self.msff(c2))
    #     c4=torch.abs(x1_4 - x2_4)*self.sigmoid(self.avg_pool(self.msff(c3)))+self.avg_pool(self.msff(c3))
    #     c5=torch.abs(x1_5 - x2_5)*self.sigmoid(self.avg_pool(self.msff(c4)))+self.avg_pool(self.msff(c4))
    #     return c2, c3, c4, c5
class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        

        self.cca = CCALayer(64)

    #     return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5
    def forward(self, d2, d3, d4, d5):#w2
        # high-level
        # mask_p5=self.cls(d5)
        # d5=self.conv_p2(d5)
        p5  = self.cca(d5)
        mask_p5=self.cls(p5)  

        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))
        # mask_p4=self.cls(p4)
        p4 = self.cca(p4)
        mask_p4=self.cls(p4)

        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))
        # mask_p3=self.cls(p3)
        p3 = self.cca(p3)
        mask_p3=self.cls(p3)
        
        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        p2 = self.cca(p2)
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5

class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        # self.backbone = DenseNetWrapper()
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2
        
        # self.ccf=FPN([224,448,896,2016],64)#8g
        # self.ccf=FPN([72,216,576,1512],64)#3.2g
        self.ccf=FPN([48,120,336,888],64)#1.6g
        # self.ccf=FPN([64,144,320,784],64)#800mf    
        # self.ccf=FPN([48,104,208,440],64)#400mf
        
        self.ffm=FFm(self.mid_d, self.en_d * 2)
        self.decoder = Decoder(self.en_d * 2)
        # self.msff = SIMDB(64)
    def forward(self, x1, x2):
        # print("x1",x1.shape)
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4 = self.backbone(x2)
        # aggregation
        x1_2, x1_3, x1_4, x1_5 = self.ccf(x1_1, x1_2, x1_3, x1_4)
        x2_2, x2_3, x2_4, x2_5 = self.ccf(x2_1, x2_2, x2_3, x2_4)#NAM
        c2, c3, c4, c5 = self.ffm(x1_2, x1_3, x1_4, x1_5,x2_2, x2_3, x2_4, x2_5)
        # fpn
        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(c2, c3, c4, c5)
        # change map
        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')


        return mask_p2, mask_p3, mask_p4, mask_p5
