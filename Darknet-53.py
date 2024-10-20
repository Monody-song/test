# 导入库
import torch
from torch import nn
import torch.nn.functional as F


# 定义卷积层：经过这个层仅变换通道数
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding):
        super(ConvolutionalLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernal_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


# 残差块结构
class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()
        self.reseblock = nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, kernal_size=1, stride=1, padding=0),
            ConvolutionalLayer(in_channels // 2, in_channels, kernal_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return x + self.reseblock(x)

# 定义残差块叠加层
class wrapLayer(nn.Module):
    def __init__(self, in_channels, count):
        super(wrapLayer, self).__init__()
        self.count = count
        self.in_channels = in_channels
        self.res = ResidualLayer(self.in_channels)

    def forward(self, x):
        for i in range(0, self.count):
            x = self.res(x)
        return x

# 卷积集合块 卷积神经网络
class ConvolutionalSetLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSetLayer, self).__init__()
        self.sub_module = nn.Sequential(
            # 讲解中卷积神经网络的的5个卷积
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.sub_module(x)

# 下采样层：既改变通道数，也改变形状
class DownSampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleLayer, self).__init__()
        self.conv = nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, kernal_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


# 上采样层
class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')

# DarkNet模型
class DarkNet53(nn.Module):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.feature_52 = nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),
            DownSampleLayer(32, 64),
            ResidualLayer(64),
            DownSampleLayer(64, 128),
            wrapLayer(128, 2),
            DownSampleLayer(128, 256),
            wrapLayer(256, 8)
        )
        self.feature_26 = nn.Sequential(
            DownSampleLayer(256, 512),
            wrapLayer(512, 8)
        )
        self.feature_13 = nn.Sequential(
            DownSampleLayer(512, 1024),
            wrapLayer(1024, 4)
        )
        self.convolset_13 = nn.Sequential(
            ConvolutionalSetLayer(1024, 512)
        )
        self.convolset_26 = nn.Sequential(
            ConvolutionalSetLayer(768, 256)
        )
        self.convolset_52 = nn.Sequential(
            ConvolutionalSetLayer(384, 128)
        )
        self.detection_13 = nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, 15, 1, 1, 0)
        )
        self.detection_26 = nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            nn.Conv2d(512, 15, 1, 1, 0)
        )
        self.detection_52 = nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            nn.Conv2d(256, 15, 1, 1, 0)
        )
        self.up_26 = nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            UpSampleLayer()
        )
        self.up_52 = nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            UpSampleLayer()
        )

    def forward(self, x):
        h_52 = self.feature_52(x)
        h_26 = self.feature_26(h_52)
        h_13 = self.feature_13(h_26)
        conval_13 = self.convolset_13(h_13)
        detection_13 = self.detection_13(conval_13)
        up_26 = self.up_26(conval_13)
        route_26 = torch.cat((up_26, h_26), dim=1)
        conval_26 = self.convolset_26(route_26)
        detection_26 = self.detection_26(conval_26)
        up_52 = self.up_52(conval_26)
        route_52 = torch.cat((up_52, h_52), dim=1)
        conval_52 = self.convolset_52(route_52)
        detection_52 = self.detection_52(conval_52)

        return detection_13, detection_26, detection_52


model = DarkNet53()

if __name__ == '__main__':
    # 输入一张416*416的图片，batch输入维度为（1，3，416，416）
    batch = torch.randn(1, 3, 416, 416)
    detection_13, detection_26, detection_52 = model(batch)
    print(detection_13.shape)
    print(detection_26.shape)
    print(detection_52.shape)
    """
    torch.Size([1, 15, 13, 13])
    torch.Size([1, 15, 26, 26])
    torch.Size([1, 15, 52, 52])
    """