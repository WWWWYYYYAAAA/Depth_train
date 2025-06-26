import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights
from torchsummary import summary

class ResidualDoubleConv(nn.Module):
    """残差双卷积块（带通道注意力机制）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 通道注意力机制
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//16, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 应用注意力机制
        se_weight = self.se(out)
        out = out * se_weight
        
        out += residual
        return self.relu(out)

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 使用ResNet34作为主干网络（更大规模）
        resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # 提取ResNet的中间层特征
        self.encoder0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.encoder1 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        # 中间扩张卷积层（增大感受野）
        self.mid_dilated = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # 解码器上采样路径（增大通道数）
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = ResidualDoubleConv(256 + 256, 256)  # 512输入通道
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ResidualDoubleConv(128 + 128, 128)  # 256输入通道
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = ResidualDoubleConv(64 + 64, 64)  # 128输入通道
        
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = ResidualDoubleConv(32 + 64, 64)  # 96输入通道
        
        # 最终输出层
        self.final_upsample = nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def forward(self, x):
        # 编码器路径
        e0 = self.encoder0(x)     # [B, 64, 100, 100]
        e1 = self.encoder1(e0)    # [B, 64, 50, 50]
        e2 = self.encoder2(e1)    # [B, 128, 25, 25]
        e3 = self.encoder3(e2)    # [B, 256, 13, 13]
        e4 = self.encoder4(e3)    # [B, 512, 7, 7]
        
        # 中间扩张卷积层
        mid = self.mid_dilated(e4)  # [B, 512, 7, 7]
        
        # 解码器路径
        d = self.up1(mid)          # [B, 256, 14, 14]
        # 尺寸对齐
        if d.shape[2:] != e3.shape[2:]:
            d = F.interpolate(d, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d = torch.cat([d, e3], dim=1)  # [B, 512, 13, 13]
        d = self.conv1(d)         # [B, 256, 13, 13]
        
        d = self.up2(d)           # [B, 128, 26, 26]
        if d.shape[2:] != e2.shape[2:]:
            d = F.interpolate(d, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d = torch.cat([d, e2], dim=1)  # [B, 256, 25, 25]
        d = self.conv2(d)         # [B, 128, 25, 25]
        
        d = self.up3(d)           # [B, 64, 50, 50]
        if d.shape[2:] != e1.shape[2:]:
            d = F.interpolate(d, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d = torch.cat([d, e1], dim=1)  # [B, 128, 50, 50]
        d = self.conv3(d)         # [B, 64, 50, 50]
        
        d = self.up4(d)           # [B, 32, 100, 100]
        # 对齐e0尺寸（100x100）
        e0_resized = F.interpolate(e0, size=(100, 100), mode='bilinear', align_corners=True)
        d = torch.cat([d, e0_resized], dim=1)  # [B, 96, 100, 100]
        d = self.conv4(d)         # [B, 64, 100, 100]
        
        # 最终上采样到200×200
        d = self.final_upsample(d)  # [B, 64, 200, 200]
        return self.final_conv(d)   # [B, 1, 200, 200]

# 测试代码
if __name__ == "__main__":
    model = ResNetUNet(in_channels=3, out_channels=1)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 测试200×200输入输出
    x = torch.randn(2, 3, 200, 200)
    y = model(x)
    print(f"输入尺寸: {x.shape} -> 输出尺寸: {y.shape}")
