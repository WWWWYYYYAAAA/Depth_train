import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """双卷积块（Conv→BN→ReLU×2）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        # 编码器路径
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in features])
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # 瓶颈层（最深特征提取）
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 解码器路径
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feature in reversed(features):
            # 转置卷积上采样
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            # 双卷积块融合特征
            self.up_convs.append(DoubleConv(feature*2, feature))
        
        # 输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        # 编码器前向传播
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转用于跳跃连接
        
        # 解码器前向传播
        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            # 自适应尺寸对齐（应对边界损失）
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, skip), dim=1)  # 跳跃连接
            x = self.up_convs[idx](x)
        
        return self.final_conv(x)  # 输出单通道图像

def DepthLoss(pred, target, alpha=0.4):
    # 1. L1损失（基础精度）
    # lossf =  nn.L1Loss()
    lossf = nn.SmoothL1Loss()
    l1_loss = lossf(pred, target)
    
    # 2. 梯度差异损失（提升边缘锐度）
    grad_x_pred = pred[:, :, :-1] - pred[:, :, 1:]
    grad_y_pred = pred[:, :-1, :] - pred[:, 1:, :]
    grad_x_target = target[:, :, :-1] - target[:, :, 1:]
    grad_y_target = target[:, :-1, :] - target[:, 1:, :]
    
    grad_loss = lossf(grad_x_pred, grad_x_target) + \
                lossf(grad_y_pred, grad_y_target)
    
    # 3. 组合损失
    return (1 - alpha) * l1_loss + alpha * grad_loss

if __name__ == "__main__":

    # 测试输入输出尺寸
    model = UNet(in_channels=3, out_channels=1)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    x = torch.randn(1, 3, 200, 200)
    y = model(x)
    print(y.shape)  # torch.Size([1, 1, 100, 100])