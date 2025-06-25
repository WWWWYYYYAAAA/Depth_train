import torch
import torch.nn as nn
from torchvision.models import resnet18

class DynamicSemanticAttention(nn.Module):
    """动态语义注意力模块（见图2[2](@ref)）"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, depth_feat, semantic_feat):
        # 特征压缩与矩阵乘法
        depth_feat = self.conv(depth_feat)  # [B, C/2, H, W]
        semantic_feat = self.conv(semantic_feat)
        attn = torch.matmul(depth_feat.flatten(2), semantic_feat.flatten(2).transpose(1, 2))  # [B, H*W, H*W]
        attn = self.softmax(attn)
        # 加权融合
        fused_feat = torch.matmul(attn, depth_feat.flatten(2)).view_as(depth_feat)
        return fused_feat + depth_feat

class DepthDecoder(nn.Module):
    """解码器：4层上采样模块 + 跳跃连接[1](@ref)"""
    def __init__(self, enc_channels=[64, 128, 256, 512], dec_channels=[256, 128, 64, 32]):
        super().__init__()
        self.upconvs = nn.ModuleList()
        for i in range(4):
            in_ch = enc_channels[-1] if i == 0 else dec_channels[i - 1]
            self.upconvs.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, dec_channels[i], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ELU(inplace=True),
                    nn.Conv2d(dec_channels[i], dec_channels[i], kernel_size=3, padding=1),
                    nn.ELU()
                )
            )
        self.attention = DynamicSemanticAttention(dec_channels[-1])
        self.final_conv = nn.Conv2d(dec_channels[-1], 1, kernel_size=1)

    def forward(self, enc_feats, semantic_map):
        x = enc_feats[-1]
        for i, upconv in enumerate(self.upconvs):
            x = upconv(x)
            if i < 3:  # 融合编码器同尺度特征
                x = torch.cat([x, enc_feats[-(i + 2)]], dim=1)
        x = self.attention(x, semantic_map)  # 动态语义引导
        return torch.sigmoid(self.final_conv(x))  # 输出归一化深度图

class MonoDepthNet(nn.Module):
    """主网络：编码器 + 解码器 + 深度预测头"""
    def __init__(self):
        super().__init__()
        # 编码器：ResNet-18（移除全连接层）
        backbone = resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.decoder = DepthDecoder()
        # 语义分支（轻量分割网络，可选ERFNet[2](@ref)）
        self.semantic_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.ELU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        enc_feats = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [4, 5, 6, 7]:  # 记录各层输出（layer1~4）
                enc_feats.append(x)
        semantic_map = self.semantic_head(enc_feats[-1])  # 生成语义特征图
        depth_pred = self.decoder(enc_feats, semantic_map)
        return depth_pred
    
class DepthLoss(nn.Module):
    def __init__(self, lambda_si=0.5, lambda_sem=0.2):
        super().__init__()
        self.lambda_si = lambda_si  # 尺度不变权重
        self.lambda_sem = lambda_sem  # 语义引导权重

    def forward(self, pred, target, semantic_map):
        # 尺度不变误差（Eigen et al.[3](@ref)）
        log_diff = torch.log(pred) - torch.log(target)
        si_loss = torch.sqrt(torch.mean(log_diff ** 2) - self.lambda_si * torch.mean(log_diff) ** 2)
        
        # 语义引导回归损失（按类别加权[2](@ref)）
        sem_weights = self.dynamic_weight(semantic_map)  # 动态调整各类别权重
        sem_loss = torch.mean(sem_weights * torch.abs(pred - target))
        
        return si_loss + self.lambda_sem * sem_loss

    def dynamic_weight(self, sem_map):
        """动态调整权重：大物体权重初期高，小物体后期高[2](@ref)"""
        # 简化实现：实际需按类别ID分配
        return torch.clamp(0.1 * sem_map + 0.9, min=0.5, max=2.0)