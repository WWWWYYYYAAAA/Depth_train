import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthLossL3(nn.Module):
    def __init__(self, alpha=0.8, rank_margin=0.05, device="cuda:0"):
        super().__init__()
        self.alpha = alpha      # MAE vs Grad 平衡因子
        self.rank_margin = rank_margin
        self.device = device
        
    def forward(self, pred, target):
        # 基础MAE损失
        l1_loss = F.l1_loss(pred, target)
        # print(pred.max(), target.max())
        # 梯度匹配损失
        grad_x_pred = pred[:, :, :-1] - pred[:, :, 1:]
        grad_y_pred = pred[:, :-1, :] - pred[:, 1:, :]
        grad_x_target = target[:, :, :-1] - target[:, :, 1:]
        grad_y_target = target[:, :-1, :] - target[:, 1:, :]
        grad_loss = F.l1_loss(grad_x_pred, grad_x_target) + \
                    F.l1_loss(grad_y_pred, grad_y_target)
        
        # 排序损失（随机采样1000个点对）
        n, h, w = pred.shape
        pts_a = torch.randint(0, h * w, (n, 1000)).to(self.device)
        pts_b = torch.randint(0, h * w, (n, 1000)).to(self.device)
        depth_a = pred.view(n, -1).gather(1, pts_a)
        depth_b = pred.view(n, -1).gather(1, pts_b)
        target_a = target.view(n, -1).gather(1, pts_a)
        target_b = target.view(n, -1).gather(1, pts_b)
        
        rank_sign = torch.sign(target_a - target_b)
        rank_loss = F.relu(self.rank_margin - rank_sign * (depth_a - depth_b)).mean()
        
        return l1_loss + self.alpha * grad_loss + 0.3 * rank_loss