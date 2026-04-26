import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算每个类别的概率
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 预测为真实类的概率

        # 计算 Focal Loss
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


# 示例使用
if __name__ == '__main__':
    inputs = torch.randn(4, 1)  # 模型输出
    targets = torch.empty(4, 1).random_(2)  # 真实标签
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(inputs, targets)
    print(loss.item())
