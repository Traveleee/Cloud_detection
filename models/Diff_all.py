import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from models.DIFF_transformer import DiffTransformerLayer, RMSNorm, SwiGLU


class DownBlock(nn.Module):
    """适配图像数据的下采样块"""

    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super().__init__()
        self.out_channels = out_channels  # 保存为实例变量
        self.maxpool = nn.MaxPool2d(2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.GroupNorm(4, out_channels)
        )

        # Transformer输入处理（保持四维格式）
        self.patch_embed = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 保持通道数
            Rearrange('b c h w -> b (h w) c')
        )

        # 位置编码处理
        self.pos_proj = nn.Linear(2, out_channels)

        # Transformer层
        self.transformers = nn.Sequential(*[
            DiffTransformerLayer(
                d_model=out_channels,
                num_heads=num_heads,
                lambda_init=0.8 - 0.6 * math.exp(-0.3 * (l - 1))
            ) for l in range(1, num_layers + 1)
        ])

        # 修改后的重建层，仅包含线性变换
        self.rebuild = nn.Linear(out_channels, out_channels)

        # 后处理层
        self.norm = RMSNorm(out_channels, mode='image')
        self.act = SwiGLU(out_channels)

    def forward(self, x):
        # 维度检查
        if x.dim() != 4:
            raise ValueError(f"输入维度异常，应为四维张量，实际维度：{x.shape}")

        # 下采样路径
        x = self.maxpool(x)
        x = self.proj(x)  # [B, C, H, W]

        # 获取空间尺寸
        B, C, H, W = x.shape

        # 序列化处理
        x_seq = self.patch_embed(x)  # [B, H*W, C]

        # 位置编码
        pos_embed = self.get_2d_pos_embed(H, W, x.device)  # [H*W, 2]
        pos_embed = pos_embed.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]
        pos_embed = self.pos_proj(pos_embed)  # [B, H*W, C]
        x_seq += pos_embed

        # Transformer处理
        x_trans = self.transformers(x_seq)  # [B, H*W, C]

        # 四维重建
        x_trans = self.rebuild(x_trans)  # [B, H*W, C]
        x_trans = rearrange(x_trans, 'b (h w) c -> b c h w', h=H, w=W)  # 动态调整形状

        # 最终处理
        return self.act(self.norm(x_trans))

    def get_2d_pos_embed(self, height, width, device):
        """生成归一化网格坐标"""
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., height, device=device),
            torch.linspace(-1., 1., width, device=device),
            indexing='xy'
        ), dim=-1)
        return grid.reshape(-1, 2)


# 保留U-Net原有组件定义
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            RMSNorm(out_channels, mode='image'),
            SwiGLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            RMSNorm(out_channels, mode='image'),
            SwiGLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DiffUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_dim=16, num_heads=4, num_layers=3):
        super().__init__()

        # 初始卷积
        self.inc = DoubleConv(in_channels, base_dim)

        # 下采样路径
        self.down1 = DownBlock(base_dim, base_dim * 2, num_heads, num_layers)
        self.down2 = DownBlock(base_dim * 2, base_dim * 4, num_heads, num_layers)
        self.down3 = DownBlock(base_dim * 4, base_dim * 8, num_heads, num_layers)
        self.down4 = DownBlock(base_dim * 8, base_dim * 16, num_heads, num_layers)

        # 上采样路径
        self.up1 = Up(base_dim * 16, base_dim * 8)
        self.up2 = Up(base_dim * 8, base_dim * 4)
        self.up3 = Up(base_dim * 4, base_dim * 2)
        self.up4 = Up(base_dim * 2, base_dim)

        # 输出层
        self.outc = OutConv(base_dim, out_channels)

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 解码器
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# 测试代码
if __name__ == "__main__":
    model = DiffUNet().cuda()
    test_case = torch.rand(1, 4, 96, 96).cuda()
    out = model(test_case)
    print(out.shape)
