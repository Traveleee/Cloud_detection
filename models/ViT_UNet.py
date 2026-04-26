import torch
import torch.nn as nn
import torch.nn.functional as F


# 补丁嵌入
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=4, patch_size=16, embed_dim=1024, img_size=96):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x += self.position_embeddings
        return x


# 转换器编码块12
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=16, depth=16, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return outputs


# 编码块12
class ViTEncoder(nn.Module):
    def __init__(self, img_size=96, patch_size=16, in_channels=4, embed_dim=1024, num_heads=16,
                 depth=16, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        self.transformer = TransformerEncoder(embed_dim, num_heads, depth, mlp_ratio, dropout)

    def forward(self, x):
        x = self.patch_embed(x)
        transformer_outputs = self.transformer(x)
        return [out[:, 1:].transpose(1, 2).reshape(x.size(0), -1, int(x.size(1) ** 0.5), int(x.size(1) ** 0.5)) for out in transformer_outputs]


# 网络架构
class UNetViT(nn.Module):
    def __init__(self, img_size=96, in_channels=4, num_classes=1):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size=16, in_channels=in_channels,
                                  embed_dim=1024, num_heads=16, depth=16)

        # 解码器使用卷积层逐渐恢复特征
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(1024 + 1024, 512, kernel_size=2, stride=2),  # 调整输入通道数
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1024 + 512, 256, kernel_size=2, stride=2),  # 调整输入通道数
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(1024 + 256, 128, kernel_size=2, stride=2),  # 调整输入通道数
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        ])

        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码阶段
        enc_outputs = self.encoder(x)

        # 解码阶段，将跳跃连接与解码器连接
        dec_out = enc_outputs[-1]  # 使用最后一层编码器输出开始解码
        for i, decoder_block in enumerate(self.decoder_blocks):
            # 从倒数第二层开始往上取跳跃连接
            if i < len(enc_outputs) - 1:
                skip_connection = enc_outputs[-(i + 2)]

                # 调整 skip_connection 大小以匹配 dec_out
                skip_connection = F.interpolate(skip_connection, size=dec_out.shape[2:], mode='bilinear',
                                                align_corners=False)

                # 拼接跳跃连接
                dec_out = torch.cat([dec_out, skip_connection], dim=1)

            # 解码器块输出
            dec_out = decoder_block(dec_out)

        dec_out = self.final_conv(dec_out)
        return F.interpolate(dec_out, size=x.shape[2:], mode='bilinear', align_corners=False)


# 测试代码
if __name__ == "__main__":
    model = UNetViT()
    img = torch.randn(1, 4, 384, 384)
    output = model(img)
    print(output.shape)  # 输出形状应为 [1, 1, 384, 384]
