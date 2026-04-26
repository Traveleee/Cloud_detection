import torch
import torch.nn as nn


# 补丁嵌入
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # shape:[1, 1, embed_dim]
        # shape:[1, 196 + 1, embed_dim]
        self.position_embeddings = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x).flatten(2).transpose(1, 2)  # [B, N, embed_dim],N:num of patches
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)  # [B, N + 1, embed_dim]
        x += self.position_embeddings  # add(),添加位置编码
        return x


# transformer编码块
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            # self-attention模块
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),  # 前馈神经网络中隐藏层的维度，通常设置为d_model的几倍
                dropout=dropout,
                activation='gelu'
            )
            for _ in range(depth)  # 循环创建指定数量的Transformer编码器层
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ViT块
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        self.transformer = TransformerEncoder(embed_dim, num_heads, depth, mlp_ratio, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = x[:, 0]  # 取分类 token 的输出
        x = self.mlp_head(x)
        return x


# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer().to(device)
    img = torch.randn(1, 3, 224, 224).to(device)
    out = model(img)
    print(out.shape)  # 输出形状应为 [1, 1000]
