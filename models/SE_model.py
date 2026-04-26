import torch
import torch.nn as nn
import torch.nn.functional as F


# SE注意力机制模块（提升模型的通道注意力能力，增强特征表示）
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        # 自适应全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 两个全连接层，sigmoid（）将输出归一到[0, 1]
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批次、通道信息
        b, c, _, _ = x.size()
        # 池化为[b, c, 1, 1]的形状
        y = self.avg_pool(x).view(b, c)
        # 全连接，输出通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # 特征图与权重相乘
        return x * y.expand_as(x)


# 残差块（缓解深度网络中的梯度消失问题）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dilation_rate):
        # 基本卷积块+SE
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.se = SEBlock(in_channels)

    def forward(self, x):
        residual = x
        # F.relu：将输入张-量中的所有负值变为0，正值保持不变
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)


# 加入SE注意力机制和残差链接的U-Net Model
class SEUNet(nn.Module):
    def __init__(self, num_of_channels=4, num_of_classes=1):
        super(SEUNet, self).__init__()

        def conv_block(in_channels, out_channels, dilation_rate):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # 编码部分的定义
        self.encoder1 = conv_block(num_of_channels, 64, 1)
        self.encoder2 = conv_block(64, 128, 2)
        self.encoder3 = conv_block(128, 256, 4)
        self.encoder4 = conv_block(256, 512, 8)

        # 最大池化与瓶颈层定义，池化窗口（2 * 2）
        self.pool = nn.MaxPool2d((2, 2))
        self.bottleneck = conv_block(512, 1024, 8)

        # 上采样部分定义
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512, 8)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256, 4)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128, 2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64, 1)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, num_of_classes, kernel_size=1)

        # 带有SE注意力机制的残差块定义
        self.res_block1 = ResidualBlock(64, 1)
        self.res_block2 = ResidualBlock(128, 2)
        self.res_block3 = ResidualBlock(256, 4)
        self.res_block4 = ResidualBlock(512, 8)

    def forward(self, x):
        # 编码部分（下采样）
        enc1 = self.encoder1(x)
        enc1 = self.res_block1(enc1)
        enc2 = self.encoder2(self.pool(enc1))
        enc2 = self.res_block2(enc2)
        enc3 = self.encoder3(self.pool(enc2))
        enc3 = self.res_block3(enc3)
        enc4 = self.encoder4(self.pool(enc3))
        enc4 = self.res_block4(enc4)

        # 瓶颈层
        bottleneck = self.bottleneck(self.pool(enc4))

        # 解码部分（上采样 + 通道拼接）
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


# Example of using the UNet model
if __name__ == "__main__":
    model = SEUNet()
    input = torch.randn(1, 4, 192, 192)
    output = model(input)
    print(output.shape)
