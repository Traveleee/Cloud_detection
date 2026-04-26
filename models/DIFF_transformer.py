import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """均方根归一化"""

    def __init__(self, d, eps=1e-10, mode='image'):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.scale = nn.Parameter(
            torch.ones(1, d, 1, 1) if mode == 'image'
            else torch.ones(1, 1, d)
        )

    def forward(self, x):
        if self.mode == 'image':
            # 图像模式在通道维度计算
            norm = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        else:
            # 序列模式在最后一个维度计算
            norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.scale


class SwiGLU(nn.Module):
    """支持多模态输入的SwiGLU"""

    def __init__(self, d_model, mode='image'):
        super().__init__()
        self.mode = mode

        # 根据模式选择参数形状
        if mode == 'image':
            self.WG = nn.Conv2d(d_model, d_model * 2, 1)  # 图像模式使用卷积
            self.W1 = nn.Conv2d(d_model, d_model * 2, 1)
            self.W2 = nn.Conv2d(d_model * 2, d_model, 1)
        else:
            self.WG = nn.Linear(d_model, d_model * 2)  # 序列模式使用线性层
            self.W1 = nn.Linear(d_model, d_model * 2)
            self.W2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # 自动检测输入维度
        if x.dim() == 4:  # 图像数据 [B, C, H, W]
            g = F.silu(self.WG(x))  # SiLU激活函数结合了ReLU和Sigmoid的优点
            z = self.W1(x)
            return self.W2(g * z)  # 对应元素相乘
        elif x.dim() == 3:  # 序列数据 [B, N, C]
            # 转换为序列模式处理
            return self.process_sequence(x)
        else:
            raise ValueError(f"不支持的输入维度：{x.dim()}")

    def process_sequence(self, x):
        """处理三维序列数据"""
        # 保持维度一致性 [B, N, C] -> [B, N, 2C]
        g = F.silu(self.WG(x))  # 门控信号
        z = self.W1(x)  # 值信号
        return self.W2(g * z)  # 门控融合


class MultiHeadDifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Mechanism.
    Replaces the conventional softmax attention with a differential attention.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model. Must be divisible by num_heads.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda.λ
        """
        super().__init__()
        assert d_model % num_heads == 0  # 断言语句

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Linear projections for queries, keys, and values
        # Project to 2 * d_head per head for differential attention
        self.W_q = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_k = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_v = nn.Linear(d_model, 2 * self.d_head * num_heads, bias=False)
        self.W_o = nn.Linear(2 * self.d_head * num_heads, d_model, bias=False)

        # Learnable parameters for lambda reparameterization
        self.lambda_q1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k1 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_q2 = nn.Parameter(torch.randn(num_heads, self.d_head))
        self.lambda_k2 = nn.Parameter(torch.randn(num_heads, self.d_head))

        self.lambda_init = lambda_init

        # Scale parameter for RMSNorm
        self.rms_scale = nn.Parameter(torch.ones(2 * self.d_head))
        self.eps = 1e-5  # Epsilon for numerical stability

        # Initialize weights (optional but recommended)
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize parameters for improved training stability.
        """
        nn.init.xavier_uniform_(self.W_q.weight)  # PyTorch中用于Xavier均匀分布初始化的函数
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.rms_scale, 1.0)  # nn.init.constant_是一个原地操作，它直接修改输入张量的值

    def forward(self, X):
        """
        Forward pass for Multi-Head Differential Attention.

        Args:
            X (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after applying differential attention.
        """
        batch, N, d_model = X.shape

        # Project inputs to queries, keys, and values
        Q = self.W_q(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        K = self.W_k(X)  # Shape: (batch, N, 2 * num_heads * d_head)
        V = self.W_v(X)  # Shape: (batch, N, 2 * num_heads * d_head)

        # Reshape重塑 and permute交换 for multi-head attention
        # New shape: (batch, num_heads, sequence_length, 2 * d_head)
        # .view 方法允许你改变张量的形状，而不改变其数据
        Q = Q.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        K = K.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)
        V = V.view(batch, N, self.num_heads, 2 * self.d_head).transpose(1, 2)

        # Split Q and K into Q1, Q2 and K1, K2
        Q1, Q2 = Q.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)
        K1, K2 = K.chunk(2, dim=-1)  # Each of shape: (batch, num_heads, N, d_head)

        # Compute lambda using reparameterization
        # lambda_val = exp(lambda_q1 . lambda_k1) - exp(lambda_q2 . lambda_k2) + lambda_init
        # Compute dot products for each head
        # Shape of lambda_val: (num_heads,)
        lambda_q1_dot_k1 = torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()  # (num_heads,)
        lambda_q2_dot_k2 = torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()  # (num_heads,)
        lambda_val = torch.exp(lambda_q1_dot_k1) - torch.exp(lambda_q2_dot_k2) + self.lambda_init  # (num_heads,)

        # Expand lambda_val to match attention dimensions
        # Shape: (batch, num_heads, 1, 1)
        lambda_val = lambda_val.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # Compute attention scores
        scaling = 1 / math.sqrt(self.d_head)
        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * scaling  # (batch, num_heads, N, N)

        # Apply softmax to get attention weights
        attention1 = F.softmax(A1, dim=-1)  # (batch, num_heads, N, N)
        attention2 = F.softmax(A2, dim=-1)  # (batch, num_heads, N, N)
        attention = attention1 - lambda_val * attention2  # (batch, num_heads, N, N)

        # Apply attention weights to values
        O = torch.matmul(attention, V)  # (batch, num_heads, N, 2 * d_head)

        # Normalize each head independently using RMSNorm
        # First, reshape for RMSNorm
        O_reshaped = O.contiguous().view(batch * self.num_heads, N, 2 * self.d_head)  # (batch*num_heads, N, 2*d_head)

        # Compute RMSNorm
        rms_norm = torch.sqrt(O_reshaped.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (batch*num_heads, N, 1)
        O_normalized = (O_reshaped / rms_norm) * self.rms_scale  # (batch*num_heads, N, 2*d_head)

        # Reshape back to (batch, num_heads, N, 2 * d_head)
        O_normalized = O_normalized.view(batch, self.num_heads, N, 2 * self.d_head)

        # Scale the normalized output
        O_normalized = O_normalized * (1 - self.lambda_init)  # Scalar scaling

        # Concatenate all heads
        # New shape: (batch, N, num_heads * 2 * d_head)
        O_concat = O_normalized.transpose(1, 2).contiguous().view(batch, N, self.num_heads * 2 * self.d_head)

        # Final linear projection
        out = self.W_o(O_concat)  # (batch, N, d_model)

        return out


class DiffTransformerLayer(nn.Module):
    """
    Single Layer of the DiffTransformer Architecture.
    Consists of Multi-Head Differential Attention followed by a SwiGLU Feed-Forward Network.
    """

    def __init__(self, d_model, num_heads, lambda_init):
        """
        Args:
            d_model (int): Dimension of the model.
            num_heads (int): Number of attention heads.
            lambda_init (float): Initial value for lambda in Differential Attention.
        """
        super().__init__()
        self.norm1 = RMSNorm(d_model, mode='sequence')
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init)
        self.norm2 = RMSNorm(d_model, mode='sequence')
        self.ff = SwiGLU(d_model, mode='sequence')

    def forward(self, x):
        """
        Forward pass for a single transformer layer.

        Args:
            x (Tensor): Input tensor of shape (batch, sequence_length, d_model).

        Returns:
            Tensor: Output tensor after processing through the layer.
        """
        # Apply Multi-Head Differential Attention with residual connection
        y = self.attn(self.norm1(x)) + x
        # Apply SwiGLU Feed-Forward Network with residual connection
        z = self.ff(self.norm2(y)) + y
        return z

