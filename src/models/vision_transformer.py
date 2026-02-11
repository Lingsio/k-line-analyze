"""
Vision Transformer (ViT) 用于K线图像分类
将K线图像分割成patches，通过Transformer学习视觉模式
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    将图像分割成patches并映射到embedding空间

    Args:
        img_size: 图像尺寸 (H, W)
        patch_size: patch大小
        in_channels: 输入通道数 (3 for RGB)
        embed_dim: embedding维度
    """
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 使用卷积实现patch提取和线性投影
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (batch, channels, H, W)
        x = self.projection(x)  # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim必须被num_heads整除"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_tokens, embed_dim = x.shape

        # 计算Q, K, V
        qkv = self.qkv(x)  # (batch, n_tokens, 3*embed_dim)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, n_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 加权求和
        x = (attn @ v).transpose(1, 2)  # (batch, n_tokens, num_heads, head_dim)
        x = x.flatten(2)  # (batch, n_tokens, embed_dim)

        # 输出投影
        x = self.projection(x)
        x = self.proj_dropout(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 多头注意力 + 残差连接
        x = x + self.attn(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for K-line Image Classification

    将K线图像分割成patches，使用Transformer学习全局视觉模式

    Args:
        img_size: 图像尺寸 (默认128x128)
        patch_size: patch大小 (默认16, 产生64个patches)
        in_channels: 输入通道数 (3 for RGB)
        num_classes: 分类类别数
        embed_dim: embedding维度
        depth: Transformer层数
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层倍数
        dropout: Dropout比例
    """
    def __init__(
        self,
        img_size=128,
        patch_size=16,
        in_channels=3,
        num_classes=2,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # CLS token (类似BERT的[CLS]，用于分类)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码 (可学习)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 初始化线性层
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, H, W) - K线图像

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Transformer编码
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 取CLS token进行分类
        cls_output = x[:, 0]  # (batch, embed_dim)
        logits = self.head(cls_output)

        return logits


class VisionTransformerSmall(VisionTransformer):
    """ViT-Small变体 (适合128x128图像的轻量级版本)"""
    def __init__(self, img_size=128, patch_size=16, in_channels=3, num_classes=2):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=384,
            depth=6,
            num_heads=6,
            mlp_ratio=4.0,
            dropout=0.1
        )


class VisionTransformerTiny(VisionTransformer):
    """ViT-Tiny变体 (更轻量，适合快速实验)"""
    def __init__(self, img_size=128, patch_size=16, in_channels=3, num_classes=2):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=192,
            depth=4,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1
        )


class VisionTransformerBase(VisionTransformer):
    """ViT-Base变体 (标准配置)"""
    def __init__(self, img_size=128, patch_size=16, in_channels=3, num_classes=2):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1
        )


# 测试代码
if __name__ == '__main__':
    # 测试ViT-Small
    model = VisionTransformerSmall(img_size=128, patch_size=16, num_classes=2)
    x = torch.randn(4, 3, 128, 128)  # batch=4, RGB图像
    out = model(x)
    print(f"输入: {x.shape}")
    print(f"输出: {out.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
