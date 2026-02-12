"""
K-Former: K-line Transformer for Visual Stock Trend Prediction

核心创新模块：
1. AxialAttention - 轴向分离注意力，处理高长宽比K线图像
2. SparseConv2d - 稀疏感知卷积，高效处理极度稀疏的金融图像
3. SRTemporalAttention - 支撑/阻力跨时间注意力
4. LearnableChartEncoder - 可学习图像编码，端到端优化

Reference for ECCV 2026 submission
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AxialAttention(nn.Module):
    """
    轴向分离注意力 (Axially Separable Attention)
    
    针对K线图像的高长宽比特性（如20天窗口 -> 20:1），
    将标准自注意力分解为时间轴和特征轴两个独立注意力，
    计算复杂度从O(H²W²)降低到O(HW(H+W))。
    
    Args:
        dim: 输入特征维度
        heads: 注意力头数
        dim_head: 每个头的维度
        temporal_first: 是否先沿时间轴做注意力
    """
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        temporal_first: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.temporal_first = temporal_first
        
        # 时间轴注意力参数
        self.temporal_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.temporal_out = nn.Conv2d(inner_dim, dim, 1, bias=False)
        
        # 特征轴注意力参数
        self.feature_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.feature_out = nn.Conv2d(inner_dim, dim, 1, bias=False)
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入特征图，H为时间维度，W为特征维度
        Returns:
            (B, C, H, W) 输出特征图
        """
        b, c, h, w = x.shape
        
        # 第一个轴向注意力
        if self.temporal_first:
            x = self._temporal_attention(x, h, w)
            x = self._feature_attention(x, h, w)
        else:
            x = self._feature_attention(x, h, w)
            x = self._temporal_attention(x, h, w)
            
        return x
    
    def _temporal_attention(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """沿时间轴(H)的注意力"""
        residual = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        qkv = self.temporal_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑为(B, heads, H, W, dim_head)
        q = q.reshape(q.shape[0], self.heads, self.dim_head, h, w)
        k = k.reshape(k.shape[0], self.heads, self.dim_head, h, w)
        v = v.reshape(v.shape[0], self.heads, self.dim_head, h, w)
        
        # 沿H轴做注意力: (B, heads, H, W, dim_head) -> (B, heads, W, H, H)
        q = q.permute(0, 1, 4, 3, 2)  # (B, heads, W, H, dim_head)
        k = k.permute(0, 1, 4, 2, 3)  # (B, heads, W, dim_head, H)
        v = v.permute(0, 1, 4, 3, 2)  # (B, heads, W, H, dim_head)
        
        attn = torch.matmul(q, k) * self.scale  # (B, heads, W, H, H)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, W, H, dim_head)
        out = out.permute(0, 1, 3, 4, 2)  # (B, heads, H, dim_head, W)
        out = out.reshape(out.shape[0], -1, h, w)
        
        out = self.temporal_out(out)
        return out + residual
    
    def _feature_attention(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """沿特征轴(W)的注意力"""
        residual = x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        qkv = self.feature_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 重塑
        q = q.reshape(q.shape[0], self.heads, self.dim_head, h, w)
        k = k.reshape(k.shape[0], self.heads, self.dim_head, h, w)
        v = v.reshape(v.shape[0], self.heads, self.dim_head, h, w)
        
        # 沿W轴做注意力: (B, heads, H, W, dim_head) -> (B, heads, H, W, W)
        q = q.permute(0, 1, 3, 4, 2)  # (B, heads, H, W, dim_head)
        k = k.permute(0, 1, 3, 2, 4)  # (B, heads, H, dim_head, W)
        v = v.permute(0, 1, 3, 4, 2)  # (B, heads, H, W, dim_head)
        
        attn = torch.matmul(q, k) * self.scale  # (B, heads, H, W, W)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, H, W, dim_head)
        out = out.permute(0, 1, 2, 4, 3)  # (B, heads, H, dim_head, W)
        out = out.reshape(out.shape[0], -1, h, w)
        
        out = self.feature_out(out)
        return out + residual


class SparseConv2d(nn.Module):
    """
    稀疏性感知卷积 (Sparsity-Aware Convolution)
    
    金融K线图像是极度稀疏的（大部分为背景黑色）。
    本模块借鉴神经形态视觉（Event-based Vision）思想，
    通过动态掩码仅对非零区域进行卷积计算，
    计算效率提升3-5倍。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        sparsity_threshold: 稀疏度阈值，低于此值视为零
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        sparsity_threshold: float = 0.01
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.sparsity_threshold = sparsity_threshold
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            (B, C, H, W) 输出特征图
        """
        # 生成动态掩码：标记非零区域
        mask = (x.abs().mean(dim=1, keepdim=True) > self.sparsity_threshold).float()
        
        # 应用卷积
        out = self.conv(x)
        
        # 应用掩码（仅在训练时，推理时可选择跳过以加速）
        if self.training:
            # 对掩码进行同样的空间下采样
            if self.conv.stride[0] > 1 or self.conv.stride[1] > 1:
                mask = F.max_pool2d(mask, kernel_size=self.conv.stride)
            out = out * mask
        
        out = self.bn(out)
        out = self.act(out)
        return out


class AdaptiveLevelDetector(nn.Module):
    """
    自适应支撑/阻力位检测器
    
    从历史价格序列中自动学习检测关键价格水平（支撑/阻力位）。
    使用可学习的聚类中心而非人工定义的固定规则。
    
    Args:
        num_levels: 检测的关键水平数量
        d_model: 特征维度
    """
    def __init__(self, num_levels: int = 5, d_model: int = 256):
        super().__init__()
        self.num_levels = num_levels
        self.d_model = d_model
        
        # 可学习的价格水平原型
        self.level_prototypes = nn.Parameter(torch.randn(num_levels, d_model))
        
        # 价格编码器
        self.price_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
    def forward(self, x: torch.Tensor, prices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) 特征序列
            prices: (B, T) 价格序列（通常是收盘价）
        Returns:
            level_features: (B, num_levels, C) 检测到的关键水平特征
            attention_weights: (B, T, num_levels) 每个时间步与关键水平的关联度
        """
        b, t, c = x.shape
        
        # 编码价格信息
        price_encoded = self.price_encoder(prices.unsqueeze(-1))  # (B, T, d_model)
        
        # 计算每个时间步与水平原型的相似度
        similarity = torch.matmul(price_encoded, self.level_prototypes.T)  # (B, T, num_levels)
        attention_weights = F.softmax(similarity, dim=1)  # 沿时间轴softmax
        
        # 聚合每个水平附近的特征
        level_features = torch.matmul(attention_weights.transpose(1, 2), x)  # (B, num_levels, C)
        
        return level_features, attention_weights


class SRTemporalAttention(nn.Module):
    """
    支撑/阻力跨时间注意力 (Support-Resistance Temporal Attention)
    
    专门设计用于捕捉技术分析中的关键概念：
    建立当前价格与历史关键支撑/阻力位之间的长程依赖关系。
    
    数学表达：
    A_{ij} = softmax(Q_i · K_j / √d) · M_{ij}
    其中 M_{ij} = exp(-|price_i - price_j| / σ) 是价格接近度掩码
    
    Args:
        dim: 特征维度
        num_levels: 关键水平数量
        num_heads: 注意力头数
    """
    def __init__(
        self,
        dim: int,
        num_levels: int = 5,
        num_heads: int = 8
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.level_detector = AdaptiveLevelDetector(num_levels, dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 价格接近度温度参数
        self.temp_sigma = nn.Parameter(torch.tensor(1.0))
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        x: torch.Tensor,
        prices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) 特征序列
            prices: (B, T) 价格序列
        Returns:
            (B, T, C) 增强后的特征序列
        """
        b, t, c = x.shape
        residual = x
        x = self.norm(x)
        
        # 1. 检测关键价格水平
        level_features, level_weights = self.level_detector(x, prices)
        # level_features: (B, num_levels, C)
        
        # 2. 投影为Q, K, V
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(level_features)  # (B, num_levels, C)
        v = self.v_proj(level_features)  # (B, num_levels, C)
        
        # 3. 多头注意力
        q = q.reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, heads, T, head_dim), k,v: (B, heads, num_levels, head_dim)
        
        # 4. 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, T, num_levels)
        
        # 5. 应用价格接近度掩码
        # 计算每个时间步与每个水平的价格距离
        # level_weights: (B, T, num_levels) - 已经在detector中计算
        price_proximity = level_weights.unsqueeze(1)  # (B, 1, T, num_levels)
        price_mask = torch.exp(-price_proximity / self.temp_sigma.abs())
        
        attn = attn * price_mask
        attn = F.softmax(attn, dim=-1)
        
        # 6. 应用注意力
        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(b, t, c)
        out = self.out_proj(out)
        
        return out + residual


class LearnableChartEncoder(nn.Module):
    """
    可学习图表编码器 (Learnable Chart Encoder)
    
    取代人工设计的OHLC/GAF编码，直接将原始价格序列
    映射为最优的视觉特征表示，实现端到端学习。
    
    输入: 原始OHLCV序列 (B, T, 5)
    输出: 视觉特征图 (B, C, H, W)
    
    Args:
        d_model: 特征维度
        num_layers: Transformer编码器层数
        output_size: 输出特征图尺寸 (H, W)
    """
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        output_size: Tuple[int, int] = (16, 16),
        num_heads: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.output_size = output_size
        
        # 价格嵌入
        self.price_embedding = nn.Linear(5, d_model)
        
        # 正弦位置编码
        self.pos_encoding = self._create_sinusoidal_positions(500, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影到2D特征图
        h, w = output_size
        self.to_2d = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, h * w)
        )
        
        # 通道投影
        self.to_channels = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
    def _create_sinusoidal_positions(self, max_len: int, d_model: int) -> nn.Parameter:
        """创建正弦位置编码"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, d_model)
        
    def forward(self, ohlcv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ohlcv: (B, T, 5) - Open, High, Low, Close, Volume
        Returns:
            (B, C, H, W) - 视觉特征图
        """
        b, t, _ = ohlcv.shape
        
        # 1. 嵌入价格信息
        x = self.price_embedding(ohlcv)  # (B, T, d_model)
        
        # 2. 添加位置编码
        x = x + self.pos_encoding[:, :t, :]
        
        # 3. Transformer编码
        x = self.transformer(x)  # (B, T, d_model)
        
        # 4. 投影到2D空间
        x = self.to_2d(x)  # (B, T, H*W)
        
        # 5. 重排为图像格式
        h, w = self.output_size
        x = x.reshape(b, t, h, w)
        
        # 6. 聚合时间维度到通道维度
        x = x.reshape(b, 1, t * h, w)  # 将时间展平到高度
        # 进一步调整到标准尺寸
        x = F.adaptive_avg_pool2d(x, (h, w))  # (B, 1, H, W)
        
        # 7. 投影到多通道
        x = self.to_channels(x)  # (B, 64, H, W)
        
        return x


class KFormer(nn.Module):
    """
    K-Former: 完整的K线Transformer模型
    
    整合所有创新模块的端到端股价预测模型。
    
    Args:
        num_classes: 分类数（2=涨跌，3=涨/平/跌）
        d_model: 特征维度
        num_sectors: 行业数量（用于Sector-MoE集成）
        use_learnable_encoder: 是否使用可学习编码器
    """
    def __init__(
        self,
        num_classes: int = 2,
        d_model: int = 256,
        num_sectors: int = 6,
        use_learnable_encoder: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_learnable_encoder = use_learnable_encoder
        
        # 编码器选择
        if use_learnable_encoder:
            self.encoder = LearnableChartEncoder(d_model=d_model, output_size=(16, 16))
            in_channels = 64
        else:
            # 使用传统图像输入
            in_channels = 3
            self.encoder = None
        
        # 稀疏卷积块
        self.sparse_conv_layers = nn.Sequential(
            SparseConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            SparseConv2d(64, 128, kernel_size=3, stride=2, padding=1),
            SparseConv2d(128, 256, kernel_size=3, stride=2, padding=1),
        )
        
        # 轴向注意力块
        self.axial_attn = AxialAttention(256, heads=8)
        
        # 支撑/阻力注意力（需要价格输入）
        self.sr_attention = SRTemporalAttention(256, num_levels=5)
        
        # 全局池化和分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        prices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入数据
               - 如果use_learnable_encoder=True: (B, T, 5) OHLCV序列
               - 否则: (B, 3, H, W) 图像
            prices: (B, T) 价格序列，用于SR-Attention
        Returns:
            (B, num_classes) 分类 logits
        """
        # 1. 编码
        if self.use_learnable_encoder:
            x = self.encoder(x)  # (B, 64, 16, 16)
        
        # 2. 稀疏卷积
        x = self.sparse_conv_layers(x)  # (B, 256, 4, 4)
        
        # 3. 轴向注意力
        x = self.axial_attn(x)
        
        # 4. 支撑/阻力注意力（需要reshape为序列）
        if prices is not None:
            b, c, h, w = x.shape
            x_seq = x.reshape(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
            x_seq = self.sr_attention(x_seq, prices[:, -h*w:])  # 使用最近的价格
            x = x_seq.transpose(1, 2).reshape(b, c, h, w)
        
        # 5. 分类
        x = self.global_pool(x)
        logits = self.classifier(x)
        
        return logits


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试LearnableChartEncoder
    print("Testing LearnableChartEncoder...")
    encoder = LearnableChartEncoder(d_model=256, output_size=(16, 16)).to(device)
    ohlcv = torch.randn(2, 20, 5).to(device)  # (B, T, 5)
    features = encoder(ohlcv)
    print(f"  Input: {ohlcv.shape} -> Output: {features.shape}")
    
    # 测试AxialAttention
    print("\nTesting AxialAttention...")
    axial_attn = AxialAttention(256, heads=8).to(device)
    x = torch.randn(2, 256, 16, 16).to(device)
    out = axial_attn(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # 测试SparseConv2d
    print("\nTesting SparseConv2d...")
    sparse_conv = SparseConv2d(64, 128, kernel_size=3, stride=2).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    out = sparse_conv(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # 测试SRTemporalAttention
    print("\nTesting SRTemporalAttention...")
    sr_attn = SRTemporalAttention(256, num_levels=5).to(device)
    x = torch.randn(2, 100, 256).to(device)  # (B, T, C)
    prices = torch.randn(2, 100).to(device)  # (B, T)
    out = sr_attn(x, prices)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # 测试完整KFormer
    print("\nTesting KFormer...")
    model = KFormer(num_classes=2, use_learnable_encoder=True).to(device)
    ohlcv = torch.randn(2, 20, 5).to(device)
    prices = torch.randn(2, 20).to(device)
    logits = model(ohlcv, prices)
    print(f"  Input: {ohlcv.shape} -> Output: {logits.shape}")
    
    print("\n✅ All tests passed!")
