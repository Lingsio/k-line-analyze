"""
Transformer模型用于K线序列分类
结合时序位置编码和注意力机制，捕捉K线序列的长期依赖关系
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码，为序列添加时间信息"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer模型用于K线序列分类

    Args:
        input_size: 输入特征维度 (默认5: OHLCV)
        d_model: Transformer隐藏层维度
        nhead: 多头注意力头数
        num_layers: Transformer层数
        dim_feedforward: 前馈网络维度
        num_classes: 分类类别数
        dropout: Dropout比例
        max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        input_size=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=2,
        dropout=0.1,
        max_seq_len=100
    ):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.input_size = input_size

        # 输入投影：将input_size维特征映射到d_model维
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # PyTorch 1.9+
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        前向传播

        Args:
            x: (batch, seq_len, input_size) - K线序列数据

        Returns:
            logits: (batch, num_classes) - 分类logits
        """
        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # 添加位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)

        # 全局平均池化
        x = x.mean(dim=1)  # (batch, d_model)

        # 分类
        logits = self.classifier(x)  # (batch, num_classes)

        return logits


class TransformerModelWithCLS(nn.Module):
    """
    带CLS token的Transformer模型（类似BERT）

    使用特殊的[CLS] token来聚合整个序列的信息
    """

    def __init__(
        self,
        input_size=5,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=2,
        dropout=0.1,
        max_seq_len=100
    ):
        super(TransformerModelWithCLS, self).__init__()

        self.d_model = d_model

        # CLS token（可学习的特殊token）
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码（max_seq_len + 1 for CLS token）
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len + 1, dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            logits: (batch, num_classes)
        """
        batch_size = x.size(0)

        # 输入投影
        x = self.input_projection(x)  # (batch, seq_len, d_model)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)

        # 取CLS token的输出
        cls_output = x[:, 0, :]  # (batch, d_model)

        # 分类
        logits = self.classifier(cls_output)

        return logits
