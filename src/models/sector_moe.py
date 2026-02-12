"""
Sector-MoE: Mixture of Experts for Sector-Adaptive Stock Prediction

混合专家模型实现真正的领域自适应，相比简单分组训练：
1. 共享的特征提取层学习市场通用规律
2. 门控网络根据输入特征自动路由到合适的行业专家
3. 每个专家学习特定行业的特有模式
4. 支持软路由和知识迁移

Reference for ECCV 2026 submission
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import math


class SectorExpert(nn.Module):
    """
    单个行业专家模块
    
    每个专家是一个轻量级的预测网络，
    专注于学习特定行业的K线形态特征。
    
    Args:
        d_model: 输入特征维度
        hidden_dim: 专家隐藏层维度
        num_classes: 输出类别数
        dropout: Dropout率
    """
    def __init__(
        self,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, d_model) 共享特征
        Returns:
            (B, num_classes) 专家预测
        """
        return self.network(x)


class GatingNetwork(nn.Module):
    """
    门控网络：决定样本应由哪些专家处理
    
    实现Top-K稀疏门控，确保每次只激活少量专家，
    提高计算效率并促进专家特化。
    
    Args:
        d_model: 输入特征维度
        num_experts: 专家数量
        top_k: 每次激活的专家数
        noise_std: 训练时添加的噪声标准差（辅助探索）
    """
    def __init__(
        self,
        d_model: int = 256,
        num_experts: int = 6,
        top_k: int = 2,
        noise_std: float = 1.0
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        
        # 门控线性层
        self.gate_linear = nn.Linear(d_model, num_experts, bias=False)
        
        # 噪声线性层（仅训练时使用）
        self.noise_linear = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, d_model) 共享特征
        Returns:
            gates: (B, num_experts) 门控权重（稀疏）
            indices: (B, top_k) 激活的专家索引
        """
        # 计算logits
        logits = self.gate_linear(x)
        
        # 训练时添加噪声以促进探索
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x))
            logits = logits + noise * self.noise_std
        
        # Top-K门控
        top_gates, top_indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        
        # 归一化Top-K门控
        top_gates = top_gates / (top_gates.sum(dim=-1, keepdim=True) + 1e-10)
        
        # 创建稀疏门控向量
        gates = torch.zeros_like(logits)
        gates.scatter_(1, top_indices, top_gates)
        
        return gates, top_indices


class LoadBalancingLoss(nn.Module):
    """
    负载均衡损失
    
    鼓励所有专家被均匀使用，避免某些专家过载而其他专家闲置。
    
    Reference: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)
    
    Loss = alpha * num_experts * sum(f_i * P_i)
    其中f_i是第i个专家的门控频率，P_i是平均路由概率
    """
    def __init__(self, num_experts: int, alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        
    def forward(
        self,
        gates: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            gates: (B, num_experts) 门控权重
            indices: (B, top_k) 激活的专家索引
        Returns:
            loss: 标量损失
        """
        # 计算每个样本的门控计数（每个专家被多少样本选中）
        expert_mask = torch.zeros_like(gates).scatter_(1, indices, 1.0)
        
        # 频率f_i：每个专家处理的样本比例
        f = expert_mask.mean(dim=0)  # (num_experts,)
        
        # 平均路由概率P_i
        P = gates.mean(dim=0)  # (num_experts,)
        
        # 负载均衡损失
        loss = self.alpha * self.num_experts * (f * P).sum()
        
        return loss


class SectorMoE(nn.Module):
    """
    Sector Mixture of Experts (Sector-MoE)
    
    完整的混合专家模型，用于行业自适应股价预测。
    
    相比简单分组训练的优势：
    1. 参数共享：共享编码器减少总参数量
    2. 软路由：样本可以部分属于多个行业
    3. 知识迁移：通过共享层实现跨行业知识传递
    4. 推理效率：单次前向传播，无需加载多个模型
    
    Args:
        num_experts: 专家数量（通常=行业数量）
        d_model: 特征维度
        hidden_dim: 专家隐藏层维度
        num_classes: 输出类别数
        top_k: 每次激活的专家数
        use_shared_encoder: 是否使用共享编码器
        encoder_type: 编码器类型 ('kformer', 'resnet', 'simple')
    """
    def __init__(
        self,
        num_experts: int = 6,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        top_k: int = 2,
        dropout: float = 0.3,
        use_shared_encoder: bool = True,
        encoder_type: str = 'simple',
        input_channels: int = 3
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.num_classes = num_classes
        self.top_k = top_k
        self.use_shared_encoder = use_shared_encoder
        
        # 共享编码器
        if use_shared_encoder:
            self.shared_encoder = self._build_encoder(
                encoder_type, input_channels, d_model
            )
        else:
            self.shared_encoder = None
            d_model = input_channels  # 直接使用输入特征
        
        # 门控网络
        self.gating_network = GatingNetwork(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # 专家网络
        self.experts = nn.ModuleList([
            SectorExpert(d_model, hidden_dim, num_classes, dropout)
            for _ in range(num_experts)
        ])
        
        # 负载均衡损失
        self.load_balancing_loss = LoadBalancingLoss(num_experts)
        
        # 初始化
        self._init_weights()
        
    def _build_encoder(
        self,
        encoder_type: str,
        input_channels: int,
        d_model: int
    ) -> nn.Module:
        """构建共享编码器"""
        if encoder_type == 'simple':
            # 简单CNN编码器
            return nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, d_model)
            )
        elif encoder_type == 'resnet':
            # 使用ResNet作为编码器
            try:
                import torchvision.models as models
                resnet = models.resnet18(pretrained=True)
                # 修改第一层以接受input_channels
                if input_channels != 3:
                    resnet.conv1 = nn.Conv2d(
                        input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                # 移除最后的分类层
                modules = list(resnet.children())[:-1]
                encoder = nn.Sequential(*modules)
                # 添加投影层
                return nn.Sequential(
                    encoder,
                    nn.Flatten(),
                    nn.Linear(512, d_model)
                )
            except ImportError:
                print("torchvision not available, falling back to simple encoder")
                return self._build_encoder('simple', input_channels, d_model)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_gates: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: 输入数据
               - 如果使用共享编码器: (B, C, H, W) 图像
               - 否则: (B, d_model) 特征
            return_gates: 是否返回门控权重
        Returns:
            output: (B, num_classes) 预测输出
            gates: (B, num_experts) 门控权重（如果return_gates=True）
            aux_loss: 负载均衡损失
        """
        # 1. 共享编码
        if self.shared_encoder is not None:
            shared_features = self.shared_encoder(x)  # (B, d_model)
        else:
            shared_features = x
        
        # 2. 门控决策
        gates, indices = self.gating_network(shared_features)
        # gates: (B, num_experts), indices: (B, top_k)
        
        # 3. 计算负载均衡损失
        aux_loss = self.load_balancing_loss(gates, indices)
        
        # 4. 专家预测
        expert_outputs = torch.stack([
            expert(shared_features) for expert in self.experts
        ], dim=1)  # (B, num_experts, num_classes)
        
        # 5. 加权融合
        # 扩展gates维度以便广播
        gates_expanded = gates.unsqueeze(-1)  # (B, num_experts, 1)
        output = (gates_expanded * expert_outputs).sum(dim=1)  # (B, num_classes)
        
        if return_gates:
            return output, gates, aux_loss
        return output, aux_loss
    
    def get_expert_usage(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        统计每个专家的使用频率
        
        Args:
            dataloader: 数据加载器
        Returns:
            各专家使用频率字典
        """
        self.eval()
        usage_counts = torch.zeros(self.num_experts)
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(next(self.parameters()).device)
                
                _, gates, _ = self.forward(x, return_gates=True)
                usage_counts += (gates > 0).sum(dim=0).cpu()
                total_samples += x.shape[0]
        
        usage_freq = usage_counts / total_samples
        return {f'Expert_{i}': freq.item() for i, freq in enumerate(usage_freq)}


class SectorMoEWithAuxiliary(nn.Module):
    """
    带辅助任务的Sector-MoE
    
    除了主要的涨跌预测任务外，还同时优化：
    1. 行业识别任务（辅助门控网络学习）
    2. 波动率预测任务（学习风险特征）
    
    多任务学习进一步提升表示质量。
    """
    def __init__(
        self,
        num_experts: int = 6,
        d_model: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 2,
        top_k: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # 主MoE模型
        self.moe = SectorMoE(
            num_experts=num_experts,
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            top_k=top_k,
            dropout=dropout
        )
        
        # 辅助任务：行业识别
        self.sector_classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # 辅助任务：波动率预测（回归）
        self.volatility_predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入图像
            return_all: 是否返回所有任务输出
        Returns:
            根据return_all返回不同输出
        """
        # 获取共享特征
        shared_features = self.moe.shared_encoder(x)
        
        # 主任务：涨跌预测
        main_output, aux_loss = self.moe.forward(
            shared_features,
            return_gates=False
        )
        
        if not return_all:
            return main_output, aux_loss
        
        # 辅助任务
        sector_logits = self.sector_classifier(shared_features)
        volatility = self.volatility_predictor(shared_features)
        
        return {
            'main': main_output,
            'sector': sector_logits,
            'volatility': volatility,
            'aux_loss': aux_loss,
            'features': shared_features
        }


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")
    
    # 测试SectorExpert
    print("\nTesting SectorExpert...")
    expert = SectorExpert(d_model=256, hidden_dim=128, num_classes=2).to(device)
    x = torch.randn(4, 256).to(device)
    out = expert(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # 测试GatingNetwork
    print("\nTesting GatingNetwork...")
    gating = GatingNetwork(d_model=256, num_experts=6, top_k=2).to(device)
    gates, indices = gating(x)
    print(f"  Input: {x.shape}")
    print(f"  Gates: {gates.shape}, sum per sample: {gates.sum(dim=1)[:3]}")
    print(f"  Indices: {indices.shape}, top experts: {indices[:3]}")
    
    # 测试LoadBalancingLoss
    print("\nTesting LoadBalancingLoss...")
    lb_loss = LoadBalancingLoss(num_experts=6)
    loss = lb_loss(gates, indices)
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试SectorMoE
    print("\nTesting SectorMoE...")
    moe = SectorMoE(
        num_experts=6,
        d_model=256,
        hidden_dim=128,
        num_classes=2,
        top_k=2,
        encoder_type='simple',
        input_channels=3
    ).to(device)
    
    x_img = torch.randn(4, 3, 128, 128).to(device)
    output, aux_loss = moe(x_img)
    print(f"  Input: {x_img.shape} -> Output: {output.shape}")
    print(f"  Auxiliary loss: {aux_loss.item():.4f}")
    
    # 测试带门控返回的前向传播
    output, gates, aux_loss = moe(x_img, return_gates=True)
    print(f"  Gates shape: {gates.shape}")
    print(f"  Gates sum per sample: {gates.sum(dim=1)[:3]}")
    
    # 测试SectorMoEWithAuxiliary
    print("\nTesting SectorMoEWithAuxiliary...")
    moe_aux = SectorMoEWithAuxiliary(
        num_experts=6,
        d_model=256,
        num_classes=2
    ).to(device)
    
    results = moe_aux(x_img, return_all=True)
    print(f"  Main output: {results['main'].shape}")
    print(f"  Sector logits: {results['sector'].shape}")
    print(f"  Volatility: {results['volatility'].shape}")
    print(f"  Features: {results['features'].shape}")
    
    print("\nAll tests passed!")
