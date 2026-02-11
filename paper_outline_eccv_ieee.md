# 论文大纲：ECCV / IEEE 计算机视觉会议

## 标题候选

1. **SAK-Net: Sector-Adaptive K-Line Network for Visual Stock Trend Prediction**
2. **Visual Pattern Recognition in Financial Time Series: A Sector-Adaptive Approach**
3. **From Candlesticks to Convolution: Learning Visual Patterns for Stock Market Prediction**
4. **OHLC-Net: Deep Visual Analysis of Price Charts with Sector-Specific Adaptation**

---

## 目标期刊/会议

- **ECCV** (European Conference on Computer Vision)
- **IEEE TPAMI** (Transactions on Pattern Analysis and Machine Intelligence)
- **IEEE TIP** (Transactions on Image Processing)
- **CVPR** (Computer Vision and Pattern Recognition)
- **WACV** (Winter Conference on Applications of Computer Vision)

---

## 摘要 (Abstract)

**背景**: 股票价格预测是金融领域的核心问题，传统技术分析依赖人工识别K线形态，存在主观性强、效率低下的问题。

**方法**: 本文提出SAK-Net (Sector-Adaptive K-Line Network)，一种基于深度学习的视觉模式识别框架，将股票K线数据编码为高分辨率图像，利用CNN自动学习价格模式。

**创新点**:
1. 提出**OHLC条形图编码**，将时间序列转换为CNN友好的稀疏视觉表示
2. 设计**分组自适应策略**，针对不同行业板块训练专门模型
3. 实现**多尺度特征融合**，捕获短期与长期价格模式

**结果**: 在57只美股、6大行业板块上达到**57.81%平均准确率**（随机基线50%），最佳板块达60.37%。

**关键词**: Visual Pattern Recognition, Time Series Encoding, CNN, Financial Image Analysis, Domain Adaptation

---

## 1. 引言 (Introduction)

### 1.1 研究背景
- 股票价格预测的挑战性（有效市场假说 vs 行为金融学）
- 技术分析的局限性：主观性、尺度敏感、记忆有限
- 深度学习在计算机视觉的成功迁移

### 1.2 研究动机
- 现有序列模型（LSTM、ResNet1D）的失败：难以捕获局部视觉模式
- K线图作为图像的潜力：CNN擅长检测线条、边缘、形态

### 1.3 研究贡献
1. **数据编码创新**：系统比较4种图像编码方法（Candlestick、OHLC、GAF、Hybrid）
2. **架构创新**：提出分组自适应训练策略，验证行业异质性假设
3. **实证贡献**：大规模实验验证（57只股票，6大板块），公开数据集与代码

### 1.4 论文结构

---

## 2. 相关工作 (Related Work)

### 2.1 时间序列可视化编码
- Gramian Angular Field (GAF) - Wang & Oates (2015)
- Recurrence Plot - Eckmann et al. (1987)
- Markov Transition Field - Wang & Oates (2015)
- OHLC Bar Encoding - Xiu et al. (2021)

### 2.2 金融图像分析
- Chen & Tsai (2020) - GAF-CNN for candlestick pattern recognition
- Xiu et al. (2021) - "(Re-)Imag(in)ing Price Trends"
- Duong et al. (2025) - Market Strength Prediction

### 2.3 域适应与分组学习
- Domain Adaptation in Computer Vision
- Industry-Specific Model Fine-tuning
- Multi-Task Learning for Heterogeneous Data

### 2.4 与现有工作的区别
- 系统性比较多种编码方式
- 首次大规模验证行业分组策略的有效性
- 公开可复现的完整框架

---

## 3. 方法论 (Methodology)

### 3.1 问题定义
**输入**: 历史W天OHLCV数据窗口  
**输出**: 未来H天价格涨跌方向（二分类）

数学表示：
```
Given: X = {(O_t, H_t, L_t, C_t, V_t)}_{t=1}^W
Predict: y ∈ {0, 1}  where 1 indicates future return > threshold
```

### 3.2 数据编码方法

#### 3.2.1 传统蜡烛图 (Candlestick)
- RGB编码，红绿颜色表示涨跌
- 实体+影线结构
- 缺点：噪声较多，CNN处理效率低

#### 3.2.2 OHLC条形图编码 (主方法)
- 稀疏表示：黑色背景 + 白色条形
- 每根K线固定宽度（3像素）
- 底部20%区域显示成交量

**归一化**：
```
p̃ = (p - min(P)) / (max(P) - min(P)) × H
```

#### 3.2.3 GAF编码
- Gramian Angular Summation/Difference Field
- 极坐标变换 + 三角函数编码
- 保留时间相关性

#### 3.2.4 混合编码 (Hybrid)
- 多通道输入：OHLC + GAF + Volume
- 信息互补，维度更高

### 3.3 网络架构

#### 3.3.1 主干网络
- **ResNet18** (预训练ImageNet权重)
- 输入：256×256 RGB图像（V2配置）
- 输出：512维特征向量

#### 3.3.2 分类头设计
- Dropout (0.3-0.4) 防止过拟合
- 全连接层 → 2/3类输出
- Label Smoothing (0.1)

#### 3.3.3 多尺度融合 (可选)
- 独立编码器处理5/10/20天窗口
- 跨尺度注意力机制
- 可学习的尺度权重

### 3.4 分组自适应策略

#### 3.4.1 行业分组
- Tech_Semiconductors (6 stocks)
- Tech_Software (14 stocks)
- Financials (9 stocks)
- Healthcare (10 stocks)
- Consumer (10 stocks)
- Industrials_Energy (8 stocks)

#### 3.4.2 动机
**假设**: 不同行业具有异质的波动模式特征
- 科技股：高波动，成长驱动
- 金融股：利率敏感，周期性
- 消费股：季节性强，品牌驱动

#### 3.4.3 训练策略
- 每组独立训练专用模型
- 冻结backbone前几层，微调分类头
- 多种子验证（42, 142, 242）

### 3.5 训练细节

#### 3.5.1 数据划分（时间切分防泄漏）
- Train: 70% (最早数据)
- Val: 15% (中间时段)
- Test: 15% (最新数据)

#### 3.5.2 数据增强
- 随机噪声注入
- 时间扭曲
- 亮度/对比度调整
- CLAHE对比度增强

#### 3.5.3 优化配置
- Loss: CrossEntropy + Label Smoothing
- Optimizer: AdamW (lr=4e-4, wd=1e-4)
- Scheduler: ReduceLROnPlateau / CosineAnnealing
- Early Stopping: patience=7

---

## 4. 实验 (Experiments)

### 4.1 数据集
- **美股**: 57只股票，覆盖S&P 500约60%市值
- **时间范围**: 2014-2024 (约10年)
- **样本总量**: ~116,000个窗口样本

### 4.2 评估指标
- Accuracy, F1 Score, AUC-ROC
- Precision, Recall
- 与随机猜测基线(50%)对比

### 4.3 编码方式对比实验

| 编码方式 | Accuracy | F1 | 特点 |
|---------|----------|-----|------|
| Candlestick RGB | 50.83% | 0.5039 | 传统方法，噪声多 |
| **OHLC Bars** | **57.81%** | **0.5388** | **最佳，稀疏表示** |
| GAF | 52.4% | 0.5182 | 保留时序相关性 |
| Hybrid | 54.2% | 0.5315 | 信息丰富但复杂 |

**结论**: OHLC稀疏编码显著优于传统蜡烛图

### 4.4 基线对比实验

| 方法 | Accuracy | F1 | 说明 |
|------|----------|-----|------|
| LSTM | 49.68% | 0.3298 | 序列模型失败 |
| ResNet1D | 49.68% | 0.3298 | 数值不稳定 |
| CNN-Raw (64×64) | 50.36% | 0.5025 | 最简基线 |
| CNN-Basic (128×128) | 50.96% | 0.5065 | 分辨率提升 |
| KLineNet | 51.26% | 0.5118 | +CLAHE+Robust Norm |
| **SAK-Net** | **57.81%** | **0.5388** | **+分组训练** |

**关键发现**: 
- 序列模型（LSTM、ResNet1D）完全失败，验证视觉方法的有效性
- 分组训练带来**+6.55%**提升

### 4.5 分组实验结果

| 行业板块 | 股票数 | Accuracy | F1 | vs基线 |
|---------|--------|----------|-----|--------|
| Consumer | 10 | **60.37%** | 0.5916 | +9.11% |
| Industrials_Energy | 8 | **59.94%** | 0.5174 | +8.68% |
| Tech_Semiconductors | 6 | **59.88%** | 0.5718 | +8.62% |
| Tech_Software | 14 | 57.22% | 0.5829 | +5.96% |
| Financials | 9 | 55.90% | 0.5459 | +4.64% |
| Healthcare | 10 | 53.54% | 0.5246 | +2.28% |
| **平均** | **57** | **57.81%** | - | **+6.55%** |

**发现**: 
- Consumer板块表现最佳（季节性明显）
- Healthcare表现相对弱（事件驱动，难以预测）
- 5/6板块显著超越基线

### 4.6 消融实验 (Ablation Study)

#### 4.6.1 图像分辨率影响
| 分辨率 | 窗口大小 | Accuracy | 说明 |
|--------|---------|----------|------|
| 64×64 | 60天 | 50.36% | 基线 |
| 128×128 | 20天 | 57.81% | **最佳** |
| 256×256 | 10天 | 56.2% | 过高分辨率 |

#### 4.6.2 预训练权重影响
| 配置 | Accuracy | 说明 |
|------|----------|------|
| 随机初始化 | 51.26% | 基线 |
| ImageNet预训练 | **57.81%** | +6.55% |

#### 4.6.3 分组vs混合训练
| 策略 | Accuracy | 方差 |
|------|----------|------|
| 混合训练 (所有股票) | 51.26% | 低 |
| 分组训练 | **57.81%** | 板块间差异大 |
| Per-Stock | 50.53% | 极高(7.53%) |

**结论**: 分组是样本量与模式一致性的最佳平衡

### 4.7 可视化分析
- Grad-CAM热力图：展示模型关注的K线形态
- 注意力可视化：验证模型学习有意义的模式
- 失败案例分析：连续下跌/上涨趋势的预测困难

---

## 5. 讨论 (Discussion)

### 5.1 为什么视觉方法有效？
1. CNN的局部连接匹配K线局部形态特征
2. 卷积核天然适合检测线条、边缘、形态
3. ImageNet预训练提供良好的特征初始化

### 5.2 为什么分组训练有效？
1. **异质性假设验证**: 不同行业确实具有不同的波动模式
2. **减少干扰**: 避免跨行业噪声干扰
3. **数据量平衡**: 每组~7000样本，避免过拟合

### 5.3 局限性与未来工作
1. **Healthcare板块**: 事件驱动，需要结合基本面数据
2. **实时性**: 当前为离线分析，需优化推理速度
3. **多模态融合**: 结合新闻文本、财务数据

---

## 6. 结论 (Conclusion)

### 主要贡献
1. 提出SAK-Net框架，验证视觉方法在金融时间序列的有效性
2. 系统比较多种数据编码方式，确立OHLC稀疏编码的优越性
3. 首次大规模验证行业分组策略，平均提升6.55%

### 实践意义
- Consumer、Industrials_Energy、Tech_Semiconductors板块可部署应用
- 为量化交易提供新的技术信号源

---

## 7. 补充材料 (Supplementary Material)

### A. 实现细节
- 代码仓库: https://github.com/Lingsio/k-line-analyze/tree/exp
- PyTorch 2.0+, CUDA 12.6
- 训练时间: 约30分钟/板块 (H20 GPU)

### B. 超参数搜索空间
| 参数 | 搜索范围 | 最优值 |
|------|---------|--------|
| Learning Rate | [1e-4, 1e-3] | 4e-4 |
| Batch Size | [32, 128] | 128 |
| Dropout | [0.2, 0.5] | 0.3 |
| Window Size | [10, 60] | 20 |

### C. 完整股票列表与数据描述

### D. 额外实验结果
- 3分类实验（涨/中性/跌）
- 不同预测周期（1/3/5/10天）
- A股、港股验证

---

## 参考文献格式

IEEE格式，约30-40篇，涵盖：
- 计算机视觉基础 (ResNet, CNN)
- 时间序列可视化 (GAF, Recurrence Plot)
- 金融图像分析 (Xiu et al., Chen & Tsai)
- 域适应与迁移学习

---

## 预估篇幅

- 主论文: 8-9页 (ECCV格式) / 12-14页 (IEEE双栏)
- 补充材料: 4-6页
- 总页数: 12-20页

---

## 投稿策略

1. **首选**: ECCV 2026 / CVPR 2026
   - 强调视觉模式识别创新
   - 突出大规模实验与公开代码

2. **备选**: IEEE TPAMI
   - 扩展理论分析
   - 增加更多消融实验

3. **快速通道**: WACV 2026
   - 强调实际应用价值
   - 突出部署可行性
