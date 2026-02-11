# Image Encoding Methods实施总结

## 已实施的改进

### 1. OHLC 条形图表示 (Xiu et al. 2021) ✅

**文件**: `src/data/image_generator.py`

**新增方法**:
- `draw_ohlc_bars()` - 绘制 OHLC 条形图
- 每个交易日占用 3 像素宽度（开盘、高低、收盘）
- 黑色背景，白色条形（稀疏表示，有利于 CNN）
- 所有股票价格归一化到统一尺度

**关键特性**:
- 与 Xiu et al. 论文一致的设计
- 归一化使不同股票具有可比性
- 保留成交量信息（底部 20% 区域）

### 2. GAF 编码 (Chen & Tsai 2020) ✅

**文件**: `src/data/image_generator.py`

**新增方法**:
- `create_gaf_ohlc()` - 将 OHLCV 数据编码为 GAF 图像
- 支持 GASF (Gramian Angular Summation Field) 和 GADF
- 多通道输出（Open, High, Low, Close, Volume）

**数学原理**:
```
1. 归一化到 [-1, 1]: x̃ = (2x - max - min) / (max - min)
2. 极坐标编码: φ = arccos(x̃)
3. GASF: cos(φ_i + φ_j)
4. GADF: sin(φ_i - φ_j)
```

**关键特性**:
- 保留时间依赖性
- 捕获相对相关性
- 对角线包含原始值信息

### 3. 短窗口 + 高分辨率 ✅

**配置**:
- 窗口大小: 10 天 (vs 原来的 20 天)
- 图像尺寸: 256×256 (vs 原来的 128×128)
- 每根 K 线宽度: ~25 像素 (vs 原来的 ~6 像素)

**优势**:
- 更清晰的 K 线形态
- 能分辨十字星、锤子线等细节
- 更符合论文中的实验设置

## 新增实验配置

### 实验脚本

**文件**: `scripts/run_v2_experiments.py`

**新增实验**:
1. `OHLC-10d-256` - 纯 OHLC 条形图
2. `GAF-10d-256` - 纯 GAF 编码
3. `OHLC-GAF-10d-256` - OHLC + GAF 混合
4. `Hybrid-10d-256` - 融合表示
5. `Candle-20d-128` - 基线对比

### 使用方法

```bash
# 运行所有 V2 实验
python scripts/run_v2_experiments.py --experiment all --num-runs 3

# 运行单个实验
python scripts/run_v2_experiments.py --experiment OHLC-10d-256

# 可视化比较
python scripts/visualize_representations.py --symbol AAPL
```

## 数据集更新

**文件**: `src/data/dataset.py`

**新增参数**:
- `chart_type`: 'candle', 'ohlc', 'gaf', 'hybrid'
- `use_gaf`: 是否启用 GAF 增强
- `gaf_method`: 'gasf' 或 'gadf'

**示例**:
```python
dataset = StockDataset(
    data_dir='data/raw/us',
    window_size=10,          # 10天窗口
    img_size=(256, 256),     # 高分辨率
    chart_type='ohlc',       # OHLC条形图
    use_gaf=True,            # 启用GAF
    gaf_method='gasf',
)
```

## 与原版的对比

| 特性 | 原版 | V2 改进 |
|------|------|---------|
| 图表类型 | 蜡烛图 | 蜡烛图 / OHLC / GAF / 混合 |
| 窗口大小 | 20天 | 10天 (可配置) |
| 图像尺寸 | 128×128 | 256×256 (可配置) |
| 每根K线宽度 | ~6像素 | ~25像素 |
| 归一化 | MinMax/Robust | 增加基于百分位的归一化 |
| 数据增强 | 基础增强 | 增加时间扭曲、数据缩放 |
| 多尺度 | 单尺度 | 支持多通道融合 |

## 预期效果

基于参考论文的结果：

| 方法 | 预期准确率 | 参考来源 |
|------|-----------|---------|
| 原版 CNN | 51.8% | 当前基线 |
| OHLC 10d/256 | 54-56% | Xiu et al. (2021) - 53%+ |
| GAF 编码 | 55-58% | Chen & Tsai (2020) - 90.7% (模式识别) |
| 混合方法 | 56-60% | 综合两者优势 |

## 实现细节

### OHLC 条形图绘制
```python
def draw_ohlc_bars(self, open_p, high_p, low_p, close_p, volume, ...):
    # 每个交易日 = 3像素
    # 像素0: 开盘横线
    # 像素1: 高低竖线
    # 像素2: 收盘横线
    # 价格归一化到图像高度
```

### GAF 编码
```python
def create_gaf_ohlc(self, open_p, high_p, low_p, close_p, volume, ...):
    # 1. 归一化到 [-1, 1]
    # 2. 极坐标转换: φ = arccos(x̃)
    # 3. 计算 GASF: cos(φ_i + φ_j)
    # 4. 输出 5 通道 (O, H, L, C, V)
```

## 训练建议

1. **批次大小**: 由于 256×256 图像更大，建议使用 batch_size=64 (vs 原来的 128)
2. **学习率**: 3e-4 (稍低于原来的 4e-4)
3. **训练轮数**: 30 轮 (更多的数据增强需要更多轮数收敛)
4. **早停耐心**: 7 轮 (防止过拟合)

## 文件变更列表

### 修改的文件
1. `src/data/image_generator.py` - 添加 OHLC, GAF, Hybrid 方法
2. `src/data/dataset.py` - 添加 chart_type, use_gaf 参数
3. `src/models/cnn_model.py` - 优化 ResNet18 以支持更大输入
4. `scripts/run_baseline_experiments.py` - 添加 V2 实验配置

### 新增的文件
1. `scripts/run_v2_experiments.py` - V2 实验专用脚本
2. `scripts/visualize_representations.py` - 可视化比较工具
3. `V2_IMPROVEMENTS.md` - 本文档

## 下一步建议

1. **运行 V2 实验**:
   ```bash
   python scripts/run_v2_experiments.py --experiment all
   ```

2. **对比结果**:
   - 比较 OHLC vs GAF vs 混合方法
   - 检查是否缓解过拟合问题
   - 验证跨股票泛化能力

3. **进一步优化**:
   - 如果 OHLC 效果好，尝试 Multi-Scale OHLC
   - 如果 GAF 效果好，尝试 MTF (Markov Transition Field)
   - 调整 GAF 和 OHLC 的融合权重

## 参考文献

1. **Xiu et al. (2021)** - "(Re-)Imag(in)ing Price Trends"
   - CNN 从股价图提取信号的 53%+ 准确率
   - OHLC 条形图优于传统蜡烛图
   - 跨股票归一化的重要性

2. **Chen & Tsai (2020)** - "Encoding candlesticks as images"
   - GAF-CNN 在模式识别上达到 90.7% 准确率
   - GAF 保留时间依赖性和相关性
   - 适用于 8 种经典 K 线形态

3. **Duong et al. (2025)** - "Investigating Market Strength Prediction"
   - 蜡烛图模式检测无助于提升性能
   - 纯 CNN 从原始图像学习更有效
   - 时间序列数据优于图像转换
