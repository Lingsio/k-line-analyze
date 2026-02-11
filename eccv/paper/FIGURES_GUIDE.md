# ECCV 2026 论文图表准备指南

## 📊 需要准备的图表清单

### Figure 1: 编码方法对比 (encoding_comparison.pdf/png)
**位置**: Section 3.2 (Image Encoding Methods)  
**描述**: 展示四种编码方法对同一段价格数据的编码结果  
**布局**: 1行4列的子图

| 子图 | 标题 | 内容 |
|------|------|------|
| (a) | Candlestick RGB | 传统红绿蜡烛图，128x128 |
| (b) | OHLC Bars | 稀疏条形图（我们的方法），128x128 |
| (c) | GAF | Gramian Angular Field热力图 |
| (d) | Hybrid | 多通道混合编码 |

**制作建议**:
```python
# 使用 matplotlib + mplfinance
# 取同一只股票的连续20天数据
# 分别用四种方法编码并保存为4个子图
```

---

### Figure 2: Sector结果对比柱状图 (sector_comparison.pdf/png)
**位置**: Section 4.4 (Sector-Adaptive Results)  
**描述**: 展示6个sector的准确率对比  
**布局**: 柱状图，X轴为sector，Y轴为准确率

**数据**:
```
Consumer: 60.37%
Industrials-Energy: 59.94%
Tech-Semiconductors: 59.88%
Tech-Software: 57.22%
Financials: 55.90%
Healthcare: 53.54%
Universal baseline: 51.26%
Random baseline: 50.00%
```

**样式**:
- 蓝色柱子：SAK-Net结果
- 灰色虚线：Universal baseline (51.26%)
- 红色虚线：Random baseline (50.00%)

---

### Figure 3: Grad-CAM注意力可视化 (gradcam.pdf/png)
**位置**: Section 4.6 (Ablation Studies) 后  
**描述**: 展示模型关注的区域  
**布局**: 1行2列

| 子图 | 标题 | 内容 |
|------|------|------|
| (a) | Successful prediction | 成功预测案例的Grad-CAM热力图 |
| (b) | Failed prediction | 失败预测案例的Grad-CAM热力图 |

**制作建议**:
```python
# 使用 pytorch-grad-cam 库
# 选择预测置信度最高和最低的案例
# 将热力图叠加在原始OHLC图像上
```

---

### Figure 4: 预测案例展示 (prediction_examples.pdf/png)
**位置**: Section 5 (Discussion) - Prediction Examples  
**描述**: 展示实际预测案例  
**布局**: 2行2列或1行2列

| 子图 | 标题 | 内容 |
|------|------|------|
| (a) | Correctly predicted uptrend | 正确预测的上涨案例，显示输入窗口和预测后的实际走势 |
| (b) | Incorrectly predicted downtrend | 错误预测的案例 |

**制作建议**:
```python
# 绘制价格曲线
# 用绿色箭头标记预测时间点
# 用虚线显示预测后的实际走势
```

---

## 🎨 图表格式要求

### 尺寸规范
- **单栏图**: 宽度 8.5cm (约3.35英寸)
- **双栏图**: 宽度 17.5cm (约6.9英寸)
- **分辨率**: 300 DPI 以上
- **格式**: PDF (首选) 或 PNG

### 字体规范
- 使用 LaTeX 默认字体 (Computer Modern)
- 图中文字大小应与正文一致 (9-10pt)
- 标签使用粗体

### 颜色规范
- 蓝色: #0066CC (主色)
- 红色: #CC0000 (下跌/失败)
- 绿色: #00AA00 (上涨/成功)
- 灰色: #666666 (基线)

---

## 📝 图表标题规范

每个图表标题应包含：
1. **简短描述** (标题)
2. **关键发现** (在caption中说明)

示例：
```latex
\caption{Comparison of four encoding methods for the same 20-day price window. 
The sparse OHLC bar representation (b) achieves the best performance by 
eliminating color noise while preserving structural information.}
```

---

## 🔧 推荐工具

### Python
```bash
pip install matplotlib mplfinance seaborn pytorch-grad-cam
```

### MATLAB
- 金融工具箱可用于绘制K线图

### 在线工具
- [Plotly](https://plotly.com/) - 交互式图表
- [Adobe Illustrator](https://www.adobe.com/products/illustrator.html) - 后期编辑

---

## ✅ 检查清单

提交前确保：
- [ ] 所有图表已生成并放入 `paper/figures/` 目录
- [ ] 图表分辨率 >= 300 DPI
- [ ] 图中文字清晰可读
- [ ] 颜色在黑白打印时仍可区分（使用不同线型/标记）
- [ ] 所有子图都有标签 (a), (b), (c), (d)
- [ ] 图表在正文中被正确引用 (\ref{fig:xxx})

---

## 📂 文件结构

```
paper/
├── main.tex
├── main.bib
├── figures/
│   ├── encoding_comparison.pdf    # Figure 1
│   ├── sector_comparison.pdf      # Figure 2
│   ├── gradcam.pdf                # Figure 3
│   └── prediction_examples.pdf    # Figure 4
└── FIGURES_GUIDE.md               # 本文件
```

---

## 💡 快速开始

### 生成示例代码框架

```python
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

# 加载数据
data = pd.read_csv('data/raw/us/AAPL.csv')

# Figure 1: 编码对比
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
# ... 绘制四种编码 ...
plt.savefig('paper/figures/encoding_comparison.pdf', dpi=300, bbox_inches='tight')

# Figure 2: Sector对比
fig, ax = plt.subplots(figsize=(8, 5))
# ... 绘制柱状图 ...
plt.savefig('paper/figures/sector_comparison.pdf', dpi=300, bbox_inches='tight')
```

需要帮助生成具体图表？请告诉我！
