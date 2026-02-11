# 论文图表状态 - SAK-Net ECCV 2026

## ✅ 已完成

### 已添加的图表

| 图表 | 文件名 | 位置 | 状态 |
|------|--------|------|------|
| **Figure 1** | encoding_comparison.pdf | Section 3.2 | ✅ 占位符已添加 |
| **Figure 2** | sector_comparison.pdf | Section 4.4 | ✅ 占位符已添加 |
| **Figure 3** | gradcam.pdf | Section 4.6 | ✅ 占位符已添加 |
| **Figure 4** | prediction_examples.pdf | Section 5 | ✅ 占位符已添加 |

### 文件结构
```
paper/
├── main.tex                    # 主论文（已更新包含图表引用）
├── main.pdf                    # 编译后PDF（12页）
├── figures/                    # 图表目录
│   ├── encoding_comparison.pdf
│   ├── encoding_comparison.png
│   ├── gradcam.pdf
│   ├── gradcam.png
│   ├── prediction_examples.pdf
│   ├── prediction_examples.png
│   ├── sector_comparison.pdf
│   └── sector_comparison.png
├── FIGURES_GUIDE.md           # 图表制作指南
└── FIGURES_STATUS.md          # 本文件
```

---

## 📝 图表详情

### Figure 1: Encoding Methods Comparison
**LaTeX引用**: `\ref{fig:encoding}`  
**页面**: Section 3.2 后

展示四种编码方法：
- (a) Candlestick RGB
- (b) OHLC Bars
- (c) GAF
- (d) Hybrid

**需要替换为**: 实际K线图编码对比

---

### Figure 2: Sector Comparison
**LaTeX引用**: `\ref{fig:sector}`  
**页面**: Section 4.4 前

柱状图展示6个sector的准确率：
- Consumer: 60.37%
- Industrials-Energy: 59.94%
- Tech-Semiconductors: 59.88%
- Tech-Software: 57.22%
- Financials: 55.90%
- Healthcare: 53.54%

**状态**: ✅ 已生成真实数据图表

---

### Figure 3: Grad-CAM Visualization
**LaTeX引用**: `\ref{fig:gradcam}`  
**页面**: Section 4.6 后

注意力可视化：
- (a) Successful prediction
- (b) Failed prediction

**需要替换为**: 实际的Grad-CAM热力图

---

### Figure 4: Prediction Examples
**LaTeX引用**: `\ref{fig:examples}`  
**页面**: Section 5 Discussion 中

预测案例展示：
- (a) Correctly predicted uptrend
- (b) Incorrectly predicted downtrend

**需要替换为**: 实际的价格走势和预测点

---

## 📊 当前论文页数

- **正文字数**: 约10页
- **图表**: 2页
- **总计**: 12页

ECCV限制: 14页 + 参考文献
剩余空间: 2页（可用于更多消融实验或补充图表）

---

## 🎯 下一步任务

### 高优先级（必须完成）
- [ ] 生成 Figure 1 的真实编码对比图
  - 使用 matplotlib/mplfinance 绘制
  - 同一段20天价格数据
  - 四种编码方法并排展示

- [ ] 生成 Figure 3 的Grad-CAM热力图
  - 使用 pytorch-grad-cam 库
  - 选择代表性的成功/失败案例
  - 将热力图叠加在原始图像上

### 中优先级（建议完成）
- [ ] 生成 Figure 4 的预测案例
  - 绘制价格曲线
  - 标记预测点和实际走势
  - 展示至少一个成功案例和一个失败案例

### 可选（如有时间）
- [ ] 添加 Figure 5: 训练曲线
  - 展示不同sector的收敛过程
  
- [ ] 添加 Figure 6: 混淆矩阵
  - 展示分类结果的详细分布

---

## 🔧 生成真实图表的代码

### Figure 1: 编码对比
```python
# 参考: scripts/generate_paper_figures.py
# 需要使用项目中的实际编码函数:
# - core/utils/kline_renderer.py (Candlestick)
# - src/data/image_generator.py (OHLC, GAF, Hybrid)
```

### Figure 3: Grad-CAM
```python
# 安装: pip install grad-cam
# 参考: https://github.com/jacobgil/pytorch-grad-cam

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layers = [model.backbone.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
```

---

## 📧 需要帮助？

如果需要我帮你：
1. 生成真实的K线图编码对比
2. 生成Grad-CAM可视化
3. 生成预测案例图

请告诉我，我可以进一步完善代码！
