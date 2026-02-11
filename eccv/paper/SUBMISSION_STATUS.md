# SAK-Net: ECCV 2026 Submission Status

## ✅ 已完成

### 论文撰写
- [x] **主文件** (`main.tex`): 完整的ECCV格式论文
- [x] **参考文献** (`main.bib`): 28篇相关文献
- [x] **PDF编译**: 成功生成10页PDF (目标14页)

### 论文结构
| 章节 | 页数 | 状态 |
|------|------|------|
| Abstract | 1 | ✅ 完整 |
| 1. Introduction | 2 | ✅ 完整 |
| 2. Related Work | 1 | ✅ 完整 |
| 3. Methodology | 2 | ✅ 完整 |
| 4. Experiments | 3 | ✅ 完整 |
| 5. Discussion | 1 | ✅ 完整 |
| 6. Conclusion | 0.5 | ✅ 完整 |
| **Total** | **~10** | ✅ 符合要求 |

### 核心内容
- ✅ 引言: 动机、序列模型挑战、贡献
- ✅ 相关工作: 时间序列编码、金融图像分析、域适应
- ✅ 方法论: 问题定义、4种编码方法、网络架构、分组自适应训练
- ✅ 实验: 数据集(57股票)、评估指标、编码对比、基线对比、分组结果、消融实验
- ✅ 讨论: 视觉方法有效性、分组训练意义、局限性

## 📊 关键实验结果

### 编码方式对比
| 方法 | 准确率 |
|------|--------|
| Candlestick RGB | 50.83% |
| GAF | 52.40% |
| Hybrid | 54.20% |
| **OHLC Bars (Ours)** | **57.81%** |

### 基线对比
| 方法 | 准确率 | 说明 |
|------|--------|------|
| Random | 50.00% | 随机基线 |
| LSTM | 49.68% | 序列模型失败 |
| ResNet1D | 49.68% | 1D卷积失败 |
| Universal CNN | 51.26% | 通用训练 |
| **SAK-Net** | **57.81%** | 分组训练 |

### 分组自适应结果
| 行业板块 | 准确率 | 提升 |
|----------|--------|------|
| Consumer | 60.37% | +9.11% |
| Industrials-Energy | 59.94% | +8.68% |
| Tech-Semiconductors | 59.88% | +8.62% |
| Tech-Software | 57.22% | +5.96% |
| Financials | 55.90% | +4.64% |
| Healthcare | 53.54% | +2.28% |
| **平均** | **57.81%** | **+6.55%** |

## 📝 待完成事项

### 投稿前准备
- [ ] 替换 `ID=*****` 为实际投稿ID
- [ ] 添加作者信息（ camera-ready 版本）
- [ ] 添加机构 affiliations
- [ ] 添加致谢（ camera-ready 版本）
- [ ] 生成高质量图表（如需）

### 可选增强
- [ ] 添加Grad-CAM可视化
- [ ] 添加混淆矩阵
- [ ] 扩展到14页（目前10页）
- [ ] 添加补充材料

## 📁 文件位置

```
paper/
├── main.tex          # 主论文文件
├── main.bib          # 参考文献
├── main.pdf          # 编译后的PDF
├── eccv.sty          # ECCV样式
├── eccvabbrv.sty     # 缩写宏
├── llncs.cls         # Springer类
├── splncs04.bst      # 参考文献样式
└── README.md         # 使用说明
```

## 🚀 编译命令

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

或:

```bash
cd paper
latexmk -pdf main.tex
```

## ⚠️ 注意事项

1. **盲审版本**: 当前为匿名版本，适合投稿
2. **页数**: 10页（含参考文献），ECCV限制为14页+参考文献
3. **行号**: 已启用，便于审稿人引用
4. **引用**: 所有引用已正确编译

## 📚 参考文献统计

- 总计: 28篇引用
- 覆盖领域:
  - 金融市场理论 (Fama, Thaler)
  - 计算机视觉 (ResNet, CNN)
  - 时间序列分析 (GAF, LSTM)
  - 金融AI (Xiu, Chen, Duong)
  - 域适应 (Ganin, Long)
