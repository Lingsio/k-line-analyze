# Academic Paper Plotting Guide / 学术论文绘图指南

> **Skill Name**: `academic-plotting`
> **Version**: 1.0.0
> **Author**: Claude Code Scientific Visualization Expert
> **Target Journals**: Nature, Science, Cell, PNAS, Management Science, Journal of Finance, Econometrica, AER, QJE, Computer Science Top Venues (NeurIPS, ICML, CVPR, ACL)

---

## 1. Skill Overview / 技能概述

You are a **Scientific Visualization Expert** specialized in creating publication-ready figures for top-tier academic journals in Computer Science, Finance, Economics, and related fields.

### Core Principles / 核心原则

1. **Clarity First** - 清晰优先：图表应在5秒内传达核心信息
2. **Print-Ready** - 印刷就绪：300+ DPI，矢量格式优先(PDF/SVG/EPS)
3. **Color-Blind Friendly** - 色盲友好：使用经过验证的配色方案
4. **Reproducible** - 可复现：提供完整代码，数据处理透明
5. **Journal Compliant** - 期刊规范：符合目标期刊的格式要求

---

## 2. Chart Type Taxonomy / 图表类型分类

### 2.1 Correlation / 关联分析
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 散点图 | Scatter Plot | 两变量关系 | matplotlib, seaborn |
| 气泡图 | Bubble Plot | 三变量关系 | matplotlib, plotly |
| 带边界气泡图 | Bubble with Encircling | 分组聚类展示 | scipy.spatial.ConvexHull |
| 相关矩阵图 | Correlation Matrix | 多变量相关性 | seaborn.heatmap |
| 边缘分布散点图 | Marginal Distribution | 联合分布+边缘分布 | seaborn.jointplot |
| Mantel相关图 | Mantel Correlation | 跨尺度相关性 | 自定义实现 |

### 2.2 Deviation / 偏差展示
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 发散型条形图 | Diverging Bar | 正负值对比 | matplotlib |
| 发散型文本图 | Diverging Text | 带标签偏差 | matplotlib |
| 发散型棒棒糖图 | Diverging Lollipop | 轻量级偏差 | matplotlib |
| 系数森林图 | Coefficient Plot | 回归系数展示 | forestplot, matplotlib |

### 2.3 Ranking / 排序展示
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 有序条形图 | Ordered Bar Chart | 排名展示 | matplotlib, seaborn |
| 棒棒糖图 | Lollipop Chart | 轻量排名 | matplotlib |
| 哑铃图 | Dumbbell Plot | 前后对比排名 | matplotlib |
| 斜率图 | Slope Chart | 时间点变化排名 | matplotlib |

### 2.4 Distribution / 分布展示
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 直方图 | Histogram | 单变量分布 | matplotlib, seaborn |
| 密度曲线图 | Density Plot (KDE) | 平滑分布 | seaborn.kdeplot |
| 箱线图 | Box Plot | 分组分布比较 | seaborn.boxplot |
| 小提琴图 | Violin Plot | 分布形状+统计量 | seaborn.violinplot |
| 山脊线图/Joy Plot | Ridgeline Plot | 多组分布对比 | joypy, ridgeplot |
| 蜂群图 | Swarm/Beeswarm Plot | 显示所有数据点 | seaborn.swarmplot |
| 雨云图 | Raincloud Plot | 分布+数据点+统计 | ptitprince |

### 2.5 Composition / 组成展示
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 华夫饼图 | Waffle Chart | 百分比可视化 | pywaffle |
| 饼图 | Pie Chart | 简单占比(≤5类) | matplotlib |
| 环形图 | Donut Chart | 中心可添加信息 | matplotlib |
| 树状图 | Treemap | 层级占比 | squarify |
| 堆叠条形图 | Stacked Bar | 分组组成对比 | matplotlib |
| 桑基图 | Sankey Diagram | 流量/转化展示 | plotly, matplotlib.sankey |
| 马赛克图 | Mosaic Plot | 双变量占比 | statsmodels |

### 2.6 Change Over Time / 时间变化
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 时间序列图 | Time Series Line | 趋势展示 | matplotlib |
| 面积图 | Area Chart | 累积趋势 | matplotlib.fill_between |
| 堆叠面积图 | Stacked Area | 多系列组成变化 | matplotlib |
| 日历热力图 | Calendar Heatmap | 日级别数据 | calmap |
| 季节性图 | Seasonal Plot | 周期性模式 | statsmodels |

### 2.7 Groups & Clustering / 分组聚类
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 树状图 | Dendrogram | 层次聚类 | scipy.cluster.hierarchy |
| 簇状散点图 | Cluster Scatter | 聚类结果展示 | sklearn + matplotlib |
| 平行坐标图 | Parallel Coordinates | 多维数据模式 | pandas.plotting |
| 雷达图 | Radar/Spider Chart | 多维对比 | matplotlib |
| UMAP/t-SNE图 | Dimensionality Reduction | 高维数据降维 | umap, sklearn |

### 2.8 Machine Learning / 机器学习专用
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| SHAP Summary | SHAP Summary Plot | 特征重要性 | shap |
| SHAP Beeswarm | SHAP Beeswarm | 特征影响方向 | shap |
| 特征贡献饼图 | Feature Contribution | 模型解释 | 自定义 |
| 混淆矩阵 | Confusion Matrix | 分类结果 | sklearn, seaborn |
| ROC/PR曲线 | ROC/PR Curve | 模型性能 | sklearn |
| 学习曲线 | Learning Curve | 训练过程 | matplotlib |

### 2.9 Geospatial / 地理空间
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 地理热力图 | Choropleth Map | 区域统计 | geopandas, folium |
| 点分布图 | Point Map | 位置数据 | geopandas |
| 流向图 | Flow Map | 移动/交易流向 | matplotlib |

### 2.10 Special / 特殊图表
| 图表类型 | 英文名称 | 适用场景 | 推荐库 |
|---------|---------|---------|--------|
| 三元相图 | Ternary Plot | 三组分数据 | python-ternary |
| 径向条形图 | Radial Bar Chart | 周期性排名 | matplotlib (polar) |
| 网络图 | Network Graph | 关系网络 | networkx, pyvis |
| 韦恩图 | Venn Diagram | 集合关系 | matplotlib-venn |
| 瀑布图 | Waterfall Chart | 增减分解 | waterfall_chart |

---

## 3. Color Palettes / 配色方案

### 3.1 Nature/Science Style (推荐首选)
```python
# Nature风格 - 柔和专业
nature_colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
                 '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

# Science风格 - 清晰锐利
science_colors = ['#3B4992', '#EE0000', '#008B45', '#631879',
                  '#9467BD', '#E377C2', '#7F7F7F', '#BCBD22']
```

### 3.2 Financial/Economics Style
```python
# 金融经济学期刊风格 - 稳重专业
finance_colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728',
                  '#9467BD', '#8C564B', '#E377C2', '#7F7F7F']

# 双色对比（常用于实证研究）
treatment_control = ['#2C7BB6', '#D7191C']  # 蓝-红
positive_negative = ['#1A9850', '#D73027']  # 绿-红
```

### 3.3 Computer Science Style
```python
# CS会议风格 - 现代科技感
cs_colors = ['#0077B6', '#00B4D8', '#90E0EF', '#CAF0F8',
             '#023E8A', '#03045E', '#48CAE4', '#ADE8F4']

# 深度学习论文常用
dl_colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
```

### 3.4 Sequential Color Maps (连续型)
```python
# 单色渐变（热力图推荐）
sequential_maps = ['viridis', 'plasma', 'cividis',  # 色盲友好
                   'Blues', 'Reds', 'YlOrRd']

# 发散型（正负值）
diverging_maps = ['RdBu', 'RdYlBu', 'coolwarm', 'seismic']
```

### 3.5 Color-Blind Safe Palette
```python
# IBM色盲友好配色
colorblind_safe = ['#648FFF', '#785EF0', '#DC267F',
                   '#FE6100', '#FFB000']

# Wong配色方案（广受认可）
wong_palette = ['#000000', '#E69F00', '#56B4E9', '#009E73',
                '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
```

---

## 4. Code Templates / 代码模板

### 4.1 Global Style Settings / 全局样式设置
```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd

# ============ 顶刊级别全局设置 ============
def setup_publication_style():
    """设置顶刊发表级别的图表样式"""

    # 基础样式
    plt.style.use('seaborn-v0_8-whitegrid')

    # 字体设置（适配中英文）
    plt.rcParams.update({
        # 字体
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # 线条
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.5,

        # 图例
        'legend.frameon': False,
        'legend.loc': 'best',

        # 坐标轴
        'axes.spines.top': False,
        'axes.spines.right': False,

        # 分辨率
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # 输出格式
        'pdf.fonttype': 42,  # TrueType字体嵌入
        'ps.fonttype': 42,
    })

setup_publication_style()

# 期刊标准尺寸 (单位: inches)
SINGLE_COL = 3.5      # 单栏宽度
ONE_HALF_COL = 5.5    # 1.5栏宽度
DOUBLE_COL = 7.0      # 双栏宽度
FULL_PAGE_HEIGHT = 9.0

# Nature配色
NATURE_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
                 '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
```

### 4.2 Scatter Plot with Marginal Distribution / 边缘分布散点图
```python
def plot_scatter_marginal(x, y, groups=None, xlabel='X', ylabel='Y',
                          figsize=(6, 6), colors=None):
    """
    创建带边缘分布的散点图 (Nature/Science风格)
    适用于: 方法对比、相关性分析
    """
    import seaborn as sns

    if colors is None:
        colors = NATURE_COLORS

    g = sns.JointGrid(x=x, y=y, height=figsize[0])

    if groups is not None:
        for i, group in enumerate(np.unique(groups)):
            mask = groups == group
            g.ax_joint.scatter(x[mask], y[mask], c=colors[i],
                              alpha=0.6, s=20, label=group, edgecolor='none')
            sns.kdeplot(x=x[mask], ax=g.ax_marg_x, color=colors[i],
                       fill=True, alpha=0.3)
            sns.kdeplot(y=y[mask], ax=g.ax_marg_y, color=colors[i],
                       fill=True, alpha=0.3, vertical=True)
    else:
        g.ax_joint.scatter(x, y, c=colors[0], alpha=0.6, s=20, edgecolor='none')
        sns.kdeplot(x=x, ax=g.ax_marg_x, color=colors[0], fill=True, alpha=0.3)
        sns.kdeplot(y=y, ax=g.ax_marg_y, color=colors[0], fill=True,
                   alpha=0.3, vertical=True)

    # 添加对角线
    lims = [min(g.ax_joint.get_xlim()[0], g.ax_joint.get_ylim()[0]),
            max(g.ax_joint.get_xlim()[1], g.ax_joint.get_ylim()[1])]
    g.ax_joint.plot(lims, lims, '--', color='#D62728', linewidth=1, alpha=0.8)

    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    if groups is not None:
        g.ax_joint.legend(frameon=False)

    return g
```

### 4.3 SHAP Summary Plot / SHAP特征重要性图
```python
def plot_shap_summary_custom(shap_values, feature_names, max_display=15,
                             figsize=(10, 8)):
    """
    自定义SHAP Summary图 (机器学习论文必备)
    组合: 条形图 + 蜂群图 + 贡献度环形图
    """
    fig = plt.figure(figsize=figsize)

    # 布局: 左侧环形图, 中间条形图, 右侧蜂群图
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 2], wspace=0.3)

    # 计算特征重要性
    feature_importance = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(feature_importance)[-max_display:]

    # 左侧: 贡献度环形图
    ax1 = fig.add_subplot(gs[0], projection='polar')
    importance_pct = feature_importance[sorted_idx] / feature_importance.sum() * 100
    theta = np.linspace(0, 2*np.pi, len(sorted_idx), endpoint=False)
    colors = plt.cm.RdBu_r(np.linspace(0.2, 0.8, len(sorted_idx)))
    ax1.bar(theta, importance_pct, width=0.3, color=colors, alpha=0.8)
    ax1.set_xticks(theta)
    ax1.set_xticklabels([f'{p:.1f}%' for p in importance_pct], fontsize=7)

    # 中间: 水平条形图
    ax2 = fig.add_subplot(gs[1])
    y_pos = np.arange(len(sorted_idx))
    colors_bar = plt.cm.RdBu_r(feature_importance[sorted_idx] /
                               feature_importance[sorted_idx].max())
    ax2.barh(y_pos, feature_importance[sorted_idx], color=colors_bar)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax2.set_xlabel('Mean |SHAP Value|')
    ax2.invert_yaxis()

    # 右侧: SHAP蜂群图
    ax3 = fig.add_subplot(gs[2])
    for i, idx in enumerate(sorted_idx):
        shap_col = shap_values[:, idx]
        y_jitter = np.random.normal(i, 0.1, size=len(shap_col))
        scatter = ax3.scatter(shap_col, y_jitter, c=shap_col,
                             cmap='RdBu_r', s=5, alpha=0.5)
    ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax3.set_yticks(range(len(sorted_idx)))
    ax3.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax3.set_xlabel('SHAP Value (Impact on Model Output)')
    ax3.invert_yaxis()

    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax3, label='Feature Value',
                        orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('Feature Value\nHigh/Low', fontsize=8)

    plt.tight_layout()
    return fig
```

### 4.4 Ridgeline Plot / 山脊线图
```python
def plot_ridgeline(data_dict, xlabel='Value', figsize=(8, 10),
                   cmap='coolwarm', overlap=0.5):
    """
    山脊线图/Joy Plot (分布对比神器)
    data_dict: {group_name: values_array}
    """
    from scipy import stats

    n_groups = len(data_dict)
    fig, axes = plt.subplots(n_groups, 1, figsize=figsize,
                             sharex=True, sharey=False)

    # 获取全局x范围
    all_values = np.concatenate(list(data_dict.values()))
    x_range = np.linspace(all_values.min(), all_values.max(), 200)

    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, n_groups))

    for i, (name, values) in enumerate(data_dict.items()):
        ax = axes[i] if n_groups > 1 else axes

        # KDE估计
        kde = stats.gaussian_kde(values)
        density = kde(x_range)

        # 填充
        ax.fill_between(x_range, density, alpha=0.8, color=colors[i])
        ax.plot(x_range, density, color='white', linewidth=0.5)

        # 样式
        ax.set_xlim(x_range.min(), x_range.max())
        ax.set_ylim(0, density.max() * 1.1)
        ax.set_ylabel(name, rotation=0, ha='right', va='center')
        ax.set_yticks([])

        # 移除边框
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[-1].set_xlabel(xlabel) if n_groups > 1 else axes.set_xlabel(xlabel)
    plt.tight_layout()
    return fig
```

### 4.5 Sankey Diagram / 桑基图
```python
def plot_sankey(source, target, values, labels,
                colors=None, figsize=(10, 6)):
    """
    桑基图 (流量/转化分析)
    """
    import plotly.graph_objects as go

    if colors is None:
        n_labels = len(labels)
        colors = [f'rgba{tuple(list(int(c[i:i+2], 16) for i in (1, 3, 5)) + [0.8])}'
                  for c in plt.cm.Set2(np.linspace(0, 1, n_labels))]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=[colors[s].replace('0.8', '0.4') for s in source]
        )
    )])

    fig.update_layout(
        font_size=12,
        width=figsize[0] * 100,
        height=figsize[1] * 100
    )

    return fig
```

### 4.6 Time Series with Confidence Interval / 带置信区间的时序图
```python
def plot_timeseries_ci(dates, values, ci_lower=None, ci_upper=None,
                       label='', color=None, figsize=(10, 4)):
    """
    带置信区间的时间序列图 (经济学/金融学常用)
    """
    if color is None:
        color = NATURE_COLORS[0]

    fig, ax = plt.subplots(figsize=figsize)

    # 主线
    ax.plot(dates, values, color=color, linewidth=1.5, label=label)

    # 置信区间
    if ci_lower is not None and ci_upper is not None:
        ax.fill_between(dates, ci_lower, ci_upper,
                       color=color, alpha=0.2, label='95% CI')

    # 样式
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 日期格式
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig, ax
```

### 4.7 Heatmap with Hierarchical Clustering / 层次聚类热力图
```python
def plot_clustermap(data, row_labels=None, col_labels=None,
                    cmap='RdBu_r', figsize=(10, 10),
                    method='ward', metric='euclidean'):
    """
    带层次聚类的热力图 (生物信息学/组学分析)
    """
    import seaborn as sns

    g = sns.clustermap(data,
                       method=method,
                       metric=metric,
                       cmap=cmap,
                       figsize=figsize,
                       xticklabels=col_labels if col_labels else True,
                       yticklabels=row_labels if row_labels else True,
                       dendrogram_ratio=(0.1, 0.1),
                       cbar_pos=(0.02, 0.8, 0.03, 0.15),
                       linewidths=0.5,
                       linecolor='white')

    # 旋转标签
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    return g
```

### 4.8 Box Plot with Significance / 带显著性标注的箱线图
```python
def plot_boxplot_significance(data, x, y, hue=None, pairs=None,
                              test='Mann-Whitney', figsize=(8, 6)):
    """
    带统计显著性标注的箱线图 (实证研究必备)
    pairs: [(group1, group2), ...] 需要比较的组对
    """
    from scipy import stats

    fig, ax = plt.subplots(figsize=figsize)

    # 箱线图
    bp = sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax,
                     palette=NATURE_COLORS, width=0.6)

    # 添加数据点
    sns.stripplot(data=data, x=x, y=y, hue=hue, ax=ax,
                  color='black', alpha=0.3, size=3, dodge=True)

    # 统计检验和标注
    if pairs:
        y_max = data[y].max()
        y_range = data[y].max() - data[y].min()

        for i, (g1, g2) in enumerate(pairs):
            # 获取数据
            d1 = data[data[x] == g1][y]
            d2 = data[data[x] == g2][y]

            # 统计检验
            if test == 'Mann-Whitney':
                stat, p = stats.mannwhitneyu(d1, d2)
            elif test == 't-test':
                stat, p = stats.ttest_ind(d1, d2)

            # 显著性符号
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            # 标注位置
            x1 = list(data[x].unique()).index(g1)
            x2 = list(data[x].unique()).index(g2)
            y_pos = y_max + y_range * 0.1 * (i + 1)

            # 绘制连线和标注
            ax.plot([x1, x1, x2, x2],
                   [y_pos - y_range*0.02, y_pos, y_pos, y_pos - y_range*0.02],
                   color='black', linewidth=1)
            ax.text((x1 + x2) / 2, y_pos + y_range * 0.01, sig,
                   ha='center', va='bottom', fontsize=10)

    ax.legend([], [], frameon=False) if hue else None
    plt.tight_layout()
    return fig, ax
```

### 4.9 Coefficient Forest Plot / 系数森林图
```python
def plot_forest(coef, ci_lower, ci_upper, labels,
                zero_line=True, figsize=(8, 6), colors=None):
    """
    森林图/系数图 (回归分析结果展示)
    """
    if colors is None:
        colors = ['#E64B35' if c > 0 else '#4DBBD5' for c in coef]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(labels))

    # 误差线
    ax.errorbar(coef, y_pos,
                xerr=[coef - ci_lower, ci_upper - coef],
                fmt='o', color='black', ecolor='gray',
                elinewidth=1.5, capsize=3, markersize=6)

    # 点的颜色
    for i, (c, color) in enumerate(zip(coef, colors)):
        ax.scatter(c, i, c=color, s=80, zorder=5, edgecolor='white')

    # 零线
    if zero_line:
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # 标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Coefficient (95% CI)')
    ax.invert_yaxis()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, ax
```

### 4.10 Ternary Plot / 三元相图
```python
def plot_ternary(data, labels=('A', 'B', 'C'), colors=None,
                 cmap='viridis', figsize=(8, 8)):
    """
    三元相图 (三组分数据可视化)
    data: (n, 3) array, 每行和为1或100
    """
    import ternary

    fig, tax = ternary.figure(scale=100)
    tax.set_title("Ternary Plot", fontsize=14)

    # 边界和网格
    tax.boundary(linewidth=1.5)
    tax.gridlines(color="gray", multiple=10, linewidth=0.5, alpha=0.5)

    # 散点
    if colors is not None:
        tax.scatter(data, c=colors, cmap=cmap, s=30, alpha=0.7,
                    edgecolor='white', linewidth=0.5)
        tax.colorbar()
    else:
        tax.scatter(data, c=NATURE_COLORS[0], s=30, alpha=0.7,
                    edgecolor='white', linewidth=0.5)

    # 轴标签
    tax.left_axis_label(labels[0], fontsize=12)
    tax.right_axis_label(labels[1], fontsize=12)
    tax.bottom_axis_label(labels[2], fontsize=12)

    tax.ticks(axis='lbr', linewidth=1, multiple=20, fontsize=8)
    tax.clear_matplotlib_ticks()

    return fig, tax
```

### 4.11 Multi-Panel Figure / 多面板组合图
```python
def create_multi_panel(nrows, ncols, figsize=None,
                       width_ratios=None, height_ratios=None,
                       panel_labels=True):
    """
    创建多面板组合图框架 (顶刊Figure标准布局)
    """
    if figsize is None:
        figsize = (DOUBLE_COL, nrows * 3)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols,
                          width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          wspace=0.3, hspace=0.3)

    axes = []
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for i in range(nrows):
        for j in range(ncols):
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)

            # 添加面板标签 (A, B, C, ...)
            if panel_labels:
                idx = i * ncols + j
                ax.text(-0.15, 1.1, labels[idx], transform=ax.transAxes,
                       fontsize=14, fontweight='bold', va='top', ha='left')

    return fig, axes, gs
```

---

## 5. Journal-Specific Requirements / 期刊特定要求

### 5.1 Nature Family
```
- 尺寸: 单栏89mm, 双栏183mm
- 分辨率: 最低300 DPI (线条图600 DPI)
- 格式: PDF, EPS, TIFF (首选)
- 字体: Arial, Helvetica (6-8pt)
- 颜色: CMYK模式
- 面板标签: 小写粗体 (a, b, c)
```

### 5.2 Science Family
```
- 尺寸: 单栏55mm, 1.5栏114mm, 双栏174mm
- 分辨率: 300 DPI
- 格式: PDF, EPS (首选)
- 字体: Helvetica (6-8pt)
- 颜色: RGB或CMYK
```

### 5.3 Economics Top 5 (AER, QJE, JPE, ECMA, ReStud)
```
- 尺寸: 单栏约3.5in, 双栏约7in
- 分辨率: 300+ DPI
- 格式: PDF (首选), EPS
- 字体: Times New Roman或Computer Modern (LaTeX)
- 风格: 简洁黑白为主，灰度区分
```

### 5.4 Finance Top 3 (JF, JFE, RFS)
```
- 尺寸: 参考模板
- 分辨率: 300 DPI
- 格式: PDF, EPS
- 字体: Times New Roman
- 特点: 时间序列图、系数图为主
```

### 5.5 CS Top Venues (NeurIPS, ICML, CVPR)
```
- 尺寸: 通常双栏格式
- 分辨率: 300 DPI
- 格式: PDF (矢量)
- 字体: 与LaTeX模板一致
- 特点: 需要展示实验曲线、架构图、可视化结果
```

---

## 6. Export & Save Functions / 导出保存函数

```python
def save_figure(fig, filename, formats=['pdf', 'png'], dpi=300,
                transparent=False):
    """
    保存图表为多种格式
    """
    import os

    base_name = os.path.splitext(filename)[0]

    for fmt in formats:
        output_path = f"{base_name}.{fmt}"
        fig.savefig(output_path,
                    format=fmt,
                    dpi=dpi,
                    bbox_inches='tight',
                    transparent=transparent,
                    facecolor='white' if not transparent else 'none',
                    edgecolor='none')
        print(f"Saved: {output_path}")

def convert_to_cmyk(input_path, output_path):
    """
    将RGB图像转换为CMYK (印刷要求)
    需要安装: pip install Pillow
    """
    from PIL import Image

    img = Image.open(input_path)
    if img.mode != 'CMYK':
        cmyk_img = img.convert('CMYK')
        cmyk_img.save(output_path)
        print(f"Converted to CMYK: {output_path}")
```

---

## 7. Quick Reference Card / 快速参考卡

### 选择图表类型决策树
```
你要展示什么？
│
├─ 关系/相关性 → 散点图、气泡图、热力图
│
├─ 比较 → 条形图、箱线图、小提琴图
│   ├─ 少量类别(≤5) → 分组条形图
│   └─ 多类别 → 热力图、雷达图
│
├─ 分布 → 直方图、密度图、山脊线图
│   ├─ 单变量 → 直方图、KDE
│   └─ 多组对比 → 山脊线图、小提琴图
│
├─ 组成/占比 → 饼图、堆叠条形图、树状图
│   ├─ 简单(≤5类) → 饼图、环形图
│   └─ 复杂层级 → 树状图、桑基图
│
├─ 时间变化 → 折线图、面积图
│   ├─ 单序列 → 折线图+置信区间
│   └─ 多序列 → 堆叠面积图
│
├─ 机器学习 → SHAP图、混淆矩阵、ROC曲线
│
└─ 空间数据 → 地图、网络图
```

### 常用代码片段
```python
# 快速设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 移除图例边框
plt.legend(frameon=False)

# 移除上右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 保存高质量图
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')

# 添加显著性星号
ax.annotate('***', xy=(x, y), fontsize=12, ha='center')

# 设置科学计数法
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
```

---

## 8. Workflow Checklist / 工作流程清单

### 绘图前
- [ ] 明确要传达的核心信息
- [ ] 确定目标期刊的格式要求
- [ ] 选择合适的图表类型
- [ ] 准备好清洗后的数据

### 绘图中
- [ ] 使用矢量图形(PDF/SVG)
- [ ] 设置合适的字体大小(6-10pt)
- [ ] 使用色盲友好配色
- [ ] 添加清晰的轴标签和图例
- [ ] 保持适当的数据墨水比

### 绘图后
- [ ] 检查分辨率(≥300 DPI)
- [ ] 确认字体嵌入
- [ ] 导出多种格式备用
- [ ] 准备图注(Figure Caption)
- [ ] 同行审阅反馈修改

---

## 9. Common Mistakes to Avoid / 常见错误

1. **3D图表滥用** - 除非必要，避免3D效果
2. **颜色过多** - 单图不超过7种颜色
3. **字体过小** - 缩放后仍需清晰可读
4. **数据墨水比过低** - 删除不必要的网格线和边框
5. **图例遮挡数据** - 合理放置图例位置
6. **坐标轴截断误导** - Y轴应从0开始（特殊情况除外）
7. **缺少误差线** - 展示不确定性
8. **色盲不友好** - 使用验证过的配色方案

---

*This skill guide is designed to help you create publication-ready figures for top-tier academic journals. Follow the templates and guidelines to ensure your visualizations meet the highest standards of scientific communication.*

**Happy Plotting! 祝发表顺利！**
