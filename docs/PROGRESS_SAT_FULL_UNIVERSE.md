# SAT (SAK-Net) 全美股训练项目进度报告

> 项目目标：将原本 57.81% 准确率的 SAK-Net 模型从 57 只股票扩展到全美股 (~4500只)，并按 GICS 板块分组训练验证

---

## 1. 已完成工作

### 1.1 数据层：全美股数据获取 ✅

| 项目 | 状态 | 详情 |
|------|------|------|
| 数据源 | ✅ | yfinance (Yahoo Finance REST API) |
| 股票数量 | ✅ | **4,537 只** (从 NASDAQ Trader 注册的 5,559 只中过滤) |
| 时间范围 | ✅ | 2000-01-01 至今 (~25年) |
| 数据格式 | ✅ | Parquet (高效压缩，列式存储) |
| 质量过滤 | ✅ | min 252交易日, min $1 价格, min 10,000 日均成交量 |
| 存储位置 | ✅ | `data/raw/us/*.parquet` |

**排除原因统计：**
- 成功: 4,537 只
- 跳过: 1,017 只 (历史不足/价格太低/成交量不足)
- 失败: 5 只 (下载错误)

**关键文件：**
- `core/stock_universe.py` - NASDAQ 股票列表获取
- `core/yfinance_fetcher.py` - yfinance 自适应限速封装
- `scripts/download_all_us_stocks.py` - 主下载脚本

---

### 1.2 分组层：GICS 板块分类 ✅

**方法：** 使用 yfinance `.info['sector']` 获取 GICS 板块信息

| 板块 (Sector) | 股票数 | 说明 |
|---------------|--------|------|
| Healthcare | 878 | 医疗保健 |
| Financial_Services | 766 | 金融服务 |
| Technology | 643 | 科技 |
| Industrials | 560 | 工业 |
| Consumer_Cyclical | 484 | 非必需消费 |
| Real_Estate | 230 | 房地产 |
| Communication_Services | 219 | 通信服务 |
| Energy | 198 | 能源 |
| Consumer_Defensive | 195 | 必需消费 |
| Basic_Materials | 246 | 基础材料 |
| Utilities | 100 | 公用事业 |
| ~~Other~~ | ~~18~~ | ~~已排除~~ (杂项，非同质板块) |
| **有效总计** | **4,519** | 11 个板块 |

**注意：** `Other` 板块 (18只) 已排除，原因是该组包含无法分类或小众板块的股票，不具备同质性，不符合"板块内训练"的假设。

**关键文件：**
- `scripts/build_sector_groups.py` - 板块信息获取与分组
- `data/us_stock_groups_full.json` - 板块定义文件
- `data/sector_cache.json` - 缓存的 yfinance sector 数据

---

### 1.3 数据层：Dataset 优化 ✅

为支持高效板块训练，给 `StockDataset` 添加了 `tickers_filter` 参数：

```python
# 只加载指定板块的股票，避免加载全部 4537 只
train_ds = StockDataset(
    mode='train',
    tickers_filter=['AAPL', 'MSFT', ...],  # 只加载该板块股票
    ...
)
```

**关键文件：**
- `src/data/dataset.py` - 添加 `tickers_filter` 参数

---

### 1.4 训练层：脚本准备 ✅

**配置与 57.81% 版本保持一致：**

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | ResNet18 | pretrained=True |
| 分类 | 2-class | 涨/跌 |
| 窗口 | 20天 | 回顾周期 |
| 预测 | 5天后 | T+5 预测 |
| 图像 | 128×128 | OHLC 条形图 |
| 增强 | CLAHE | 对比度增强 |
| 阈值 | dynamic | 基于波动率动态阈值 |
| Dropout | 0.3 | 正则化 |
| Label smoothing | 0.1 | 防止过自信 |
| Batch size | 128 | H20 96GB 适配 |
| LR | 4e-4 | AdamW |
| Epochs | 50 | Early stopping patience=7 |

**训练脚本：**
- `scripts/train_full_universe.py` - 全美股板块训练

**训练流程：**
1. 对每个板块独立训练模型
2. 每个板块运行 2 个 seed (42, 142)，取最佳
3. 汇总各板块准确率，计算总体平均
4. 保存完整结果到 `outputs/full_universe/`

---

## 2. 与原版 57.81% 的对比

| 维度 | 原版 (57.81%) | 新版 (全美股) |
|------|---------------|---------------|
| 股票数 | 57 只 | 4,519 只 |
| 板块数 | 6 个 | 11 个 |
| 板块定义 | 手工分组 | GICS 标准 |
| 数据源 | TradingView | yfinance |
| 时间范围 | ~5年 | ~25年 (2000至今) |
| 模型架构 | ResNet18 | ResNet18 (相同) |
| 超参数 | 20天/5天/128px | 完全相同 |

---

## 3. 待执行任务

### 3.1 训练执行 ⏸️ (暂停中)

**状态：** 用户指示不在当前机器训练

**执行命令（准备就绪）：**
```bash
# 训练所有板块（推荐）
python scripts/train_full_universe.py --all --num-runs 2

# 或单板块测试
python scripts/train_full_universe.py --sector Technology --num-runs 2
```

**预计时间：**
- 单板块 (100-800只): 30-120 分钟
- 全部 11 板块: 8-15 小时 (H20 96GB GPU)

### 3.2 结果分析 (待训练后)

预期输出：
- `outputs/full_universe/results_YYYYMMDD_HHMMSS.json`
- 各板块准确率、F1、AUC
- 总体平均准确率（目标：验证 ~57% 或更高）

---

## 4. 文件清单

### 新增文件
```
core/stock_universe.py              # 股票列表获取
core/yfinance_fetcher.py            # yfinance 封装
scripts/download_all_us_stocks.py   # 全美股下载
scripts/build_sector_groups.py      # 板块分组构建
scripts/train_full_universe.py      # 全美股训练脚本
data/us_stock_groups_full.json      # 板块定义 (11 sectors, 4519 stocks)
data/sector_cache.json              # yfinance sector 缓存
```

### 修改文件
```
core/tradingview_fetcher.py         # 使 tvDatafeed 可选
src/data/dataset.py                 # 添加 tickers_filter 参数
```

### 数据文件
```
data/raw/us/*.parquet               # 4537 只股票数据
```

---

## 5. 关键决策记录

| 决策 | 原因 |
|------|------|
| 使用 yfinance | TradingView WebSocket 不稳定，yfinance REST API 更可靠 |
| 排除 Other 板块 | 18 只股票为杂项，不具备同质板块特征，训练无意义 |
| 保持 57.81% 配置 | 控制变量，验证扩展性而非调优 |
| tickers_filter | 避免每次训练加载全部 4537 只，仅加载目标板块 |

---

## 6. 下一步行动 (用户决定)

选项 A: **迁移到云端训练**
- 打包数据 + 代码
- 生成 Docker 或云端启动脚本

选项 B: **本地其他机器**
- 数据复制方案
- 环境配置检查清单

选项 C: **先小样本验证**
- 取 2-3 个板块先试跑
- 验证代码正确性后再全量

---

**报告时间：** 2026-02-13  
**数据就绪：** 4,519 只股票 / 11 板块  
**训练状态：** 准备就绪，等待执行
