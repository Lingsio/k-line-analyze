# K-Line Pattern Finder

基于深度学习的K线形态相似度检索引擎，支持用户通过选取当前股票K线片段，在历史数据中查找相似走势，并提供后续走势预测分析。

## 功能特点

- **多市场支持**: 美股、台股、A股、港股、加密货币
- **智能形态匹配**: CNN + DTW 混合算法
- **向量检索**: FAISS 高效相似度搜索
- **统计分析**: 胜率、平均收益率、置信度评估
- **专业 UI**: TradingView 风格深色主题

## 技术栈

### 后端
- Python 3.11+
- FastAPI
- PyTorch (ResNet18 + Triplet Loss)
- FAISS
- yfinance, AKShare, CCXT

### 前端
- React 18 + TypeScript
- TradingView Lightweight Charts
- Tailwind CSS
- Recharts

## 快速开始

### 1. 安装依赖

```bash
# 后端
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 前端
cd frontend
npm install
```

### 2. 启动服务

```bash
# 后端 (终端 1)
cd backend
uvicorn app.main:app --reload --port 8000

# 前端 (终端 2)
cd frontend
npm run dev
```

### 3. 访问应用

打开浏览器访问 http://localhost:5173

## Docker 部署

```bash
docker-compose up -d
```

## 使用流程

1. **选择市场和股票**: 在顶部选择市场类型，输入股票代码
2. **选取 K 线区域**: 在图表上点击拖动选择一段 K 线形态
3. **设置搜索参数**: 选择搜索范围和返回结果数量
4. **查看相似结果**: 系统返回历史上最相似的 K 线片段
5. **分析预测**: 查看相似形态的后续走势统计

## 项目结构

```
k-line-analyze/
├── backend/                 # Python 后端
│   ├── app/
│   │   ├── api/routes/     # API 路由
│   │   ├── services/       # 业务逻辑
│   │   ├── models/         # 深度学习模型
│   │   └── utils/          # 工具函数
│   ├── trained_models/     # 训练好的模型
│   └── faiss_index/        # FAISS 索引
├── frontend/               # React 前端
│   ├── src/
│   │   ├── components/     # UI 组件
│   │   ├── services/       # API 服务
│   │   ├── hooks/          # React Hooks
│   │   └── types/          # TypeScript 类型
├── scripts/                # 脚本工具
│   ├── train_model.py      # 模型训练
│   ├── build_index.py      # 索引构建
│   └── download_data.py    # 数据下载
└── data/                   # 数据目录
```

## API 文档

启动后端后访问 http://localhost:8000/docs 查看 Swagger API 文档。

### 主要接口

- `GET /api/v1/stock/{symbol}/kline` - 获取 K 线数据
- `POST /api/v1/search/similar` - 搜索相似形态
- `GET /api/v1/analysis/statistics` - 获取全局统计

## 构建索引

首次使用需要构建历史数据索引：

```bash
# 1. 下载历史数据
python scripts/download_data.py --markets us tw

# 2. 训练模型 (可选，已提供预训练)
python scripts/train_model.py

# 3. 构建索引
python scripts/build_index.py --markets us tw --window-size 60
```

## 配置说明

### 后端配置 (backend/.env)

```env
DEBUG=true
CORS_ORIGINS=["http://localhost:5173"]
EMBEDDING_DIM=256
IMAGE_SIZE=128
```

### 前端配置 (frontend/.env)

```env
VITE_API_URL=http://localhost:8000
```

## 注意事项

1. **数据源限制**: 部分数据源有请求频率限制
2. **存储需求**: 完整索引需要 ~100GB 存储空间
3. **GPU 加速**: 训练时建议使用 CUDA GPU
4. **风险提示**: 历史表现不代表未来收益

## 许可证

MIT License
