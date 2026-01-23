# K-Line Pattern Finder
# AI-Driven Financial Time Series Similarity Search / AI 驱动的金融形态相似度搜索引擎

<div align="center">

[English](#english-version) | [中文版本](#中文版本)

</div>

---

## English Version


### Abstract

The **K-Line Pattern Finder** is a sophisticated financial analysis system designed to identify historical recurrence of market behaviors through computer vision and vector similarity search. By encoding candlestick chart patterns into high-dimensional vector embeddings using Convolutional Neural Networks (CNN), the system enables real-time retrieval of historically similar price structures across global financial markets. Integrated with Large Language Models (LLM), it provides automated technical analysis and actionable market insights.

### System Architecture

The system operates on a decoupled client-server architecture, comprising three primary subsystems:

1.  **Vector Search Engine (Core)**: Handles embedding storage and similarity retrieval using FAISS.
2.  **Analytical Backend (API)**: Python/FastAPI service orchestrating data flow, AI analysis, and market data fetching.
3.  **Visualization Frontend (UI)**: React-based interface for interactive charting and pattern comparison.

#### Feature Extraction & Vector Indexing
*   **Data Acquisition**: Aggregates historical OHLCV data from US, CN, HK, TW, and Crypto markets.
*   **Feature Encoding**: Pre-trained CNN model maps 60-day price patterns into 256-dimensional vectors.
*   **Indexing**: Utilizes FAISS IVF-PQ (Inverted File with Product Quantization) for scalable, sub-millisecond retrieval.

### Installation and Setup

#### Prerequisites
*   Python 3.8+
*   Node.js 16+
*   CUDA-compatible GPU (Recommended)

#### Backend Configuration
```bash
cd backend
pip install -r requirements.txt
# Configure .env with LLM_API_KEY
python -m uvicorn app.main:app --reload
```

#### Frontend Configuration
```bash
cd frontend
npm install
npm run dev
```

### Index Building

To enable similarity search, a comprehensive vector index must be built:

```bash
python scripts/build_index.py
```
This script automates the fetching of full-market symbol lists and constructs the vector index with GPU acceleration.

### License

This project is licensed under the [MIT License](LICENSE).

---

## 中文版本


### 摘要

**K-Line Pattern Finder** 是一个先进的金融分析系统，旨在通过计算机视觉和向量相似度搜索技术，识别市场行为的历史重现性。该系统利用卷积神经网络（CNN）将K线图表形态编码为高维向量嵌入，从而能够在全球金融市场中实时检索历史上相似的价格结构。结合大语言模型（LLM），系统能够提供自动化的技术分析和可执行的市场洞察。

### 系统架构

本系统采用解耦的客户端-服务器架构，由三个主要子系统组成：

1.  **向量搜索引擎（核心）**：使用 FAISS 处理向量存储和相似度检索。
2.  **分析后端（API）**：基于 Python/FastAPI，负责数据流编排、AI 分析和市场数据获取。
3.  **可视化前端（UI）**：基于 React 的交互式界面，用于图表展示和形态对比。

#### 特征提取与向量索引
*   **数据采集**：聚合美股、A股、港股、台股及加密货币市场的历史 OHLCV 数据。
*   **特征编码**：预训练 CNN 模型将 60 天的价格形态映射为 256 维向量。
*   **索引构建**：采用 FAISS IVF-PQ（带乘积量化的倒排索引）结构，实现可扩展的亚毫秒级检索。

### 安装与配置

#### 前置条件
*   Python 3.8+
*   Node.js 16+
*   CUDA 兼容显卡（推荐）

#### 后端配置
```bash
cd backend
pip install -r requirements.txt
# 在 .env 文件中配置 LLM_API_KEY
python -m uvicorn app.main:app --reload
```

#### 前端配置
```bash
cd frontend
npm install
npm run dev
```

### 索引构建

为了启用相似度搜索，需要构建完整的数据索引：

```bash
python scripts/build_index.py
```
该脚本会自动抓取全市场的股票代码列表，并利用 GPU 加速构建向量索引。

### 许可

本项目采用 [MIT License](LICENSE) 许可证。
