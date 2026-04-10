# 带有语义缓存的高效 AI Agent (Fast AI Agents with a Semantic Cache)

本项目实现了一个高级的 AI Agentic 工作流，综合使用了 [LangGraph](https://python.langchain.com/docs/langgraph) 和由 [RedisVL](https://github.com/RedisVentures/redisvl) 支持的语义缓存（Semantic Cache）。它通过存储和复用过去对语义相似查询的响应，大幅优化了 API 请求成本并降低了响应延迟。

## 核心特性
- **LangGraph Agent 工作流**: 智能地将复杂查询分解为子问题、检查缓存、通过 LLM 进行资料检索和研究、评估回答的质量，并最终综合生成完整的响应。
- **语义缓存 (Semantic Caching)**: 利用 `redisvl` 和本地 HuggingFace 嵌入模型 (`all-MiniLM-L6-v2`) 搜索并检索以前对相同或高度相似用户查询的回答。
- **大模型与追踪系统**: LLM 推理基于字节跳动火山引擎 (ARK) 提供支持，同时可选开启 LangSmith 对 Agent 的所有推理步骤和链路进行追踪监控。

## 前置要求

1. **Python 3.9+**
2. **本地 Redis 服务:** 确保本地已启动 Redis 服务并监听在 `6379` 端口。你可以通过 Docker 非常方便地启动一个实例：
    ```bash
    docker run -d -p 6379:6379 redis
    ```

## 安装与配置

### 1. 克隆仓库
```bash
git clone <your-github-repo-url>
cd agentCache
```

### 2. 设置虚拟环境
强烈推荐使用 Python 虚拟环境 (`.venv`) 来隔离项目依赖。
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量
在项目根目录创建一个名为 `.env` 的文件，填入你的 API 密钥：

> **避坑提醒**: 在 `.env` 中填写 API Key 时，**千万不要**在前后加反引号 (`` ` ``) 或单引号 (`'`)，除非你的密钥本身包含空格等特殊字符。直接粘贴纯文本即可。

```env
# Redis 配置
REDIS_URL=redis://localhost:6379
CACHE_NAME=semantic-cache
CACHE_DISTANCE_THRESHOLD=0.3
CACHE_TTL_SECONDS=3600

# 火山引擎 (ARK) API Key (运行代码必需)
ARK_API_KEY=your_ark_api_key_here

# 搜索引擎 API Key (如果用到 Tavily 搜索代理)
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith 追踪 (可选)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
```

## 运行项目

### 选项 A: 运行标准 Python 脚本（推荐作为项目展示）
你可以直接在终端中自动执行主 LangGraph 工作流跑完所有的测试场景（Scenarios）。这是面试或展示给别人看最直观的运行方式。
```bash
python main.py
```

### 选项 B: 交互式 Jupyter Notebook
如果你想一步一步地学习其原理并可视化交互，可以用 VS Code 或 Jupyter 原生打开 `L5.ipynb` 文件，依次执行单元格。

## 项目结构说明
- `main.py` - 主执行入口脚本，包含核心逻辑和完整的场景演示流程。
- `L5.ipynb` - 本项目的交互式 Jupyter Notebook 版本（作为学习备份）。
- `agent/` - 包含 LangGraph 的状态定义、节点 (nodes) 逻辑和工具路由配置。
- `cache/` - 语义缓存的核心工具封装和配置。
- `data/` - 包含用于 FAQ 向量检索知识库的初始测试数据集 (CSV等)。
