import re  # 导入正则表达式库（虽然在此脚本中未直接调用，可能为后续扩展预留）
from pathlib import Path  # 导入路径处理库，用于跨平台文件路径操作

# 导入 Redis 向量库工具：用于将文本转换为向量（Embedding）
from redisvl.utils.vectorize import HFTextVectorizer
# 导入 LangChain 文本切分器：专门用于处理 Markdown 和递归切分文本
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 导入自定义内部模块：用于创建知识库索引和结构化切分 Markdown
from knowledge.indexer import create_knowledge_base_from_texts, _split_markdown_into_structured_chunks

# --- 配置原始文档路径 ---
# 使用 pathlib 动态获取路径：当前文件所在目录的父目录的父目录下的 data/raw_docs.md
# 这种写法保证了代码在不同环境下运行都能准确找到 data 文件夹
RAW_DOCS_MD_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_docs.md"

def init_app_knowledge_base():
    """
    初始化并构建应用程序的知识库。
    
    步骤：
    1. 加载中文字符向量模型。
    2. 读取本地 Markdown 格式的原始文档。
    3. 将 Markdown 转换为带有层级结构的文本块（Chunks）。
    4. 将文本块存入 Redis 向量数据库并建立索引。
    
    Returns:
        tuple: (kb_index, embeddings) 
               - kb_index: Redis 向量索引对象，用于后续的检索操作。
               - embeddings: 向量化模型实例，用于将查询词转换为向量。
    """
    # 1. 初始化向量模型
    # 使用 BAAI (北京智源) 的 bge-large-zh-v1.5 模型，这是目前中文效果极佳的开源向量化模型
    embeddings = HFTextVectorizer(model="BAAI/bge-large-zh-v1.5")
    
    # 2. 读取原始 Markdown 文档内容
    # 确保使用 utf-8 编码读取中文
    markdown_text = RAW_DOCS_MD_PATH.read_text(encoding="utf-8")
    
    # 3. 将 Markdown 文本切分为结构化的块
    # 调用自定义函数，它会根据 Markdown 的标题层级（# ## ###）来切分，保持内容的逻辑相关性
    raw_docs = _split_markdown_into_structured_chunks(markdown_text)

    # 4. 创建知识库并将数据持久化到 Redis
    # 返回值中的前两个变量通常是内部 ID，此处用下划线 _ 忽略，只保留 kb_index
    _, _, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,                          # 经过切分后的结构化文档块
        source_id="customer_support_docs",       # 给这份数据打上标签，标识为“客户支持文档”
        redis_url="redis://localhost:6379",      # Redis 服务器地址

        # 重要配置：skip_chunking=True
        # 已经在文本切块阶段做了分层结构化，入库时不需要再切块了，否则会破坏原有的层级信息。
        skip_chunking=True,
    )

    # --- 核心概念总结 ---
    
    # embeddings：
    # 作用：它是一个“翻译器”，将人类能理解的文字（如“如何退货？”）
    # 转换成计算机和数学算法能计算的数字向量（即 Embedding 向量）。

    # kb_index：
    # 作用：它是一个“智能管理员”或“高速检索器”。
    # 它负责管理 Redis 中海量的向量数据，当收到一个查询向量时，
    # 它能通过向量相似度计算（如余弦相似度），快速从数据库中找出语义最匹配的原始文本。
    
    return kb_index, embeddings