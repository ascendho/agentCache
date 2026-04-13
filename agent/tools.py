"""
深度研究工作流的工具模块。
该模块负责知识库语义检索的具体实现：
1) 将自然语言查询转为向量嵌入（Embedding）；
2) 在 Redis 向量索引中执行向量近邻搜索（Vector Search）；
3) 对结果进行处理，返回带相关度分数的格式化文本。
"""

import logging
import numpy as np
from typing import Any

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.query import VectorQuery

# 配置模块日志记录器，用于在控制台追踪工具的调用情况
logger = logging.getLogger("agentic-workflow")

# 全局依赖变量：在初始化阶段由主程序注入
# kb_index: RedisVL 的 SearchIndex 实例
# embeddings: 向量化模型实例
kb_index = None
embeddings = None


def initialize_tools(
    knowledge_base_index: Any, openai_embeddings: HFTextVectorizer
):
    """
    初始化工具模块所需的依赖项。
    由于 LangChain 的 @tool 装饰器通常定义为全局函数，
    我们通过此初始化函数动态注入当前请求所需的知识库索引和向量模型。

    Args:
        knowledge_base_index: 已连接到 Redis 的 SearchIndex 实例。
        openai_embeddings: 负责将文本转换为向量的模型实例。
    """
    global kb_index, embeddings
    kb_index = knowledge_base_index
    embeddings = openai_embeddings


@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """
    在 Redis 向量知识库中搜索与问题最相关的信息片段。
    这是供 Agent 调用的标准工具。它能够将查询语句转化为向量，
    并在索引中寻找语义最接近的文本块。

    Args:
        query: 需要在知识库中检索的关键词或描述性问题。
        top_k: 指定返回最相关的结果数量，默认返回 3 条。

    Returns:
        一段格式化的字符串，包含检索到的文本内容及其相关度分数（relevance）。
    """
    # 防御式检查：如果工具未被正确初始化，直接返回错误描述，
    # 这样 LLM 能够识别到错误并尝试采取其他行动，而不是让整个程序崩溃。
    if not kb_index or not embeddings:
        return "错误：知识库尚未初始化，请先调用 initialize_tools()。"

    logger.info(
        f"🔍 Using search_knowledge_base tool for query: '{query}' (top_k={top_k})"
    )

    try:
        # 1) 将用户输入的查询文本转化为向量（Embedding）
        query_vector = embeddings.embed(query)

        # 2) 构造 Redis 向量查询对象
        # vector_field_name 必须与 knowledge_base_utils.py 中定义的 schema 保持一致
        search_query = VectorQuery(
            vector=query_vector,
            vector_field_name="content_vector",
            return_fields=[
                "content",
                "vector_distance",
                "header_1",
                "header_2",
                "header_3",
                "is_announcement",
            ], # 返回原始文本、向量距离和结构化元信息
            num_results=top_k,
        )

        # 在 Redis 中执行查询
        results = kb_index.query(search_query)

        # 如果没有找到任何匹配项
        if not results:
            return f"未找到与该问题相关的信息：{query}"

        # 3) 将原始检索结果格式化为易于 LLM 理解的可读文本
        formatted_results = []
        for i, result in enumerate(results, 1):
            # 将向量距离转换为相关度分数。距离越小，分数越高（1.0 表示完全匹配）
            relevance = 1.0 - float(result["vector_distance"])
            header_parts = [
                str(result.get("header_1", "") or "").strip(),
                str(result.get("header_2", "") or "").strip(),
                str(result.get("header_3", "") or "").strip(),
            ]
            header_path = " > ".join([h for h in header_parts if h])
            is_announcement = str(result.get("is_announcement", "false")).lower() == "true"

            prefix_lines = []
            if header_path:
                prefix_lines.append(f"Section: {header_path}")
            if is_announcement:
                prefix_lines.append("Tag: ANNOUNCEMENT")
            prefix_block = "\n".join(prefix_lines)

            body = f"Result {i} (relevance: {relevance:.3f}):\n{result['content']}"
            formatted_results.append(
                f"{prefix_block}\n{body}" if prefix_block else body
            )

        # 返回合并后的字符串
        return "\n\n".join(formatted_results)

    except Exception as e:
        # 捕获检索过程中的任何异常（如 Redis 连接中断、索引不存在等）
        logger.error(f"Error during KB search: {e}")
        return f"知识库检索异常：{str(e)}"