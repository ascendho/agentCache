"""
Knowledge Base Utilities
知识库工具模块。

Simple utilities for creating Redis-based knowledge bases from text content.
提供从文本快速构建 Redis 向量知识库的简化能力。
"""

import logging
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple

import redis
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.index import SearchIndex

# 初始化日志记录器
logger = logging.getLogger("kb-utils")

class KnowledgeBaseManager:
    """管理基于 Redis 的知识库索引及其生命周期。"""
    
    def __init__(self, redis_client: redis.Redis, embeddings: Optional[HFTextVectorizer] = None):
        """
        初始化知识库管理器。
        
        Args:
            redis_client: Redis 客户端实例。
            embeddings: 向量化工具（如果为 None，则默认使用 HuggingFace 的轻量级模型）。
        """
        self.redis_client = redis_client
        # 默认使用 sentence-transformers 模型进行文本向量化
        self.embeddings = embeddings or HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
        # 用于跟踪当前活跃的索引（以 URL 的哈希值为键）
        self.active_indexes = {} 
    
    def create_knowledge_base(
        self, 
        source_id: str, 
        content, 
        chunk_size: int = 2500, 
        chunk_overlap: int = 250,
        skip_chunking: bool = False
    ) -> Tuple[bool, str, Optional[SearchIndex]]:
        """
        从输入内容创建知识库向量索引。
        
        Args:
            source_id: 内容来源标识符（如 URL 或自定义 ID）。
            content: 输入内容。可以是字符串（将自动切片）或字符串列表（直接使用）。
            chunk_size: 文本切片的大小（字符数）。
            chunk_overlap: 切片之间的重叠大小。
            skip_chunking: 如果为 True，则将整段字符串视为单个切片。
            
        Returns:
            元组 (是否成功, 提示消息, 搜索索引对象)
        """
        try:
            if not content:
                return False, "No content to process", None
            
            # 基于 source_id 生成 8 位 MD5 哈希作为索引名，确保 Redis 键名合法且不冲突
            source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
            index_name = f"kb-{source_hash}"
            
            # --- 步骤 1: 文本预处理与切片 ---
            if isinstance(content, list):
                # 如果已经是列表，则直接使用
                text_chunks = content
                source_type = "text_list"
                logger.info(f"Using provided list of {len(text_chunks)} text chunks")
            elif isinstance(content, str):
                if skip_chunking:
                    # 不切片，直接作为单条记录
                    text_chunks = [content]
                    source_type = "text_single"
                    logger.info("Using entire text as single chunk")
                else:
                    # 使用 LangChain 的递归字符切片器，尝试按段落、句子保留语境
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    
                    docs = [Document(page_content=content, metadata={"source_id": source_id, "source": "text"})]
                    doc_chunks = splitter.split_documents(docs)
                    text_chunks = [chunk.page_content for chunk in doc_chunks]
                    source_type = "text_chunked"
                    logger.info(f"Split text into {len(text_chunks)} chunks")
            else:
                return False, f"Unsupported content type: {type(content)}", None
            
            # --- 步骤 2: 定义 Redis 向量索引 Schema ---
            # 包含：文本内容、来源 ID、来源类型、切片索引以及 1536 维的向量字段
            schema = {
                "index": {"name": index_name, "prefix": f"kb:{source_hash}:"},
                "fields": [
                    {"name": "content", "type": "text"},          # 原始文本内容
                    {"name": "source_id", "type": "tag"},         # 来源标签
                    {"name": "source_type", "type": "tag"},       # 类型标签
                    {"name": "chunk_index", "type": "numeric"},   # 切片顺序编号
                    {
                        "name": "content_vector",
                        "type": "vector",
                        "attrs": {
                            "dims": 1536,                 # 向量维度（对应 OpenAI 或指定模型）
                            "distance_metric": "cosine",  # 距离算法：余弦相似度
                            "algorithm": "hnsw",          # 算法：HNSW (适合高维度、高性能检索)
                            "datatype": "float32",
                        },
                    },
                ],
            }
            
            # --- 步骤 3: 在 Redis 中创建并初始化索引 ---
            kb_index = SearchIndex.from_dict(schema, redis_client=self.redis_client)
            # overwrite=True 确保每次处理同一个 URL 时会覆盖旧的索引数据
            kb_index.create(overwrite=True)
            
            # --- 步骤 4: 生成嵌入向量（Embedding）并装载数据 ---
            payload = []
            for i, text_chunk in enumerate(text_chunks):
                try:
                    # 调用模型将文本转为向量字节流
                    embedding = self.embeddings.embed(text_chunk, as_buffer=True)
                    payload.append({
                        "content": text_chunk,
                        "source_id": source_id,
                        "source_type": source_type,
                        "chunk_index": i,
                        "content_vector": embedding,
                    })
                except Exception as e:
                    logger.warning(f"Failed to embed chunk {i}: {e}")
                    continue
            
            if not payload:
                return False, "Failed to create embeddings for content", None
            
            # 将封装好的负载批量写入 Redis
            kb_index.load(payload)
            
            # 记录到活跃索引字典中，方便管理
            self.active_indexes[source_hash] = {
                "index": kb_index,
                "source_id": source_id,
                "source_type": source_type,
                "chunks": len(text_chunks),
                "created_at": time.time()
            }
            
            success_msg = f"✅ Created knowledge base with {len(text_chunks)} chunks ({source_type})"
            logger.info(success_msg)
            return True, success_msg, kb_index
            
        except Exception as e:
            error_msg = f"❌ Knowledge base creation failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def get_index_for_source(self, source_id: str) -> Optional[SearchIndex]:
        """通过 source_id 获取对应的 Redis 搜索索引对象"""
        source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
        index_info = self.active_indexes.get(source_hash)
        return index_info["index"] if index_info else None
    
    def get_index_for_url(self, url: str) -> Optional[SearchIndex]:
        """通过 URL 获取索引（get_index_for_source 的别名，用于向后兼容）"""
        return self.get_index_for_source(url)
    
    def clear_knowledge_base(self, source_id: str = None) -> str:
        """
        清空知识库索引。
        
        Args:
            source_id: 指定要删除的 source_id。如果为 None，则清空所有已注册的索引。
            
        Returns:
            操作状态消息。
        """
        try:
            if source_id:
                # 删除特定的索引
                source_hash = hashlib.md5(source_id.encode()).hexdigest()[:8]
                if source_hash in self.active_indexes:
                    index_info = self.active_indexes[source_hash]
                    index_info["index"].drop() # 物理删除 Redis 中的索引和数据
                    del self.active_indexes[source_hash]
                    return f"✅ Cleared knowledge base for {source_id}"
                else:
                    return f"⚠️ No knowledge base found for {source_id}"
            else:
                # 遍历并清空所有索引
                cleared_count = 0
                for source_hash, index_info in list(self.active_indexes.items()):
                    try:
                        index_info["index"].drop()
                        cleared_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to clear index {source_hash}: {e}")
                
                self.active_indexes.clear()
                return f"✅ Cleared {cleared_count} knowledge bases"
                
        except Exception as e:
            error_msg = f"❌ Clear operation failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    def get_status(self) -> Dict[str, Any]:
        """获取所有当前活跃知识库的状态和元数据。"""
        status = {
            "total_indexes": len(self.active_indexes),
            "indexes": []
        }
        
        for source_hash, index_info in self.active_indexes.items():
            status["indexes"].append({
                "source_hash": source_hash,
                "source_id": index_info["source_id"],
                "source_type": index_info["source_type"],
                "chunks": index_info["chunks"],
                "created_at": index_info["created_at"],
                "age_seconds": time.time() - index_info["created_at"]
            })
        
        return status

def create_knowledge_base_from_texts(
    texts: List[str],
    source_id: str = "custom_texts",
    redis_url: str = "redis://localhost:6379",
    skip_chunking: bool = True
) -> Tuple[bool, str, Optional[SearchIndex]]:
    """
    便捷函数：直接从文本列表创建知识库。
    
    Args:
        texts: 文本字符串列表。
        source_id: 来源标识。
        redis_url: Redis 连接地址。
        skip_chunking: 是否跳过分块处理。
        
    Returns:
        元组 (是否成功, 提示消息, 搜索索引对象)
    """
    redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
    kb_manager = KnowledgeBaseManager(redis_client)
    return kb_manager.create_knowledge_base(source_id, texts, skip_chunking=skip_chunking)