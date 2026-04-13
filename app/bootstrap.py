import re
from pathlib import Path

from redisvl.utils.vectorize import HFTextVectorizer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from agent import create_knowledge_base_from_texts


RAW_DOCS_MD_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_docs.md"


def _split_markdown_into_structured_chunks(markdown_text: str):
    """按 Markdown 层级切块，并对超长段落进行递归兜底切块。"""
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True,
    )
    header_docs = header_splitter.split_text(markdown_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    structured_chunks = []
    for doc in header_docs:
        child_docs = recursive_splitter.split_documents([doc])
        for child in child_docs:
            content = child.page_content.strip()
            if not content:
                continue
            metadata = {
                "header_1": child.metadata.get("header_1", ""),
                "header_2": child.metadata.get("header_2", ""),
                "header_3": child.metadata.get("header_3", ""),
                "is_announcement": bool(
                    re.search(r"(?m)^\s*>", content)
                    or "最新系统公告" in content
                    or "黑五" in content and "补偿" in content
                ),
            }
            structured_chunks.append({"content": content, "metadata": metadata})
    return structured_chunks


def init_app_knowledge_base():
    """构建演示用知识库并返回索引与向量模型。"""
    embeddings = HFTextVectorizer(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    markdown_text = RAW_DOCS_MD_PATH.read_text(encoding="utf-8")
    raw_docs = _split_markdown_into_structured_chunks(markdown_text)

    _, _, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,
        source_id="customer_support_docs",
        redis_url="redis://localhost:6379",

        # 已经在文本切块阶段做了分层结构化，入库时不需要再切块了，否则会破坏原有的层级信息。
        skip_chunking=True,
    )

    # embeddings 是一个加载了神经网络权重（MiniLM 模型）的 Python 对象。
    # 作用：它提供了一个 embed_query() 方法，可以将输入文本转换为向量（数字数组），以便在 Redis 向量数据库中进行相似度搜索。

    # kb_index 是一个“检索器对象”或者说是“向量数据库的索引连接器”。
    # 它脑子里记住了整个 Redis 数据库中所有文档片段（Chunks）的位置。
    # 作用：你给它一串数字（向量），它跑到 Redis 里算出最接近的几个文本块，然后原封不动地拿回来给你。
    return kb_index, embeddings
