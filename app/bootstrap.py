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


def create_knowledge_base():
    """构建演示用知识库并返回索引与向量模型。"""
    embeddings = HFTextVectorizer(model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    markdown_text = RAW_DOCS_MD_PATH.read_text(encoding="utf-8")
    raw_docs = _split_markdown_into_structured_chunks(markdown_text)

    _, _, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,
        source_id="customer_support_docs",
        redis_url="redis://localhost:6379",
        skip_chunking=True,
    )
    return kb_index, embeddings
