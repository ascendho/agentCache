import getpass
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 加载当前目录下的 .env 文件，将其中定义的变量注入到环境变量中
load_dotenv()


def _to_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def load_ark_key():
    """
    从环境变量加载 ARK API 密钥，如果不存在则提示用户输入。

    注：ARK API 通常用于访问火山引擎（字节跳动）提供的 LLM 服务。
    """
    if not os.getenv("ARK_API_KEY"):
        # 如果环境变量中没有该 Key，则在终端安全地提示用户输入（输入内容不可见）
        api_key = getpass.getpass("Enter your ARK API key: ")
        os.environ["ARK_API_KEY"] = api_key
    else:
        # 如果已经存在，则提示已加载
        print("> ARK API key is already loaded in the environment")


# 系统全局配置字典
# 这些参数决定了 Redis 的连接方式以及语义缓存的敏感度
config = dict(
    # Redis 服务器连接地址，默认为本地 6379 端口
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    
    # 语义缓存的索引名称
    cache_name=os.getenv("CACHE_NAME", "semantic-cache"),
    
    # 距离阈值：用于判断两个问题的语义是否足够接近以命中缓存
    # 数值越小要求越严苛（0.0 表示完全一致，1.0 表示差异很大）
    distance_threshold=float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.3")),
    
    # 缓存有效期（秒）：默认为 3600 秒（1小时）
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),

    # 缓存链路开关（可独立启停，无需改主流程代码）
    enable_fuzzy_cache=_to_bool_env("ENABLE_FUZZY_CACHE", False),
    enable_cross_encoder_reranker=_to_bool_env("ENABLE_CROSS_ENCODER_RERANKER", False),
    enable_llm_reranker=_to_bool_env("ENABLE_LLM_RERANKER", False),

    # 各层参数
    fuzzy_distance_threshold=float(os.getenv("FUZZY_DISTANCE_THRESHOLD", "0.15")),
    cross_encoder_model=os.getenv(
        "CROSS_ENCODER_MODEL",
        "Alibaba-NLP/gte-reranker-modernbert-base",
    ),

    llm_reranker_model=os.getenv("LLM_RERANKER_MODEL", "doubao-seed-2-0-lite-260215"),
    llm_reranker_batch_size=int(os.getenv("LLM_RERANKER_BATCH_SIZE", "5")),
    llm_reranker_top_k=int(os.getenv("LLM_RERANKER_TOP_K", "5")),
)