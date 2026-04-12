import getpass
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 加载当前目录下的 .env 文件，将其中定义的变量注入到环境变量中
load_dotenv()

def load_ark_key():
    """
    Load ARK API key from environment variables or prompt the user.
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
    # 注：距离越小越相似，0.1-0.2 比较合理
    distance_threshold=float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.2")),
    
    # 缓存有效期（秒）：默认为 3600 秒（1小时）
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
)