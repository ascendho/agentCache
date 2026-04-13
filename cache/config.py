import getpass
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 注：Python 原生的 os.getenv() 不会自动去读取 .env 文件。
# 加载 .env 文件，将其中定义的变量注入到环境变量中（不指定具体路径，依靠 “自动寻路”/向上冒泡 机制）
load_dotenv()

def load_ark_key():
    """
    从环境变量加载 ARK API 密钥，如果不存在则提示用户输入。
    注：ARK API 通常用于访问火山引擎（字节跳动）提供的 LLM 服务。
    """
    if not os.getenv("ARK_API_KEY"):
        # 如果环境变量中没有该 Key，则在终端安全地提示用户输入（输入内容不可见）
        api_key = getpass.getpass("请输入你的 ARK API key: ")
        os.environ["ARK_API_KEY"] = api_key
    else:
        # 如果已经存在，则提示已加载
        print("> ARK API key 已经加载到环境变量中")


# 系统全局配置字典
# 这些参数决定了 Redis 的连接方式以及语义缓存的敏感度
config = dict(
    # Redis 服务器连接地址，默认为本地 6379 端口
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    
    # 语义缓存的索引名称
    # 相当于在 Redis 里建了一张名为 semantic-cache 的表或命名空间。
    # 如果你以后还要开发一个“内部员工HR助手”，可以把那个命名叫 hr-cache，
    # 这样不同业务的缓存数据就不会互相串门、互相污染。
    cache_name=os.getenv("CACHE_NAME", "semantic-cache"),
    
    # 距离阈值：用于判断两个问题的语义是否足够接近以命中缓存
    # 数值越小要求越严苛（0.0 表示完全一致，1.0 表示差异很大）
    # 注：距离越小越相似，0.1-0.2 比较合理
    distance_threshold=float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.2")),
    
    # 缓存有效期（秒）：默认为 3600 秒（1小时），
    # 保证长尾问题（出现频率极低、个性化极强、表述极其复杂，但种类加起来却无穷无尽）答案的时效性。
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
)