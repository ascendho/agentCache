import os  # 导入操作系统接口模块，用于获取系统环境变量

from dotenv import load_dotenv  # 导入 dotenv 库，用于从 .env 文件中加载环境变量

# --- 环境变量加载逻辑 ---
# 注：Python 原生的 os.getenv() 不会自动去读取磁盘上的 .env 文件。
# load_dotenv() 会在当前目录或向上级目录递归查找 .env 文件，
# 将其中定义的变量（如 REDIS_URL=xxx）注入到系统的 os.environ 中，
# 这样后续代码才能通过 os.getenv() 拿到这些配置。
load_dotenv()

# --- 系统全局配置字典 ---
# centralize_config: 将离散的环境变量汇聚成一个字典，方便在程序各处（如初始化 Redis 或 缓存引擎时）统一调用。
config = dict(
    # 1. Redis 连接配置
    # 格式通常为: redis://[[username]:[password]]@host:port/db
    # 如果环境变量中没设置 REDIS_URL，则默认连接本地默认端口。

    # redis_url 在 builder.py 里面也存在，都考虑用 env 读取吧，或者统一提前写入？减少冗余代码
    redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
    
    # 2. 语义缓存的逻辑命名空间
    # 作用：它定义了在 Redis 中存储向量索引的名称。
    # 场景：如果你以后还要开发一个“内部员工 HR 助手”，可以把那个命名叫 hr-cache，
    # 这样不同业务的缓存数据就不会互相串门、互相污染（即实现逻辑隔离）。
    cache_name=os.getenv("CACHE_NAME", "semantic-cache"),
    
    # 3. 语义匹配的严苛程度（核心参数）
    # os.getenv 拿到的是字符串，必须显式转换为 float 类型。
    # 原理：基于余弦距离（Cosine Distance）。
    # - 数值越接近 0：要求语义必须“几乎一模一样”才能命中。
    # - 数值越大（如 0.5）：语义稍微有点擦边就会命中缓存。
    # 经验值：0.1-0.2 通常是生产环境中平衡“准确率”与“命中率”的最佳区间。
    distance_threshold=float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.2")),
    
    # 4. 数据的生命周期管理（TTL）
    # os.getenv 拿到的是字符串，必须显式转换为 int 类型。
    # 单位：秒。默认为 3600 秒（1小时）。
    # 意义：
    # a) 节省 Redis 内存，自动清理老旧、低频的缓存数据。
    # b) 保证时效性：防止 FAQ 文档更新后，缓存里还存着一个小时前的旧答案。

    # 这个配置有必要嘛？因为我们每次运行 main.py 的时候都会清空缓存 
    ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
)