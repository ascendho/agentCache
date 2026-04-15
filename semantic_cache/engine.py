# 导入 Python 标准库及第三方核心组件
from typing import Dict, List, Optional      # 类型标注：用于静态代码检查，提高可维护性
import pandas as pd                        # 数据处理：用于加载 FAQ 等表格数据
import redis                               # Redis 驱动：底层存储连接
from pydantic import BaseModel              # 声明式数据验证：确保缓存返回的数据结构规范
from redisvl.extensions.cache.embeddings import EmbeddingsCache  # 向量特征缓存：避免重复对相同文本进行嵌入计算
from redisvl.extensions.cache.llm import SemanticCache          # 语义缓存：基于向量相似度实现 LLM 问答对存储
from redisvl.utils.vectorize import HFTextVectorizer             # 向量化器：将文本转换为数学向量（模型驱动）

# ==========================================
# 数据模型定义 (基于 Pydantic)
# 作用：定义严格的 Schema，确保代码在传递数据时拥有明确的字段语义和类型检查
# ==========================================

class CacheResult(BaseModel):
    """
    单条缓存命中结果的数据模型。
    当缓存引擎找到匹配的问答对时，会将 Redis 中的原始数据封装为此对象。
    """
    prompt: str                   # 缓存库中存储的原始“提问/Prompt”（即 key）
    response: str                 # 对应的“标准答案/Response”（即 value）
    vector_distance: float        # 数学意义上的空间距离：数值越小，代表两个向量在空间中越靠近（语义越像）
    cosine_similarity: float      # 人类直观意义上的相似度：0.0 到 1.0 之间，1.0 代表完全一致
    seed_id: Optional[int] = None # 溯源标记：如果该数据来自 FAQ 文档，记录其在原始表格中的行 ID 或主键

class CacheResults(BaseModel):
    """
    一次检索请求返回的聚合结果对象。
    封装了用户的原始问题以及系统找到的所有可能匹配项。
    """
    query: str                  # 用户当前实时输入的提问文本
    matches: List[CacheResult]  # 匹配结果列表：若未命中则返回空列表 []

    def __repr__(self):
        """
        重写对象的字符串表示方法。
        目的：在日志或 Debug 控制台中直接打印出“问题 -> 命中了哪些缓存项”的直观摘要。
        """
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"

# ==========================================
# 工具函数
# ==========================================

def try_connect_to_redis(redis_url: str):
    """
    Redis 连通性健康检查工具。
    
    设计意图：
    由于语义缓存高度依赖向量数据库，如果在智能体启动初期无法连接 Redis，
    程序应该立即报错并停止，而不是在运行时产生不可预期的异常。
    
    Args:
        redis_url: Redis 格式的连接串，如 redis://localhost:6379
    """
    try:
        # 创建 Redis 客户端实例
        r = redis.Redis.from_url(redis_url)
        # 发送 PING 指令，如果 Redis 没响应或连接拒绝，这里会抛出异常
        r.ping() 
        print("✅ Redis 正在运行且可访问!")
        return r
    except redis.ConnectionError:
        # 连接失败时的友好提示
        print(f"❌ 无法连接到 Redis ({redis_url})。请确保 Redis 服务已启动。")
        raise

# ==========================================
# 核心类：语义缓存包装器 (SemanticCacheWrapper)
# 封装了底层 redisvl 的复杂逻辑，为上层业务提供“一键式”缓存检查与预热接口
# ==========================================

class SemanticCacheWrapper:
    # 这些参数是不是又重复在 config.py 里了？感觉有点冗余了，或者说 config.py 里就直接写死好了，毕竟现在也没什么变化的需求
    def __init__(self, name: str = "semantic-cache", distance_threshold: float = 0.3, ttl: int = 3600, redis_url: Optional[str] = None):
        """
        初始化语义缓存引擎。
        
        💡 核心架构：双重缓存机制 (Dual Cache Strategy)
        1. 第一层：EmbeddingsCache (向量特征缓存)
           - 作用：如果两个用户问了同样的话，不用再次调用 BGE 模型去计算向量，直接从 Redis 拿向量结果。
           - 收益：大幅减少 CPU/GPU 开销，降低响应延迟。
        2. 第二层：SemanticCache (语义问答缓存)
           - 作用：存储“问题向量 -> 答案文本”。
           - 收益：实现“意思相近即命中”，绕过大模型生成，直接秒回答案。
        """
        # 确定 Redis 连接地址，默认为本地
        # 又重复了！双重缓存什么意思，没看懂呢
        redis_conn_url = redis_url or "redis://localhost:6379"
        self.redis = try_connect_to_redis(redis_conn_url)
        
        # --- 初始化第 1 层：EmbeddingsCache ---
        # 存活时间设为语义缓存的 24 倍（向量计算结果较稳定，建议长期保存）

        # ttl 有必要吗？因为我们每次运行 main.py 的时候都会清空缓存
        self.embeddings_cache = EmbeddingsCache(redis_client=self.redis, ttl=ttl * 24)
        
        # --- 初始化向量模型加载器 ---
        # 指定使用 BAAI 的中文向量模型。注意：此处的 cache 参数挂载了上面的向量缓存。
        self.langcache_embed = HFTextVectorizer(model="BAAI/bge-large-zh-v1.5", cache=self.embeddings_cache)
        
        # --- 初始化第 2 层：SemanticCache ---
        self.cache = SemanticCache(
            name=name,                              # 索引名称，区分不同业务逻辑
            vectorizer=self.langcache_embed,        # 指定文本转向量的驱动器
            redis_client=self.redis,                # 数据库连接
            distance_threshold=distance_threshold,  # 判定阈值：越小代表越严格（0.1 代表基本要一字不差）
            ttl=ttl                                 # 缓存自动过期时间
        )
        
        # 内部状态：用于在当前生命周期内快速回溯问题与原始数据 ID 的关系
        self._seed_id_by_question: Dict[str, int] = {}

    # 此函数感觉有点冗余？意义在哪里？
    @classmethod
    def from_config(cls, config) -> "SemanticCacheWrapper":
        """
        工厂方法：支持通过标准 Python 字典对象进行初始化。
        便于与系统的 config.py 或 .env 环境变量配置方案集成。
        """
        return cls(
            redis_url=config.get("redis_url", "redis://localhost:6379"),
            name=config.get("cache_name", "semantic-cache"),
            distance_threshold=float(config.get("distance_threshold", 0.3)),
            ttl=int(config.get("ttl_seconds", 3600)),
        )

    def hydrate_from_df(self, df: pd.DataFrame, q_col: str = "question", a_col: str = "answer", clear: bool = True, ttl_override: Optional[int] = None, return_id_map: bool = False) -> Optional[Dict[str, int]]:
        """
        缓存预热（数据灌注）：将现有的 FAQ 知识批量导入语义缓存。
        
        Args:
            df: 包含问答对的 DataFrame。
            q_col: 问题列的列名。
            a_col: 答案列的列名。
            clear: 是否在导入前清空该索引下的所有旧缓存。
            ttl_override: 预热数据的有效期。如果为 None，则继承初始化时的默认 TTL。
            return_id_map: 是否返回一个“问题 -> ID”的映射字典。
        """
        # 这个预处理是不是不应该写在此函数里面？感觉职责有点不单一了，或者说这个函数就专门负责灌注，至于要不要清空缓存应该由调用方决定，这样更灵活一些
        # 1. 预处理：根据需要清理旧数据，防止新旧逻辑冲突
        if clear:
            print("正在彻底清空旧语义缓存数据与向量特征缓存...")
            self.cache.clear()            # 清理问答索引
            self.embeddings_cache.clear()  # 清理向量转换结果
            
        self._seed_id_by_question = {}
        question_to_id: Dict[str, int] = {}
        
        # 智能识别 ID 列：优先使用表格原有的 'id'，否则使用行索引
        id_col = "id" if "id" in df.columns else None
        
        # 2. 批量灌注逻辑
        # 使用 to_dict(orient="records") 将表格转为字典列表，方便循环处理
        for idx, row in enumerate(df.to_dict(orient="records")):
            q, a = row[q_col], row[a_col]
            
            # 将“标准问答对”写入 Redis。底层会自动调用 BGE 模型进行向量化。
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            
            # 3. 记录溯源关系
            # 获取 ID：优先取列值，取不到则取索引
            seed_id = int(row[id_col]) if id_col and row.get(id_col) is not None else idx
            # 在内存中维护一份对照表，方便 check 时找回 seed_id
            self._seed_id_by_question[str(q)] = seed_id
            
            if return_id_map and q not in question_to_id:
                question_to_id[q] = seed_id
                
        return question_to_id if return_id_map else None

    def check(self, query: str, distance_threshold: Optional[float] = None, num_results: int = 1) -> CacheResults:
        """
        语义检索：检查用户当前的提问是否命中了缓存。
        
        Args:
            query: 用户输入的字符串。
            distance_threshold: 临时指定的阈值（可覆盖全局配置）。
            num_results: 希望返回的结果数量（通常只取最相似的一个）。
            
        Returns:
            CacheResults: 一个包含命中的结构化数据的对象。
        """
        # 1. 执行向量相似度搜索
        # 底层逻辑：先将 query 转为向量，然后在 Redis 中进行 HNSW 近似最近邻搜索
        candidates = self.cache.check(query, distance_threshold=distance_threshold, num_results=num_results)
        
        # 2. 处理“未命中”情况 (Cache Miss)
        if not candidates:
            # 返回一个 matches 为空的空壳对象，告知上层应用：没找到，请去呼叫大模型。
            return CacheResults(query=query, matches=[])
            
        # 3. 处理“命中”情况 (Cache Hit) 并进行数据清洗
        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            # item 通常是一个包含 prompt, response, vector_distance 的字典
            result = dict(item)
            
            # --- 关键：数学计算与数据增强 ---
            # 提取向量距离：代表不相似度
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            
            # 距离(Distance)到相似度(Similarity)的转换逻辑：
            # 在余弦距离度量下：Similarity = 1 - (Distance / 2) [前提是向量已做L2归一化]
            # 这里采用该公式将“越小越好”的距离，转化为“越大越好”的百分比相似度。
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            
            # 记录原始查询词
            result["query"] = query
            
            # 尝试通过 query 找回预热时记录的原始 ID (用于前端展示或日志审计)
            result["seed_id"] = self._seed_id_by_question.get(str(result.get("prompt", "")))
            
            # 将清洗后的数据实例化为 Pydantic 数据模型
            results.append(CacheResult(**result))
            
        # 返回最终的结构化结果集
        return CacheResults(query=query, matches=results)