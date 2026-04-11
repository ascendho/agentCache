"""
语义缓存包装器（简化版）。

该模块在 redisvl 的 SemanticCache 之上提供更易用的统一接口，核心能力包括：
1) 单条/批量查询语义缓存；
2) 将底层原始命中结果标准化为统一数据结构；
3) 支持可选 reranker（重排序器）对候选进行二次筛选与重排；
4) 支持从 DataFrame 或问答对批量预热缓存。

典型使用方式：

    cache = SemanticCacheWrapper()

    def my_reranker(query: str, candidates: List[dict]) -> List[dict]:
        filtered = [c for c in candidates if c.get("vector_distance", 1.0) < 0.5]
        return sorted(filtered, key=lambda x: len(x.get("response", "")))

    cache.register_reranker(my_reranker)
    results = cache.check("What is Python?")
    cache.clear_reranker()
"""

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import redis
from pydantic import BaseModel, Field
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from scipy.spatial.distance import cosine
from tqdm.auto import tqdm

from cache.config import config as default_config


class CacheResult(BaseModel):
    """
    统一的缓存命中结果结构。

    核心字段：
    - prompt: 命中的缓存键（问题文本）
    - response: 对应缓存答案
    - vector_distance: 向量距离（越小越相似）
    - cosine_similarity: 由距离换算得到的相似度（越大越相似）

    reranker 元数据字段：
    - reranker_type: 重排序器类型（如 cross_encoder / llm）
    - reranker_score: 重排序器原始分数
    - reranker_reason: 重排序器解释信息（多见于 LLM reranker）
    """

    prompt: str
    response: str
    vector_distance: float
    cosine_similarity: float

    reranker_type: Optional[str] = None
    reranker_score: Optional[float] = None
    reranker_reason: Optional[str] = None


class CacheResults(BaseModel):
    """一次查询对应的命中结果集合。"""

    query: str
    matches: List[CacheResult]

    def __repr__(self):
        return f"(Query: '{self.query}', Matches: {[m.prompt for m in self.matches]})"


def try_connect_to_redis(redis_url: str):
    """连接 Redis 并做可用性探测，失败时给出可执行提示。"""
    try:
        r = redis.Redis.from_url(redis_url)
        r.ping()
        print("✅ Redis is running and accessible!")
    except redis.ConnectionError:
        print(
            """
            ❌ Cannot connect to Redis. Please make sure Redis is running on localhost:6379
                Try: docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
            """
        )
        raise

    return r


class SemanticCacheWrapper:
    """
    语义缓存包装器主类。

    该类负责：
    - 初始化 Redis、Embedding 缓存与语义缓存实例；
    - 提供统一查询接口（check/check_many）；
    - 管理可选 reranker；
    - 支持缓存预热与重建。
    """

    def __init__(
        self,
        name: str = "semantic-cache",
        distance_threshold: float = 0.3,
        ttl: int = 3600,
        redis_url: Optional[str] = None,
    ):
        # 允许调用方显式传 redis_url；否则回退到配置文件或默认本地地址。
        redis_conn_url = redis_url or getattr(
            default_config, "redis_url", "redis://localhost:6379"
        )
        self.redis = try_connect_to_redis(redis_conn_url)

        # EmbeddingsCache 用于缓存向量化结果，降低重复 embed 的开销。
        self.embeddings_cache = EmbeddingsCache(redis_client=self.redis, ttl=ttl * 24)
        self.langcache_embed = HFTextVectorizer(
            model="redis/langcache-embed-v1", cache=self.embeddings_cache
        )

        # 语义缓存主体：基于向量相似度做命中判断。
        self.cache = SemanticCache(
            name=name,
            vectorizer=self.langcache_embed,
            redis_client=self.redis,
            distance_threshold=distance_threshold,
            ttl=ttl,
        )

        # reranker 是一个可插拔函数：输入候选列表，输出重排后的候选。
        self._reranker: Optional[Callable[[str, List[dict]], List[dict]]] = None

    def pair_distance(self, question: str, answer: str) -> float:
        """计算问题与答案文本之间的语义距离。"""
        q_emb = self.langcache_embed.embed(question)
        a_emb = self.langcache_embed.embed(answer)

        distance = cosine(q_emb, a_emb)
        return distance.item()

    def set_cache_entries(self, question_answer_pairs: List[Tuple[str, str]]):
        """用给定问答对重建缓存（先清空再写入）。"""
        self.cache.clear()
        for question, answer in question_answer_pairs:
            self.cache.store(prompt=question, response=answer)

    # ==========================
    # 构造与预热相关方法
    # ==========================
    @classmethod
    def from_config(
        cls,
        config,
    ) -> "SemanticCacheWrapper":
        """
        从配置对象创建包装器实例。把配置字典统一转成“可直接用的缓存对象”，避免到处手写参数。

        期望配置中包含：
        - redis_url
        - cache_name
        - distance_threshold
        - ttl_seconds
        """
        return cls(
            redis_url=config["redis_url"],
            name=config["cache_name"],
            distance_threshold=float(config["distance_threshold"]),
            ttl=int(config["ttl_seconds"]),
        )

    def hydrate_from_df(
        self,
        df: pd.DataFrame,
        *,
        q_col: str = "question",
        a_col: str = "answer",
        clear: bool = True,
        ttl_override: Optional[int] = None,
        return_id_map: bool = False,
    ) -> Optional[Dict[str, int]]:
        """
        从 DataFrame 批量预加载缓存。

        适用于 FAQ/历史问答预热场景；可选返回 question->id 映射，
        便于后续对照分析。
        """
        if clear:
            self.cache.clear()
        question_to_id: Dict[str, int] = {}
        idx = 0
        for row in df[[q_col, a_col]].itertuples(index=False, name=None):
            q, a = row
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            if return_id_map and q not in question_to_id:
                question_to_id[q] = idx
            idx += 1
        return question_to_id if return_id_map else None

    def hydrate_from_pairs(
        self,
        pairs: Iterable[Tuple[str, str]],
        *,
        clear: bool = True,
        ttl_override: Optional[int] = None,
        return_id_map: bool = False,
    ) -> Optional[Dict[str, int]]:
        """从 (question, answer) 迭代器批量写入缓存。"""
        if clear:
            self.cache.clear()
        question_to_id: Dict[str, int] = {}
        idx = 0
        for q, a in pairs:
            self.cache.store(prompt=q, response=a, ttl=ttl_override)
            if return_id_map and q not in question_to_id:
                question_to_id[q] = idx
            idx += 1
        return question_to_id if return_id_map else None

    # ==========================
    # reranker 管理方法
    # ==========================
    def register_reranker(self, reranker: Callable[[str, List[dict]], List[dict]]):
        """
        注册重排序函数。

        函数签名应为：
            reranker(query: str, candidates: List[dict]) -> List[dict]

        其中 candidates 为语义检索初筛候选，返回值应为“过滤或重排后”的候选列表。
        """
        if not callable(reranker):
            raise TypeError("Reranker must be a callable function")
        self._reranker = reranker

    def clear_reranker(self):
        """清除已注册 reranker，恢复默认语义检索排序。"""
        self._reranker = None

    def has_reranker(self) -> bool:
        """判断当前是否已注册 reranker。"""
        return self._reranker is not None

    # ==========================
    # 查询相关方法
    # ==========================
    def check(
        self,
        query: str,
        distance_threshold: Optional[float] = None,
        num_results: int = 1,
        use_reranker_distance: bool = False,
    ) -> List[CacheResult]:
        """
        查询单个问题在语义缓存中的命中结果。

        Args:
            query: 查询文本。
            distance_threshold: 向量距离阈值（越小越严格）。
            num_results: 最终返回结果条数上限。
            use_reranker_distance: 若为 True，使用 reranker_distance 覆盖原始向量距离。

        Returns:
            CacheResults：包含 query 与标准化后的 matches 列表。
        """

        # 若启用 reranker，需要扩大初始候选池，给 reranker 足够重排空间。
        _num_results = (
            num_results if not self.has_reranker() else max(10, 3 * num_results)
        )
        candidates = self.cache.check(
            query, distance_threshold=distance_threshold, num_results=_num_results
        )

        if not candidates:
            return CacheResults(query=query, matches=[])

        # 候选重排：由业务侧注入策略（交叉编码器、LLM 打分等）。
        if self.has_reranker():
            candidates = self._reranker(query, candidates)

        results: List[CacheResult] = []
        for item in candidates[:num_results]:
            result = dict(item)
            result["vector_distance"] = float(result.get("vector_distance", 0.0))
            # 将距离线性映射为相似度，便于展示与排序理解。
            result["cosine_similarity"] = float((2 - result["vector_distance"]) / 2)
            result["query"] = query

            if self.has_reranker():
                result["reranker_type"] = result.get("reranker_type")
                result["reranker_score"] = result.get("reranker_score")
                result["reranker_reason"] = result.get("reranker_reason")
                if use_reranker_distance:
                    # 某些 reranker 会给出更可信的距离/分数，可选择覆盖。
                    result["vector_distance"] = result["reranker_distance"]

            results.append(CacheResult(**result))

        return CacheResults(query=query, matches=results)

    def check_many(
        self,
        queries: List[str],
        distance_threshold: Optional[float] = None,
        show_progress: bool = False,
        num_results: int = 1,
        use_reranker_distance=False,
    ) -> List[Optional[CacheResult]]:
        """
        批量查询多个问题的缓存命中结果。

        Args:
            queries: 查询文本列表。
            distance_threshold: 距离阈值。
            show_progress: 是否显示进度条。
            num_results: 每个 query 返回条数上限。
            use_reranker_distance: 是否使用 reranker 距离覆盖原始距离。

        Returns:
            按输入顺序返回 CacheResults 列表。
        """
        results: List[Optional[CacheResult]] = []
        # 顺序遍历保证输出与输入一一对应，便于回填原数据集。
        for q in tqdm(queries, disable=not show_progress):
            cache_results = self.check(
                q, distance_threshold, num_results, use_reranker_distance
            )
            results.append(cache_results)
        return results
