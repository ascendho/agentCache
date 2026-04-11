import logging

from cache.config import config
from cache.cross_encoder import CrossEncoder
from cache.faq_data_container import FAQDataContainer
from cache.fuzzy_cache import FuzzyCache
from cache.llm_evaluator import LLMEvaluator
from cache.wrapper import SemanticCacheWrapper

logger = logging.getLogger("agentic-workflow")


def setup_semantic_cache():
    """初始化语义缓存，并使用 FAQ 数据进行预热。"""
    cache = SemanticCacheWrapper.from_config(config)
    data = FAQDataContainer()
    cache.hydrate_from_df(data.faq_df, clear=True)

    # 第一层：模糊匹配（最便宜，先执行）
    if config.get("enable_fuzzy_cache", True):
        fuzzy_cache = FuzzyCache()
        fuzzy_cache.hydrate_from_df(data.faq_df, clear=True)
        cache.register_fuzzy_cache(
            fuzzy_cache,
            distance_threshold=float(config.get("fuzzy_distance_threshold", 0.15)),
        )
        logger.info("已启用 Fuzzy 缓存层")
    else:
        cache.clear_fuzzy_cache()
        logger.info("已关闭 Fuzzy 缓存层")

    # 第二/三层：Reranker（先 CrossEncoder，再 LLM）
    rerank_steps = []

    if config.get("enable_cross_encoder_reranker", True):
        cross_encoder = CrossEncoder(
            model_name_or_path=config.get(
                "cross_encoder_model",
                "Alibaba-NLP/gte-reranker-modernbert-base",
            )
        )
        rerank_steps.append(cross_encoder.create_reranker())
        logger.info("已启用 CrossEncoder Reranker")
    else:
        logger.info("已关闭 CrossEncoder Reranker")

    if config.get("enable_llm_reranker", True):
        llm_evaluator = LLMEvaluator.construct_with_ark(
            model=config.get("llm_reranker_model", "doubao-seed-lite")
        )
        llm_reranker = llm_evaluator.create_reranker(
            batch_size=int(config.get("llm_reranker_batch_size", 5))
        )
        llm_top_k = int(config.get("llm_reranker_top_k", 5))

        def _llm_step(query, candidates):
            selected = candidates[:llm_top_k] if llm_top_k > 0 else candidates
            return llm_reranker(query, selected)

        rerank_steps.append(_llm_step)
        logger.info("已启用 LLM Reranker")
    else:
        logger.info("已关闭 LLM Reranker")

    if rerank_steps:
        def _chained_reranker(query, candidates):
            ranked = candidates
            for step in rerank_steps:
                if not ranked:
                    break
                ranked = step(query, ranked)
            return ranked

        cache.register_reranker(_chained_reranker)
    else:
        cache.clear_reranker()

    return cache
