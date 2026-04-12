"""
Routing and edge logic for the deep research agent workflow.
深度研究 Agent 工作流的路由和边（Edge）逻辑。

This module contains the conditional routing functions that determine
the flow of execution through the agentic workflow graph.
此模块包含条件路由函数，用于决定 Agent 工作流图中的执行流向。
"""

import logging

from typing import Literal, Dict, Any

# 获取日志记录器
logger = logging.getLogger("agentic-workflow")

# 全局变量，用于存储语义缓存实例
cache = None


def initialize_edges(semantic_cache):
    """
    初始化具有所需依赖项的边（Edge）模块。

    Args:
        semantic_cache: The semantic cache instance (语义缓存实例)
    """
    global cache
    cache = semantic_cache


def route_after_cache_check(state: Dict[str, Any]) -> Literal["research", "synthesize"]:
    """
    Intelligent routing based on cache results.
    基于缓存结果的智能路由。

    Determines whether we need to route to the researcher agent
    (if cache misses exist) or can proceed directly to synthesis
    (if all sub-questions were cached).
    确定是需要路由到研究员 Agent（如果存在缓存未命中），还是直接进行内容综合（如果所有子问题都已缓存）。
    
    返回逻辑：
    - "research": 只要有一个子问题没被缓存命中，就需要启动检索/研究流程。
    - "synthesize": 所有子问题都从缓存拿到了答案，直接进行最终汇总。
    """
    # 从状态中获取子问题的缓存命中情况
    cache_hits = state.get("cache_hits", {})

    # 筛选出所有未命中的子问题
    cache_misses = [sq for sq, hit in cache_hits.items() if not hit]

    if cache_misses:
        logger.info(
            f"🔀 路由到研究节点：检测到 {len(cache_misses)} 个缓存未命中"
        )
        for miss in cache_misses:
            logger.info(f"   🔍 待研究：'{miss[:50]}...'")
        return "research"
    else:
        # 如果命中了所有缓存，则直接跳过研究阶段
        logger.info("🔀 路由到汇总节点：所有子问题均已命中缓存")
        return "synthesize"


def route_after_quality_evaluation(
    state: Dict[str, Any],
) -> Literal["research", "synthesize"]:
    """
    Intelligent routing after quality evaluation - decide if more research is needed.
    质量评估后的智能路由 - 决定是否需要进行更多研究。

    Determines whether we need additional research iterations for inadequate answers
    or can proceed to synthesis with the current research quality.
    确定我们是需要由于回答不充分进行额外的研究迭代，还是可以基于当前的研究质量直接综合结果。
    """
    # 获取各个子问题的质量评分、当前迭代次数和最大允许迭代次数
    quality_scores = state.get("research_quality_scores", {})
    research_iterations = state.get("research_iterations", {})
    max_iterations = state.get("max_research_iterations", 2)

    needs_more_research = []
    for sub_question, score in quality_scores.items():
        current_iteration = research_iterations.get(sub_question, 0)
        # 质量评分阈值设为 0.7。如果分数过低且没超过最大迭代次数，则标记为需要重新研究
        if score < 0.7 and current_iteration < max_iterations:
            needs_more_research.append(sub_question)

    if needs_more_research:
        logger.info(
            f"🔄 路由到补充研究：{len(needs_more_research)} 个问题需要改进"
        )
        for sq in needs_more_research:
            iteration = research_iterations.get(sq, 0)
            score = quality_scores.get(sq, 0)
            logger.info(
                f"   🔍 待改进：'{sq[:40]}...' (分数: {score:.2f}, 轮次: {iteration + 1})"
            )
        # 返回 research 节点，形成一个循环（Loop）
        return "research"
    else:
        # 如果质量全部合格，或者达到了最大迭代次数上限
        logger.info("🔀 路由到汇总节点：研究质量已达标")
        # 在进入综合阶段前，将这些经过验证的高质量结果存入缓存
        cache_validated_research(state)
        return "synthesize"


def cache_validated_research(state: Dict[str, Any]):
    """
    Cache research results that have passed quality validation.
    This ensures we only cache high-quality, validated responses.
    缓存通过质量验证的研究结果。
    此操作能确保我们只缓存高质量且经验证的回复。
    """
    # 检查缓存组件是否已初始化
    if not cache:
        logger.warning("⚠️ 缓存未初始化，跳过研究结果写回")
        return

    # 检查是否全局禁用了缓存功能
    if not state.get("cache_enabled", True):
        return  # 禁用时跳过缓存

    cache_hits = state.get("cache_hits", {})
    sub_answers = state.get("sub_answers", {})
    quality_scores = state.get("research_quality_scores", {})

    cached_count = 0
    try:
        # 遍历所有子问题的答案
        for sub_question, answer in sub_answers.items():
            # 只有满足以下两个条件才存入缓存：
            # 1. 这个答案不是原本就从缓存里拿的（即本次是新研究出来的）
            # 2. 答案的质量评分必须大于等于 0.7（确保缓存池的纯净，不存垃圾内容）
            if (
                not cache_hits.get(sub_question, False)
                and quality_scores.get(sub_question, 0) >= 0.7
            ):
                # 调用语义缓存实例的 store 方法进行持久化
                cache.cache.store(prompt=sub_question, response=answer)
                cached_count += 1
                logger.info(
                    f"   💾 已写回高质量研究结果：'{sub_question[:40]}...'"
                )

        if cached_count > 0:
            logger.info(f"✅ 已写回 {cached_count} 条高质量研究结果")
    except Exception as e:
        logger.warning(f"⚠️ 研究结果写回失败: {e}")