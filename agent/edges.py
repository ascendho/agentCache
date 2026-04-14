import logging
from typing import Literal

logger = logging.getLogger("agentic-workflow")

def cache_router(state) -> Literal["synthesize_response", "research"]:
    query = state["query"]
    cache_hit = state.get("cache_hit", False)
    
    if cache_hit:
        logger.info(f"👉 路由: 缓存命中，跳过研究节点 -> '{query[:20]}...'")
        return "synthesize_response"
    else:
        logger.info(f"👉 路由: 未命中缓存，开始研究 -> '{query[:20]}...'")
        return "research"

def research_quality_router(state) -> Literal["synthesize_response", "research"]:
    query = state["query"]
    score = state.get("research_quality_score", 0.0)
    iterations = state.get("research_iterations", 1)
    max_iterations = state.get("max_research_iterations", 1)
    
    if score >= 0.7:
        logger.info(f"👉 路由: 质量达标，进入内容合成 ({score:.2f}) -> '{query[:20]}...'")
        return "synthesize_response"
    elif iterations >= max_iterations:
        logger.info(f"👉 路由: 达到最大研究次数 ({iterations})，不论质量是否达标，强行合成 -> '{query[:20]}...'")
        return "synthesize_response"
    else:
        logger.info(f"👉 路由: 研究质量不达标，进入新一轮研究 ({iterations}/{max_iterations}) -> '{query[:20]}...'")
        return "research"
