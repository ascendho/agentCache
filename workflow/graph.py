import logging
from langgraph.graph import StateGraph, END

from workflow.nodes import (
    WorkflowState,
    initialize_nodes,
    check_cache_node,
    research_node,
    evaluate_quality_node,
    synthesize_response_node,
)
from workflow.edges import cache_router, research_quality_router
from workflow.tools import initialize_tools

def create_agent_graph(semantic_cache=None, kb_index=None, embeddings=None) -> StateGraph:
    """初始化并构建LangGraph计算图"""
    initialize_nodes(semantic_cache)
    if kb_index and embeddings:
        initialize_tools(kb_index, embeddings)

    workflow = StateGraph(WorkflowState)

    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("research", research_node)
    workflow.add_node("evaluate_quality", evaluate_quality_node)
    workflow.add_node("synthesize_response", synthesize_response_node)

    workflow.set_entry_point("check_cache")

    workflow.add_conditional_edges(
        "check_cache",
        cache_router,
        {
            "synthesize_response": "synthesize_response",
            "research": "research"
        }
    )

    workflow.add_edge("research", "evaluate_quality")

    workflow.add_conditional_edges(
        "evaluate_quality",
        research_quality_router,
        {
            "synthesize_response": "synthesize_response",
            "research": "research"
        }
    )

    workflow.add_edge("synthesize_response", END)

    logger = logging.getLogger("agentic-workflow")
    logger.info("LangGraph计算图已构建！")

    return workflow.compile()
