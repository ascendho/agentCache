from langgraph.graph import StateGraph, END
from .nodes import (
    WorkflowState,
    initialize_nodes,
    decompose_query_node,
    check_cache_node,
    research_node,
    evaluate_quality_node,
    synthesize_response_node,
)
from .edges import (
    initialize_edges,
    route_after_cache_check,
    route_after_quality_evaluation,
)
from .tools import initialize_tools

def initialize_agent(semantic_cache, knowledge_base_index, openai_embeddings):
    """
    初始化 Agent 运行所需的全部依赖组件。

    包括：节点（Nodes）、路由逻辑（Edges）以及检索工具（Tools）。
    
    Args:
        semantic_cache: 语义缓存实例。
        knowledge_base_index: Redis 向量索引名称。
        openai_embeddings: 向量嵌入模型实例。
    """
    # 将依赖注入到各个模块的全局变量或配置中
    initialize_nodes(semantic_cache)
    initialize_edges(semantic_cache)
    initialize_tools(knowledge_base_index, openai_embeddings)

def build_workflow(cache, kb_index, embeddings):
    """
    构建并编译 LangGraph 工作流。

    该函数定义了状态机的拓扑结构，即数据如何在各个处理节点之间流转。
    流程图预览：
    [开始] -> 问题拆解 -> 缓存检查 
               |           |
               |      (未命中) -> 研究/检索 -> 质量评估 --(不合格)--> [返回研究]
               |           |          |         |
               |      (全命中) --------+---(合格)-+-> 综合输出 -> [结束]
    """
    # 1. 确保所有基础组件已就绪
    initialize_agent(cache, kb_index, embeddings)
    
    # 2. 初始化状态图，并声明状态结构为 WorkflowState
    workflow = StateGraph(WorkflowState)

    # 3. 注册工作流中的处理节点
    # 每个节点对应 nodes.py 中的一个函数，负责处理特定的业务逻辑
    workflow.add_node("decompose_query", decompose_query_node)      # 节点：拆解复杂问题为子问题
    workflow.add_node("check_cache", check_cache_node)              # 节点：检查语义缓存
    workflow.add_node("research", research_node)                    # 节点：从向量知识库中检索并分析
    workflow.add_node("evaluate_quality", evaluate_quality_node)    # 节点：评估研究结果的质量
    workflow.add_node("synthesize", synthesize_response_node)       # 节点：汇总所有子答案生成最终回复

    # 4. 指定工作流的起始入口
    workflow.set_entry_point("decompose_query")

    # 5. 构建节点之间的边（Edges）和条件路由
    
    # 线性连接：拆解完成后立即检查缓存
    workflow.add_edge("decompose_query", "check_cache")
    
    # 条件分支：根据缓存命中的结果决定去向
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache_check, # 路由逻辑函数
        {
            "research": "research",      # 如果有缓存未命中，跳转到研究节点
            "synthesize": "synthesize",  # 如果全部命中，直接跳转到最后的综合节点
        },
    )
    
    # 线性连接：研究完成后进入质量评估阶段
    workflow.add_edge("research", "evaluate_quality")
    
    # 条件分支：根据研究结果的质量决定是否需要补充研究
    workflow.add_conditional_edges(
        "evaluate_quality",
        route_after_quality_evaluation, # 路由逻辑函数
        {
            "research": "research",      # 质量不佳且未达迭代上限，返回研究节点（形成循环）
            "synthesize": "synthesize",  # 质量合格或已达最大迭代，进入综合节点
        },
    )
    
    # 最终连接：生成回复后结束工作流
    workflow.add_edge("synthesize", END)

    # 6. 编译图结构，将其转化为可直接调用的应用对象
    return workflow.compile()