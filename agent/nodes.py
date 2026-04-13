"""
深度研究 Agent 的工作流节点。
该模块包含用于实现带语义缓存的 Agent 工作流核心逻辑的所有节点函数。
"""

import time
import logging
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict, Tuple
import concurrent.futures

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from .tools import search_knowledge_base

# 获取日志记录器
logger = logging.getLogger("agentic-workflow")

# 全局变量：存储语义缓存实例
cache = None

# 全局变量：存储单例 LLM 实例，避免重复创建
_analysis_llm = None
_research_llm = None


def _extract_token_usage(response: Any) -> Tuple[int, int]:
    """从 LangChain 响应对象中尽力提取输入/输出 token 统计。"""
    if isinstance(response, dict):
        usage_meta = response.get("usage_metadata", {})
        response_meta = response.get("response_metadata", {})
    else:
        usage_meta = getattr(response, "usage_metadata", None) or {}
        response_meta = getattr(response, "response_metadata", None) or {}

    # 常见字段 1：AIMessage.usage_metadata = {input_tokens, output_tokens, total_tokens}
    in_tokens = int(usage_meta.get("input_tokens", 0) or 0)
    out_tokens = int(usage_meta.get("output_tokens", 0) or 0)

    if in_tokens > 0 or out_tokens > 0:
        return in_tokens, out_tokens

    # 常见字段 2：AIMessage.response_metadata.token_usage = {prompt_tokens, completion_tokens}
    token_usage = response_meta.get("token_usage", {}) if isinstance(response_meta, dict) else {}
    in_tokens = int(token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0)) or 0)
    out_tokens = int(token_usage.get("completion_tokens", token_usage.get("output_tokens", 0)) or 0)

    return in_tokens, out_tokens


def _append_llm_usage(
    state: "WorkflowState",
    model: str,
    provider: str,
    response: Any,
    node: str,
) -> List[Dict[str, Any]]:
    """追加一条 LLM 调用记录，供主流程后续成本评估使用。"""
    usage = list(state.get("llm_usage", []))
    in_tokens, out_tokens = _extract_token_usage(response)
    usage.append(
        {
            "model": model,
            "provider": provider,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "node": node,
            "timestamp": datetime.now().isoformat(),
        }
    )
    return usage


def get_analysis_llm():
    """获取配置的分析型 LLM 实例（通常使用推理能力更强的模型，如 DeepSeek-V3）。"""
    global _analysis_llm
    if _analysis_llm is None:
        _analysis_llm = ChatOpenAI(
            model="deepseek-v3-2-251201", 
            temperature=0.1, 
            max_tokens=400,
            api_key=os.getenv("ARK_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
    return _analysis_llm


def get_research_llm():
    """获取配置的研究型 LLM 实例（用于工具调用、检索结果分析）。"""
    global _research_llm
    if _research_llm is None:
        _research_llm = ChatOpenAI(
            model="ep-m-20260411093114-9hftc",
            temperature=0.2, 
            max_tokens=400,
            api_key=os.getenv("ARK_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
    return _research_llm

class WorkflowMetrics(TypedDict):
    """
    用于跟踪工作流性能的综合指标字典。
    
    对相关指标进行分组，结构更清晰并方便后续分析缓存效果与时延。
    """
    total_latency: float            # 总时延
    decomposition_latency: float    # 问题拆解耗时
    cache_latency: float            # 缓存查询耗时
    research_latency: float         # 研究检索耗时
    synthesis_latency: float        # 最终综合耗时
    
    cache_hit_rate: float           # 缓存命中率 (0.0 到 1.0)
    cache_hits_count: int           # 缓存命中个数
    
    questions_researched: int       # 实际进行研究的问题数
    total_research_iterations: int  # 总研究迭代次数
    
    llm_calls: Dict[str, int]       # 统计不同 LLM 模型的调用次数
    
    sub_question_count: int         # 拆解出的子问题总数
    execution_path: str             # 执行路径描述（例如 "decomposed → cache_checked"）


class WorkflowState(TypedDict):
    """
    带语义缓存和质量评估的 Agent 工作流状态定义。
    该字典贯穿整个 LangGraph 执行过程，存储所有中间结果。
    """
    original_query: str              # 用户的原始提问
    sub_questions: List[str]         # 拆解后的子问题列表
    sub_answers: Dict[str, str]      # 存储每个子问题的答案（可能来自缓存或研究）
    final_response: Optional[str]    # 最终汇总后的回答

    cache_hits: Dict[str, bool]      # 记录每个子问题是否命中缓存
    cache_confidences: Dict[str, float] # 记录缓存命中的置信度
    cache_seed_ids: Dict[str, Optional[int]] # 命中缓存时对应 FAQ 种子 id
    cache_enabled: bool              # 开关：是否启用缓存功能（用于 A/B 测试）

    research_iterations: Dict[str, int] # 记录每个子问题已进行的迭代次数
    max_research_iterations: int        # 最大迭代限制
    research_quality_scores: Dict[str, float] # 每个子问题的研究质量得分 (0.0-1.0)
    research_feedback: Dict[str, str]   # 针对质量不合格子问题的改进建议
    current_research_strategy: Dict[str, str] # 每个子问题当前采用的搜索策略

    execution_path: List[str]        # 节点执行历史列表
    active_sub_question: Optional[str] # 当前正在处理的子问题名

    metrics: WorkflowMetrics         # 性能度量指标
    timestamp: str                   # 运行时间戳
    comparison_mode: bool            # 是否为缓存对比模式

    llm_calls: Dict[str, int]        # 细粒度统计 LLM 调用次数
    llm_usage: List[Dict[str, Any]]  # 每次 LLM 调用的 token 元数据记录


def initialize_metrics() -> WorkflowMetrics:
    """初始化默认指标结构。"""
    return {
        "total_latency": 0.0,
        "decomposition_latency": 0.0,
        "cache_latency": 0.0,
        "research_latency": 0.0,
        "synthesis_latency": 0.0,
        "cache_hit_rate": 0.0,
        "cache_hits_count": 0,
        "questions_researched": 0,
        "total_research_iterations": 0,
        "llm_calls": {},
        "sub_question_count": 0,
        "execution_path": "",
    }


def update_metrics(current_metrics: WorkflowMetrics, **updates) -> WorkflowMetrics:
    """以不可变风格更新指标，返回一个新的字典。"""
    updated = current_metrics.copy()
    for key, value in updates.items():
        if key == "llm_calls" and isinstance(value, dict):
            # 增量更新 LLM 调用次数
            updated["llm_calls"] = {**updated["llm_calls"], **value}
        else:
            updated[key] = value
    return updated


def initialize_nodes(semantic_cache):
    """
    初始化节点依赖。
    
    Args:
        semantic_cache: 注入语义缓存实例。
    """
    global cache
    cache = semantic_cache


def decompose_question_node(state: WorkflowState) -> WorkflowState:
    """
    节点：将复杂查询分解为专注的、可缓存的子问题。
    
    逻辑：使用分析型 LLM 将用户提问拆解为 2-5 个独立的子问题。
    目的：实现细粒度缓存。例如，“A和B的区别”会被拆为“A是什么”和“B是什么”，
    下次有人问“A和C的区别”时，“A是什么”就可以直接从缓存读取。
    """
    start_time = time.perf_counter()
    query = state["original_query"]

    logger.info(f"🧠 调度器：正在拆解问题：'{query[:50]}...'")

    try:
        # 构建拆解提示词
        decomposition_prompt = f"""
        请分析下面这个客服场景问题，判断是否需要拆分为多个子问题。
        
        原始问题：{query}
        
        规则：
        - 如果问题简单且只聚焦一个主题，请仅输出：SINGLE_QUESTION
        - 如果问题包含多个独立方面且适合分别检索，请拆成 2-5 个具体子问题
        - 每个子问题都应自包含且适合做缓存复用
        
        如果需要拆分：只输出子问题，每行一个，不要编号。
        如果不需要拆分：请严格输出 SINGLE_QUESTION。
        """

        response = get_analysis_llm().invoke(
            [HumanMessage(content=decomposition_prompt)]
        )

        # 记录一次 LLM 调用计数
        llm_calls = state.get("llm_calls", {}).copy()
        llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + 1
        llm_usage = _append_llm_usage(
            state=state,
            model="deepseek-v3-2-251201",
            provider="volcengine",
            response=response,
            node="decompose_question_node",
        )

        response_content = response.content.strip()
        if response_content == "SINGLE_QUESTION":
            sub_questions = [query]
            logger.info("🧠 问题较简单，保持为单问题")
        else:
            # 解析 LLM 返回的行，过滤掉序号
            sub_questions = [
                line.strip()
                for line in response_content.split("\n")
                if line.strip()
                and not line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "-", "*"))
                and line.strip() != "SINGLE_QUESTION"
            ]

            # 异常处理：如果没有拆出结果，回退到原始提问
            if not sub_questions or len(sub_questions) == 1:
                sub_questions = [query]
            elif len(sub_questions) > 5:
                sub_questions = sub_questions[:5] # 限制数量，防止任务爆炸

        # 初始化每个子问题的研究状态
        research_iterations = {sq: 0 for sq in sub_questions}
        research_quality_scores = {sq: 0.0 for sq in sub_questions}
        research_feedback = {sq: "" for sq in sub_questions}
        current_research_strategy = {sq: "initial" for sq in sub_questions}

        decomposition_time = (time.perf_counter() - start_time) * 1000

        # 更新度量指标
        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            decomposition_latency=decomposition_time,
            sub_question_count=len(sub_questions),
            llm_calls={"analysis_llm": llm_calls.get("analysis_llm", 0)}
        )

        return {
            **state,
            "sub_questions": sub_questions,
            "sub_answers": {},
            "cache_hits": {},
            "cache_confidences": {},
            "research_iterations": research_iterations,
            "research_quality_scores": research_quality_scores,
            "research_feedback": research_feedback,
            "current_research_strategy": current_research_strategy,
            "execution_path": state["execution_path"] + ["decomposed"],
            "llm_calls": llm_calls,
            "llm_usage": llm_usage,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ 问题拆解失败: {e}")
        # 失败回退：将原始问题作为一个子问题处理
        return {
            **state,
            "sub_questions": [query],
            "sub_answers": {},
            "cache_hits": {},
            "cache_confidences": {},
            "research_iterations": {query: 0},
            "research_quality_scores": {query: 0.0},
            "research_feedback": {query: ""},
            "current_research_strategy": {query: "initial"},
            "execution_path": state["execution_path"] + ["decomposition_failed"],
        }


def check_cache_node(state: WorkflowState) -> WorkflowState:
    """
    节点：独立检查每个子问题的语义缓存。
    逻辑：遍历所有子问题，在 Redis 语义缓存中进行向量相似度检索。
    Workflow 位置：拆解节点之后。
    关键：如果置信度（Confidence）高于阈值，则认为命中，直接获取答案，无需调用昂贵的搜索/LLM。
    """
    if not cache:
        raise RuntimeError("Cache not initialized. Please call initialize_nodes() first.")

    start_time = time.perf_counter()
    sub_questions = state.get("sub_questions", [])

    logger.info(f"🔍 调度器：开始检查缓存，子问题数量={len(sub_questions)}")

    cache_hits = {}
    cache_confidences = {}
    cache_seed_ids = {}
    sub_answers = {}
    total_hits = 0

    try:
        # 支持全局禁用缓存（用于对比测试）
        if not state.get("cache_enabled", True):
            logger.info("🔀 缓存已关闭：本轮按全部未命中处理（用于对比）")
            updated_metrics = update_metrics(
                state.get("metrics", initialize_metrics()),
                cache_latency=0.0,
                cache_hit_rate=0.0,
                cache_hits_count=0
            )
            
            return {
                **state,
                "cache_hits": {sq: False for sq in sub_questions},
                "cache_confidences": {sq: 0.0 for sq in sub_questions},
                "cache_seed_ids": {sq: None for sq in sub_questions},
                "execution_path": state["execution_path"] + ["cache_disabled"],
                "metrics": updated_metrics,
            }

        # 逐个检查子问题缓存
        for sub_question in sub_questions:
            # 执行语义检索
            cache_results = cache.check(sub_question, num_results=1)

            if cache_results.matches:
                result = cache_results.matches[0]
                # 计算置信度得分（基于向量距离）
                confidence = (2.0 - result.vector_distance) / 2.0
                cache_hits[sub_question] = True
                cache_confidences[sub_question] = confidence
                cache_seed_ids[sub_question] = result.seed_id
                # 直接获取缓存中的答案
                sub_answers[sub_question] = result.response
                total_hits += 1

                logger.info(f"   ✅ 缓存命中: '{sub_question[:40]}...' (置信度: {confidence:.3f})")
            else:
                cache_hits[sub_question] = False
                cache_confidences[sub_question] = 0.0
                cache_seed_ids[sub_question] = None
                logger.info(f"   ❌ 缓存未命中: '{sub_question[:40]}...'")

        cache_latency = (time.perf_counter() - start_time) * 1000
        hit_rate = total_hits / len(sub_questions) if sub_questions else 0

        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            cache_latency=cache_latency,
            cache_hit_rate=hit_rate,
            cache_hits_count=total_hits
        )

        return {
            **state,
            "cache_hits": cache_hits,
            "cache_confidences": cache_confidences,
            "cache_seed_ids": cache_seed_ids,
            # 将缓存命中的答案合并到子问题答案库中
            "sub_answers": {**state.get("sub_answers", {}), **sub_answers},
            "execution_path": state["execution_path"] + ["cache_checked"],
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ 缓存检查失败: {e}")
        return {
            **state,
            "cache_hits": {sq: False for sq in sub_questions},
            "cache_confidences": {sq: 0.0 for sq in sub_questions},
            "cache_seed_ids": {sq: None for sq in sub_questions},
            "execution_path": state["execution_path"] + ["cache_check_failed"],
        }


def synthesize_response_node(state: WorkflowState) -> WorkflowState:
    """
    节点：将各部分的子答案汇总成一段连贯、全面的最终回答。
    逻辑：将缓存命中的答案和新研究出的答案按照原始子问题顺序排列，
    交给 LLM 进行润色和整合，确保语气自然且直接回答了用户的原始提问。
    """
    start_time = time.perf_counter()
    sub_questions = state.get("sub_questions", [])
    sub_answers = state.get("sub_answers", {})

    logger.info(f"🔗 调度器：开始汇总答案，当前可用答案数={len(sub_answers)}")

    try:
        # 组装 Q&A 对作为上下文
        qa_pairs = []
        for sq in sub_questions:
            if sq in sub_answers:
                qa_pairs.append(f"Q: {sq}\nA: {sub_answers[sq]}")

        if not qa_pairs:
            logger.warning("⚠️ 没有可用于汇总的答案")
            return {
                **state,
                "final_response": "抱歉，我暂时没有检索到可用答案。",
                "execution_path": state["execution_path"] + ["synthesis_failed"],
            }

        # 最终汇总提示词
        synthesis_prompt = f"""
        你是一名专业的中文客服助手。请将下列问答对整理成一段连贯、完整、自然的中文回复，
        直接回答用户的原始问题。
        
        原始问题：{state['original_query']}
        
        已收集信息：
        {chr(10).join(qa_pairs)}
        
        输出要求：使用自然、清晰、礼貌的中文。
        """

        messages = [
            SystemMessage(content="你是一名专业的中文客服助手。"),
            HumanMessage(content=synthesis_prompt),
        ]

        response = get_analysis_llm().invoke(messages)
        final_response = response.content.strip()

        llm_calls = state.get("llm_calls", {}).copy()
        llm_calls["analysis_llm"] = llm_calls.get("analysis_llm", 0) + 1
        llm_usage = _append_llm_usage(
            state=state,
            model="deepseek-v3-2-251201",
            provider="volcengine",
            response=response,
            node="synthesize_response_node",
        )

        synthesis_time = (time.perf_counter() - start_time) * 1000

        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            synthesis_latency=synthesis_time,
            llm_calls={"analysis_llm": llm_calls.get("analysis_llm", 0)}
        )

        return {
            **state,
            "final_response": final_response,
            "execution_path": state["execution_path"] + ["synthesized"],
            "llm_calls": llm_calls,
            "llm_usage": llm_usage,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ 答案汇总失败: {e}")
        return {
            **state,
            "final_response": "抱歉，在整理答案时发生了错误。",
            "execution_path": state["execution_path"] + ["synthesis_error"],
        }


def evaluate_quality_node(state: WorkflowState) -> WorkflowState:
    """
    节点：在综合答案前评估研究结果的质量和充分性。
    逻辑：使用 LLM 作为裁判，对每一个新研究出的答案进行打分 (0.0-1.0)。
    如果得分低于 0.7，会提供具体反馈，触发下一轮研究迭代。
    """
    start_time = time.perf_counter()
    sub_questions = state.get("sub_questions", [])
    sub_answers = state.get("sub_answers", {})
    cache_hits = state.get("cache_hits", {})

    logger.info(f"🎯 质量评估：开始评估 {len(sub_answers)} 条答案")

    quality_scores = state.get("research_quality_scores", {}).copy()
    feedback = state.get("research_feedback", {}).copy()
    needs_more_research = []
    llm_usage = list(state.get("llm_usage", []))

    llm_calls = state.get("llm_calls", {}).copy()
    research_llm = get_research_llm()

    try:
        # 定义子问题评估函数（用于并行调用）
        def evaluate_sub_question(sub_question, answer, current_iteration):
            evaluation_prompt = f"""
            请评估这条研究结果是否足以回答用户问题。
            
            子问题：{sub_question}
            研究结果：{answer}
            当前迭代：第 {current_iteration + 1} 轮
            
            请输出：
            1. 质量分（0.0 到 1.0，1.0 最好，0.7 及以上视为合格）
            2. 若低于 0.7，请给出简短改进建议
            
            输出格式必须严格为：
            SCORE: 0.X
            FEEDBACK: [改进建议]
            """
            
            try:
                evaluation = research_llm.invoke([HumanMessage(content=evaluation_prompt)])
                lines = evaluation.content.strip().split("\n")
                score_line = [l for l in lines if l.startswith("SCORE:")][0]
                feedback_line = [l for l in lines if l.startswith("FEEDBACK:")][0]
                
                score = float(score_line.split("SCORE:")[1].strip())
                feedback_text = feedback_line.split("FEEDBACK:")[1].strip()
                in_tokens, out_tokens = _extract_token_usage(evaluation)
                return sub_question, "success", score, feedback_text, in_tokens, out_tokens
            except Exception as e:
                # 解析失败默认给合格分，防止死循环
                return sub_question, "error", 0.8, f"Evaluation parsing failed: {e}", 0, 0

        # 并行执行所有子问题的评估
        eval_tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for sub_question in sub_questions:
                # 缓存命中的结果默认为满分，无需重复评估
                if cache_hits.get(sub_question, False):
                    quality_scores[sub_question] = 1.0
                    continue
                if sub_question not in sub_answers:
                    needs_more_research.append(sub_question)
                    continue
                
                answer = sub_answers[sub_question]
                current_iteration = state.get("research_iterations", {}).get(sub_question, 0)
                eval_tasks.append(executor.submit(evaluate_sub_question, sub_question, answer, current_iteration))

            for future in concurrent.futures.as_completed(eval_tasks):
                sub_question, status, score, feedback_text, in_tokens, out_tokens = future.result()
                llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1
                llm_usage.append(
                    {
                        "model": "ep-m-20260411093114-9hftc",
                        "provider": "volcengine",
                        "input_tokens": int(in_tokens),
                        "output_tokens": int(out_tokens),
                        "node": "evaluate_quality_node",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                
                quality_scores[sub_question] = score
                feedback[sub_question] = feedback_text
                
                if score < 0.7:
                    logger.info(f"   🔄 {sub_question[:40]}... - 分数: {score:.2f}（需改进）")
                else:
                    logger.info(f"   ✅ {sub_question[:40]}... - 分数: {score:.2f}（合格）")

        evaluation_time = (time.perf_counter() - start_time) * 1000

        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            llm_calls={"research_llm": llm_calls.get("research_llm", 0)}
        )

        return {
            **state,
            "research_quality_scores": quality_scores,
            "research_feedback": feedback,
            "execution_path": state["execution_path"] + ["quality_evaluated"],
            "llm_calls": llm_calls,
            "llm_usage": llm_usage,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ 质量评估失败: {e}")
        return {
            **state,
            "research_quality_scores": {sq: 0.8 for sq in sub_questions},
            "research_feedback": {sq: "Evaluation failed" for sq in sub_questions},
            "execution_path": state["execution_path"] + ["evaluation_failed"],
            "llm_calls": llm_calls,
        }


def research_node(state: WorkflowState) -> WorkflowState:
    """
    节点：执行研究逻辑，支持策略调整和迭代。
    逻辑：仅处理“缓存未命中”的子问题。
    使用 create_react_agent 创建一个带工具调用能力的智能体，调用 search_knowledge_base 工具从 Redis 中检索。
    如果是第二轮迭代，会结合上一轮的反馈（Feedback）生成更深入的搜索指令。
    """
    start_time = time.perf_counter()
    cache_hits = state.get("cache_hits", {})
    sub_answers = state.get("sub_answers", {}).copy()
    research_iterations = state.get("research_iterations", {}).copy()
    current_strategies = state.get("current_research_strategy", {}).copy()
    feedback = state.get("research_feedback", {})
    questions_researched = 0

    llm_calls = state.get("llm_calls", {}).copy()
    llm_usage = list(state.get("llm_usage", []))

    from langgraph.prebuilt import create_react_agent

    research_llm = get_research_llm()
    # 构造 ReAct 智能体，赋予其搜索知识库的能力
    researcher_agent = create_react_agent(
        model=research_llm, tools=[search_knowledge_base]
    )

    logger.info("🔬 研究阶段：开始按策略执行检索")

    try:
        # 并行处理每个需要研究的子问题
        def process_research(sub_question):
            current_iteration = research_iterations.get(sub_question, 0)
            strategy = current_strategies.get(sub_question, "initial")
            logger.info(f"🔍 正在研究：'{sub_question[:50]}...'（第 {current_iteration + 1} 轮）")
            
            research_prompt = sub_question
            # 如果是迭代改善阶段，注入质量评估的反馈
            if current_iteration > 0 and feedback.get(sub_question):
                research_prompt = f"""
                上一轮研究结果不够充分。改进建议：{feedback[sub_question]}
                原始问题：{sub_question}
                请根据上述建议进行更深入的检索与补充。
                """
            
            # 调用 ReAct Agent 进行研究
            research_result = researcher_agent.invoke({"messages": [HumanMessage(content=research_prompt)]})
            
            if research_result and "messages" in research_result:
                last_msg = research_result["messages"][-1]
                answer = last_msg.content
                in_tokens, out_tokens = _extract_token_usage(last_msg)
                return sub_question, "success", answer, current_iteration, in_tokens, out_tokens
            else:
                return sub_question, "failure", "未检索到有效信息。", current_iteration, 0, 0

        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for sub_question, is_cached in cache_hits.items():
                if not is_cached: # 关键：跳过缓存已命中的问题
                    tasks.append(executor.submit(process_research, sub_question))
            
            for future in concurrent.futures.as_completed(tasks):
                sub_question, status, answer, current_iteration, in_t, out_t = future.result()
                llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1

                llm_usage.append(
                    {
                        "model": "ep-m-20260411093114-9hftc",
                        "provider": "volcengine",
                        "input_tokens": int(in_t),
                        "output_tokens": int(out_t),
                        "node": "research_node",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                if status == "success":
                    sub_answers[sub_question] = answer
                    questions_researched += 1
                    research_iterations[sub_question] = current_iteration + 1
                    current_strategies[sub_question] = f"iteration_{current_iteration + 1}"
                else:
                    sub_answers[sub_question] = answer

        research_time = (time.perf_counter() - start_time) * 1000

        updated_metrics = update_metrics(
            state.get("metrics", initialize_metrics()),
            research_latency=research_time,
            questions_researched=questions_researched,
            total_research_iterations=sum(research_iterations.values()),
            llm_calls={"research_llm": llm_calls.get("research_llm", 0)}
        )

        return {
            **state,
            "sub_answers": sub_answers,
            "research_iterations": research_iterations,
            "current_research_strategy": current_strategies,
            "execution_path": state["execution_path"] + ["researched"],
            "llm_calls": llm_calls,
            "llm_usage": llm_usage,
            "metrics": updated_metrics,
        }

    except Exception as e:
        logger.error(f"❌ 研究阶段失败: {e}")
        return {
            **state,
            "sub_answers": sub_answers,
            "execution_path": state["execution_path"] + ["research_failed"],
            "llm_calls": llm_calls,
        }


def decompose_query_node(state: WorkflowState) -> WorkflowState:
    """
    工作流入口节点：按编号顺序准备子问题，不再进行 LLM 拆解。

    规则：
    - 如果原始问题里包含 "1."、"2." 这类编号行，则按出现顺序提取为子问题。
    - 如果没有编号行，则将原始问题作为单个子问题处理。
    - 后续缓存检查/研究节点仍可并行执行，以降低端到端延迟。
    """
    start_time = time.perf_counter()
    query = state["original_query"]

    logger.info("🧠 入口节点：按编号顺序准备子问题（不做拆解）")

    numbered_questions: List[str] = []
    for raw_line in query.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # 匹配如 "1. 问题"、"2) 问题"、"3、问题" 的常见编号格式
        matched = re.match(r"^\d+\s*[\.|\)|、]\s*(.+)$", line)
        if matched:
            question = matched.group(1).strip()
            if question:
                numbered_questions.append(question)

    sub_questions = numbered_questions or [query.strip()]

    research_iterations = {sq: 0 for sq in sub_questions}
    research_quality_scores = {sq: 0.0 for sq in sub_questions}
    research_feedback = {sq: "" for sq in sub_questions}
    current_research_strategy = {sq: "initial" for sq in sub_questions}

    preparation_time = (time.perf_counter() - start_time) * 1000
    updated_metrics = update_metrics(
        state.get("metrics", initialize_metrics()),
        decomposition_latency=preparation_time,
        sub_question_count=len(sub_questions),
    )

    logger.info("🧠 子问题准备完成：%d 条", len(sub_questions))
    return {
        **state,
        "sub_questions": sub_questions,
        "sub_answers": {},
        "cache_hits": {},
        "cache_confidences": {},
        "cache_seed_ids": {},
        "research_iterations": research_iterations,
        "research_quality_scores": research_quality_scores,
        "research_feedback": research_feedback,
        "current_research_strategy": current_research_strategy,
        "execution_path": state["execution_path"] + ["numbered_prepared"],
        "metrics": updated_metrics,
    }