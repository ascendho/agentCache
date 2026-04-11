
import logging
import warnings
from cache.llm_evaluator import set_ark_key
from main import setup_logging, create_knowledge_base, setup_semantic_cache
from agent.graph import build_workflow
from agent import run_agent, display_results

def run_quick_test():
    """
    运行精简测试，验证工作流与语义缓存是否正常。

    通过两个语义相近的问题，观察第一次未命中、第二次命中的典型路径，
    以较低成本快速确认核心链路可用。
    """
    warnings.simplefilter("ignore")
    set_ark_key()
    logger = setup_logging()

    print("\n" + "="*60)
    print("🚀 开始快速核心逻辑测试")
    print("="*60)

    # 1) 复用主程序初始化流程：知识库 + 缓存 + 工作流
    logger.info("Initializing Knowledge Base...")
    kb_index, embeddings = create_knowledge_base()

    logger.info("Setting up Semantic Cache...")
    cache = setup_semantic_cache()

    logger.info("Building LangGraph Workflow...")
    workflow_app = build_workflow(cache, kb_index, embeddings)

    # ------------------
    # 场景 1：首次提问（预期：缓存未命中，走检索与归纳路径）
    # ------------------
    print("\n" + "-"*60)
    print("🧪 测试用例 1: 首次查询（预期：缓存未命中 -> 触发检索）")
    print("-"*60)
    query_1 = "How much does the Premium support plan cost?"
    result1 = run_agent(workflow_app, query_1)
    display_results(result1)

    # ------------------
    # 场景 2：语义相似提问（预期：缓存命中，直接返回答案）
    # ------------------
    print("\n" + "-"*60)
    print("🧪 测试用例 2: 语义相似查询（预期：缓存命中 -> 直接返回）")
    print("-"*60)
    query_2 = "What is the price of the Premium support plan per month?"
    result2 = run_agent(workflow_app, query_2)
    display_results(result2)

    print("\n✅ 测试完成！若第二个问题命中缓存，说明核心缓存链路运行正常。")
    print("相较完整场景测试，本脚本资源消耗更低。\n")

if __name__ == "__main__":
    run_quick_test()
