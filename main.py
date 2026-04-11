# 运行指令：python main.py

import csv
import json
import logging
import os
import warnings
from datetime import datetime
from cache.llm_evaluator import set_ark_key
from redisvl.utils.vectorize import HFTextVectorizer
from agent import create_knowledge_base_from_texts
from cache.wrapper import SemanticCacheWrapper
from cache.config import config
from cache.faq_data_container import FAQDataContainer
from cache.evals import PerfEval, format_cost
from agent.graph import build_workflow
from agent import (
    run_agent,
    display_results,
    analyze_agent_results
)
from scenarios import SCENARIO_1_QUERY, SCENARIO_2_QUERY, SCENARIO_3_QUERY


def _to_bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def export_results(all_results, total_costs, output_dir="output_images"):
    """
    将运行结果导出为 CSV + JSON 文件，便于后续分析与归档。

    Returns:
        导出文件路径字典。
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_csv = os.path.join(output_dir, f"run_summary_{ts}.csv")
    usage_csv = os.path.join(output_dir, f"llm_usage_{ts}.csv")
    result_json = os.path.join(output_dir, f"run_results_{ts}.json")

    # 场景级汇总：每条主查询一行。
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_index",
                "original_query",
                "sub_questions_count",
                "cache_hits",
                "cache_hit_rate",
                "analysis_llm_calls",
                "research_llm_calls",
                "total_latency_ms",
                "cost",
                "currency",
                "final_response",
            ],
        )
        writer.writeheader()

        for idx, result in enumerate(all_results, 1):
            sub_questions = result.get("sub_questions", [])
            cache_hits_map = result.get("cache_hits", {})
            hits = sum(1 for v in cache_hits_map.values() if v)
            total_sq = len(sub_questions)
            hit_rate = (hits / total_sq) if total_sq else 0.0

            llm_calls = result.get("llm_calls", {})
            total_latency = str(result.get("total_latency", "0ms")).replace("ms", "")

            per_perf = PerfEval()
            for call in result.get("llm_usage", []):
                per_perf.record_llm_call(
                    model=call.get("model", "unknown-model"),
                    provider=call.get("provider", "openai"),
                    input_tokens=int(call.get("input_tokens", 0) or 0),
                    output_tokens=int(call.get("output_tokens", 0) or 0),
                )
            per_perf.set_total_queries(1)
            per_cost = per_perf.get_costs()

            writer.writerow(
                {
                    "scenario_index": idx,
                    "original_query": result.get("original_query", ""),
                    "sub_questions_count": total_sq,
                    "cache_hits": hits,
                    "cache_hit_rate": f"{hit_rate:.4f}",
                    "analysis_llm_calls": llm_calls.get("analysis_llm", 0),
                    "research_llm_calls": llm_calls.get("research_llm", 0),
                    "total_latency_ms": total_latency,
                    "cost": per_cost.get("total_cost", 0.0),
                    "currency": per_cost.get("currency", total_costs.get("currency", "CNY")),
                    "final_response": result.get("final_response", ""),
                }
            )

    # 调用级明细：每次 LLM 调用一行。
    with open(usage_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_index",
                "provider",
                "model",
                "input_tokens",
                "output_tokens",
            ],
        )
        writer.writeheader()

        for idx, result in enumerate(all_results, 1):
            for call in result.get("llm_usage", []):
                writer.writerow(
                    {
                        "scenario_index": idx,
                        "provider": call.get("provider", "openai"),
                        "model": call.get("model", "unknown-model"),
                        "input_tokens": int(call.get("input_tokens", 0) or 0),
                        "output_tokens": int(call.get("output_tokens", 0) or 0),
                    }
                )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "cost_summary": total_costs,
        "results": all_results,
    }
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "summary_csv": summary_csv,
        "usage_csv": usage_csv,
        "result_json": result_json,
    }

def setup_logging():
    """初始化全局日志配置，统一输出格式。"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("agentic-workflow")

def create_knowledge_base():
    """构建演示用知识库并返回索引与向量模型。"""
    # 这里使用轻量句向量模型，兼顾速度与基本语义检索质量。
    embeddings = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    raw_docs = [
        "Our premium support plan includes 24/7 phone support, priority email response within 2 hours, and dedicated account management. Premium support costs $49/month.",
        "Account upgrade process: Go to Account Settings → Plan & Billing → Select Upgrade. Available plans: Basic $9/month, Pro $29/month, Enterprise $99/month.",
        "API rate limits by plan: Free tier 100 requests/hour, Basic 1,000 requests/hour, Pro 10,000 requests/hour, Enterprise unlimited with fair-use policy.",
        "Data export options: CSV, JSON, XML formats supported. Large exports (>1GB) may take up to 24 hours to process.",
        "Third-party integrations: Native support for Slack, Microsoft Teams, Zoom, Salesforce, HubSpot. 200+ additional integrations available via Zapier.",
        "Security features: SOC2 compliance, end-to-end encryption, GDPR compliance, SSO integration, audit logs, IP whitelisting.",
        "Billing and payments: We accept all major credit cards, PayPal, and ACH transfers. Enterprise customers can pay by invoice with NET30 terms.",
        "Account recovery: Use forgot password link, verify email, or contact support with account verification details. Response within 4 hours.",
    ]

    success, message, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,
        source_id="customer_support_docs",
        redis_url="redis://localhost:6379",
        skip_chunking=True
    )
    
    return kb_index, embeddings

def setup_semantic_cache():
    """初始化语义缓存，并使用 FAQ 数据进行预热。"""
    # 预热后第一次请求即可命中部分常见问答，便于对比缓存收益。
    cache = SemanticCacheWrapper.from_config(config)
    data = FAQDataContainer()
    cache.hydrate_from_df(data.faq_df, clear=True)
    return cache

def main():
    """程序入口：完成初始化、执行场景测试并统计缓存表现。"""
    warnings.simplefilter("ignore")
    set_ark_key()
    logger = setup_logging()

    # 1) 创建知识库：把原始文本写入向量索引，供后续检索。
    logger.info("Initializing Knowledge Base...")
    kb_index, embeddings = create_knowledge_base()

    # 2) 初始化语义缓存：用于拦截相似问题，减少重复推理成本。
    logger.info("Setting up Semantic Cache...")
    cache = setup_semantic_cache()

    # 3) 组装工作流：节点、路由、工具在这里被编排成可执行图。
    logger.info("Building LangGraph Workflow...")
    workflow_app = build_workflow(cache, kb_index, embeddings)

    # 4) 依次执行三个场景，观察命中率随对话推进是否提升。
    show_console_results = _to_bool_env("SHOW_CONSOLE_RESULTS", False)

    logger.info("Running Scenario 1: Enterprise Platform Evaluation")
    result1 = run_agent(workflow_app, SCENARIO_1_QUERY)
    if show_console_results:
        display_results(result1)

    logger.info("Running Scenario 2: Implementation Planning")
    result2 = run_agent(workflow_app, SCENARIO_2_QUERY)
    if show_console_results:
        display_results(result2)

    logger.info("Running Scenario 3: Pre-Purchase Comprehensive Review")
    result3 = run_agent(workflow_app, SCENARIO_3_QUERY)
    if show_console_results:
        display_results(result3)

    # 5) 汇总 LLM token 元数据并做成本评估（支持可配置币种展示）。
    perf = PerfEval()
    all_results = [result1, result2, result3]
    for result in all_results:
        for call in result.get("llm_usage", []):
            perf.record_llm_call(
                model=call.get("model", "unknown-model"),
                provider=call.get("provider", "openai"),
                input_tokens=int(call.get("input_tokens", 0) or 0),
                output_tokens=int(call.get("output_tokens", 0) or 0),
            )
    perf.set_total_queries(len(all_results))
    costs = perf.get_costs()

    logger.info(
        "Cost Summary | currency=%s | total=%s | per_query=%s | per_call=%s | calls=%d",
        costs.get("currency", "CNY"),
        format_cost(costs.get("total_cost", 0.0), costs.get("currency", "CNY")),
        format_cost(costs.get("avg_cost_per_query", 0.0), costs.get("currency", "CNY")),
        format_cost(costs.get("avg_cost_per_call", 0.0), costs.get("currency", "CNY")),
        costs.get("calls", 0),
    )

    export_paths = export_results(all_results, costs)
    logger.info("Results exported | summary=%s", export_paths["summary_csv"])
    logger.info("Results exported | usage=%s", export_paths["usage_csv"])
    logger.info("Results exported | json=%s", export_paths["result_json"])

    # 6) 输出全局统计并落盘可视化图，用于复盘性能收益。
    logger.info("Analyzing Agent Performance...")
    total_questions, total_cache_hits = analyze_agent_results(
        [result1, result2, result3]
    )
    logger.info(f"Total Sub-Questions Evaluated: {total_questions}, Total Cache Hits: {total_cache_hits}")

if __name__ == "__main__":
    main()
