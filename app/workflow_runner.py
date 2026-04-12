from typing import Any, Dict

from agent import analyze_agent_results, display_results, run_agent
from .scenarios import SCENARIO_RUNS
from utility.results_export import export_results


def run_workflow_scenarios(
    workflow_app: Any,
    logger,
    show_console_results: bool = False,
) -> Dict[str, Any]:
    """运行场景、导出结果并输出分析日志。"""
    all_results = []
    for scenario in SCENARIO_RUNS:
        logger.info("运行场景：%s", scenario["title"])
        result = run_agent(workflow_app, scenario["query"])
        all_results.append(result)
        if show_console_results:
            display_results(result)

    export_paths = export_results(all_results)
    logger.info("结果已导出 | 场景汇总=%s", export_paths["summary_csv"])
    logger.info("结果已导出 | 全量JSON=%s", export_paths["result_json"])

    logger.info("分析 Agent 性能...")
    total_questions, total_cache_hits = analyze_agent_results(all_results)
    logger.info("子问题总数: %s, 缓存命中总数: %s", total_questions, total_cache_hits)

    return {
        "results": all_results,
        "exports": export_paths,
        "total_questions": total_questions,
        "total_cache_hits": total_cache_hits,
    }
