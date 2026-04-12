import csv
import json
import os
from typing import Dict, List


def export_results(all_results: List[Dict], output_dir: str = "outputs") -> Dict[str, str]:
    """
    将运行结果导出为 CSV + JSON 文件，便于后续分析与归档。

    Returns:
        导出文件路径字典。
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "run_summary.csv")
    result_json = os.path.join(output_dir, "run_results.json")

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
                    "final_response": result.get("final_response", ""),
                }
            )

    payload = {"results": all_results}
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "summary_csv": summary_csv,
        "result_json": result_json,
    }
