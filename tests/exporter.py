import csv
import os
from typing import Dict, List


def export_results(all_results: List[Dict], total_wall_time_sec: float, output_dir: str = "outputs") -> Dict[str, str]:
    """
    将运行结果导出为纯净的统报表CSV文件，分别提供明细以及对命中率、吞吐量等指标的性能聚合(TXT)。
    """
    os.makedirs(output_dir, exist_ok=True)

    summary_csv = os.path.join(output_dir, "run_summary.csv")
    perf_metrics_txt = os.path.join(output_dir, "performance_report.txt")

    # 1. 场景级汇总：每条主查询一行。
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_index",
                "original_query",
                "cache_hit",
                "cache_matched_question",
                "cache_confidence",
                "research_iterations",
                "analysis_llm_calls",
                "research_llm_calls",
                "total_latency_ms",
                "final_response",
            ],
        )
        writer.writeheader()

        for idx, result in enumerate(all_results, 1):
            cache_hit = result.get("cache_hit", False)
            cache_confidence = result.get("cache_confidence", 0.0)
            research_iterations = result.get("research_iterations", 0)

            llm_calls = result.get("llm_calls", {})
            metrics = result.get("metrics", {})
            total_latency = f"{metrics.get('total_latency', 0):.0f}"

            writer.writerow(
                {
                    "scenario_index": idx,
                    "original_query": result.get("query", ""),
                    "cache_hit": str(cache_hit),
                    "cache_matched_question": result.get("cache_matched_question", ""),
                    "cache_confidence": f"{cache_confidence:.4f}",
                    "research_iterations": research_iterations,
                    "analysis_llm_calls": llm_calls.get("analysis_llm", 0),
                    "research_llm_calls": llm_calls.get("research_llm", 0),
                    "total_latency_ms": total_latency,
                    "final_response": result.get("final_response", ""),
                }
            )

    # 2. 计算并聚合性能级指标报表
    total_queries = len(all_results)
    cache_hits = [r for r in all_results if r.get("cache_hit", False)]
    cache_misses = [r for r in all_results if not r.get("cache_hit", False)]
    
    num_hits = len(cache_hits)
    hit_rate = num_hits / total_queries if total_queries > 0 else 0.0
    throughput = total_queries / total_wall_time_sec if total_wall_time_sec > 0 else 0.0
    
    def avg_latency(subset):
        if not subset: return 0.0
        return sum(r.get("metrics", {}).get("total_latency", 0) for r in subset) / len(subset)
    
    def avg_llm_calls(subset):
        if not subset: return 0.0
        return sum(sum(r.get("llm_calls", {}).values()) for r in subset) / len(subset)
    
    miss_latency = avg_latency(cache_misses)
    hit_latency = avg_latency(cache_hits)
    
    miss_cost = avg_llm_calls(cache_misses)
    # hit_cost = avg_llm_calls(cache_hits)  # Usually 0 for cache hits
    
    # WCL Calculations (基于图1 WCL公式)
    acl = hit_latency
    all_time = miss_latency - hit_latency if miss_latency > hit_latency else 0.0
    chr_rate = hit_rate
    wcl = acl * chr_rate + (all_time + acl) * (1 - chr_rate)
    
    # Total Latency Reduction (基于整体执行耗时)
    theory_total_time_without_cache = miss_latency * total_queries
    actual_total_time = sum(r.get("metrics", {}).get("total_latency", 0) for r in all_results)
    
    if theory_total_time_without_cache > 0:
        latency_reduction = ((theory_total_time_without_cache - actual_total_time) / theory_total_time_without_cache) * 100
    else:
        latency_reduction = 0.0
    
    # 成本节省 (Cost Savings) 
    cost_savings_calls = hit_rate * miss_cost
    
    # 吞吐量指标
    baseline_qps = 1000.0 / miss_latency if miss_latency > 0 else 0.0
    max_qps = 1000.0 / hit_latency if hit_latency > 0 else 0.0

    report_text = f"""======================================================
         AGENT CACHE PERFORMANCE REPORT
======================================================

1. 总体概况 (Overview)
------------------------------------------------------
测试集总请求数 : {total_queries} 次
缓存命中数     : {num_hits} 次
缓存命中率     : {hit_rate * 100:.2f}%
测试总墙上时间 : {total_wall_time_sec:.2f} 秒

2. 工作负载延迟分析 (WCL - Workload Cache Latency)
------------------------------------------------------
WCL = ACL * CHR + (ALL + ACL) * (1 - CHR)

- ACL (Average Cache Latency) = {acl:.0f} ms
- ALL (Average LLM Latency)   = {all_time:.0f} ms
- CHR (Cache Hit Ratio)       = {chr_rate * 100:.2f}%

=> WCL = {wcl:.0f} ms

3. 吞吐量对比 (Throughput / QPS)
------------------------------------------------------
无缓存基线 QPS (0%命中)       : {baseline_qps:.2f} 请求/秒
加入缓存后实测 QPS            : {throughput:.2f} 请求/秒
全量缓存理论峰值 QPS(100%命中): {max_qps:.2f} 请求/秒
吞吐量提升倍数                : {throughput / baseline_qps if baseline_qps > 0 else 0:.2f} 倍

4. 延迟总降低 (Total Latency Reduction)
------------------------------------------------------
理论上无缓存总耗时            : {theory_total_time_without_cache:.0f} ms
实际执行总耗时                : {actual_total_time:.0f} ms
📌 Latency Reduction         : {latency_reduction:.2f}% 

(公式: (理论无缓存总耗时 - 实际执行总耗时) / 理论无缓存总耗时)

5. 成本节省 (Cost Savings)
------------------------------------------------------
大模型平均单次调用数 (LLM_cost): {miss_cost:.2f} 次
📌 Cost Savings (节省调用频次) : {cost_savings_calls:.2f} 次 / 请求

(公式: Hit_rate x LLM_cost_per_query)
======================================================
"""
    with open(perf_metrics_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 尝试删除旧的csv指标文件
    old_csv_path = os.path.join(output_dir, "performance_metrics.csv")
    if os.path.exists(old_csv_path):
        try:
            os.remove(old_csv_path)
        except OSError:
            pass

    return {
        "summary_csv": summary_csv,
        "perf_metrics_txt": perf_metrics_txt,
    }
