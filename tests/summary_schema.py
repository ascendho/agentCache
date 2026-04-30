"""测试导出的明细行 schema 单一真源。

把 `run_summary.csv` 的列集合显式化为 `SUMMARY_FIELDNAMES`，
并把行构造器集中到 `build_summary_row`，避免列定义散落到 exporter 中、
又通过 `dict.keys()` 派生 fieldnames 形成隐式契约。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple


# 明细 CSV 的列顺序 —— 这是对外契约，调整需谨慎，并同步更新单测。
SUMMARY_FIELDNAMES: Tuple[str, ...] = (
    "test_index",
    "original_query",
    "intercepted",
    "cache_hit",
    "cache_candidate_found",
    "cache_match_type",
    "cache_reuse_mode",
    "cache_matched_question",
    "cache_confidence",
    "cache_rerank_passed",
    "cache_rerank_attempt",
    "cache_rerank_score",
    "cache_rerank_reason",
    "cache_reranker_reason",
    "cache_validation_reason",
    "cache_residual_query",
    "cache_written_prompts",
    "analysis_llm_calls",
    "research_llm_calls",
    "total_llm_calls",
    "analysis_input_tokens",
    "analysis_output_tokens",
    "analysis_cached_input_tokens",
    "research_input_tokens",
    "research_output_tokens",
    "research_cached_input_tokens",
    "total_input_tokens",
    "total_output_tokens",
    "total_cached_input_tokens",
    "analysis_cost_rmb",
    "research_cost_rmb",
    "total_cost_rmb",
    "precheck_latency_ms",
    "cache_latency_ms",
    "rerank_latency_ms",
    "research_latency_ms",
    "supplement_latency_ms",
    "synthesis_latency_ms",
    "total_latency_ms",
    "final_response",
)


def build_summary_row(idx: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """将单条工作流结果展平为 CSV 明细行所需的纯字段字典。

    返回字典的键集合必须与 :data:`SUMMARY_FIELDNAMES` 完全一致，
    任何新增/移除字段都应在此函数与列表中同步修改。
    """
    llm_calls = result.get("llm_calls", {}) or {}
    llm_usage = result.get("llm_usage", {}) or {}
    metrics = result.get("metrics", {}) or {}

    def _int(field: str) -> int:
        return int(llm_usage.get(field, 0) or 0)

    def _money(field: str) -> str:
        return f"{float(llm_usage.get(field, 0) or 0):.6f}"

    def _ms(field: str) -> str:
        return f"{metrics.get(field, 0):.0f}"

    row: Dict[str, Any] = {
        "test_index": idx,
        "original_query": result.get("query", ""),
        "intercepted": str(result.get("intercepted", False)),
        "cache_hit": str(result.get("cache_hit", False)),
        "cache_candidate_found": str(bool(result.get("cache_matched_question"))),
        "cache_match_type": result.get("cache_match_type", "none"),
        "cache_reuse_mode": result.get("cache_reuse_mode", "none"),
        "cache_matched_question": result.get("cache_matched_question", ""),
        "cache_confidence": f"{result.get('cache_confidence', 0.0):.4f}",
        "cache_rerank_passed": str(result.get("cache_rerank_passed", False)),
        "cache_rerank_attempt": result.get("cache_rerank_attempt", "none"),
        "cache_rerank_score": f"{result.get('cache_rerank_score', 0.0):.4f}",
        "cache_rerank_reason": result.get("cache_rerank_reason", ""),
        "cache_reranker_reason": result.get("cache_reranker_reason", ""),
        "cache_validation_reason": result.get("cache_validation_reason", ""),
        "cache_residual_query": result.get("cache_residual_query", ""),
        "cache_written_prompts": json.dumps(result.get("cache_written_prompts", []) or [], ensure_ascii=False),
        "analysis_llm_calls": llm_calls.get("analysis_llm", 0),
        "research_llm_calls": llm_calls.get("research_llm", 0),
        "total_llm_calls": sum(llm_calls.values()),
        "analysis_input_tokens": _int("analysis_input_tokens"),
        "analysis_output_tokens": _int("analysis_output_tokens"),
        "analysis_cached_input_tokens": _int("analysis_cached_input_tokens"),
        "research_input_tokens": _int("research_input_tokens"),
        "research_output_tokens": _int("research_output_tokens"),
        "research_cached_input_tokens": _int("research_cached_input_tokens"),
        "total_input_tokens": _int("total_input_tokens"),
        "total_output_tokens": _int("total_output_tokens"),
        "total_cached_input_tokens": _int("total_cached_input_tokens"),
        "analysis_cost_rmb": _money("analysis_cost_rmb"),
        "research_cost_rmb": _money("research_cost_rmb"),
        "total_cost_rmb": _money("total_cost_rmb"),
        "precheck_latency_ms": _ms("precheck_latency"),
        "cache_latency_ms": _ms("cache_latency"),
        "rerank_latency_ms": _ms("rerank_latency"),
        "research_latency_ms": _ms("research_latency"),
        "supplement_latency_ms": _ms("supplement_latency"),
        "synthesis_latency_ms": _ms("synthesis_latency"),
        "total_latency_ms": _ms("total_latency"),
        "final_response": result.get("final_response", ""),
    }
    _validate_row_keys(row)
    return row


def _validate_row_keys(row: Mapping[str, Any]) -> None:
    """校验行字典的键集合与 SUMMARY_FIELDNAMES 完全一致，缺/多即报错。"""
    expected = set(SUMMARY_FIELDNAMES)
    actual = set(row.keys())
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            "summary row schema mismatch: "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
