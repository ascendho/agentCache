"""Tests for tests.summary_schema — explicit CSV row contract."""

import json
import unittest

from tests.summary_schema import (
    SUMMARY_FIELDNAMES,
    build_summary_row,
    _validate_row_keys,
)


class SummarySchemaTests(unittest.TestCase):
    def test_fieldnames_unique_and_ordered(self):
        self.assertEqual(len(SUMMARY_FIELDNAMES), len(set(SUMMARY_FIELDNAMES)))
        self.assertEqual(SUMMARY_FIELDNAMES[0], "test_index")
        self.assertEqual(SUMMARY_FIELDNAMES[-1], "final_response")

    def test_required_columns_present(self):
        required = {
            "test_index",
            "original_query",
            "intercepted",
            "cache_hit",
            "cache_match_type",
            "cache_reuse_mode",
            "total_cost_rmb",
            "total_latency_ms",
            "precheck_latency_ms",
            "synthesis_latency_ms",
            "final_response",
        }
        self.assertTrue(required.issubset(set(SUMMARY_FIELDNAMES)))

    def test_build_row_keys_match_schema_for_empty_result(self):
        row = build_summary_row(1, {})
        self.assertEqual(tuple(row.keys()), SUMMARY_FIELDNAMES)

    def test_build_row_keys_match_schema_for_full_result(self):
        result = {
            "query": "如何修改地址？",
            "intercepted": False,
            "cache_hit": True,
            "cache_match_type": "exact",
            "cache_reuse_mode": "full_reuse",
            "cache_matched_question": "如何修改地址？",
            "cache_confidence": 0.987,
            "cache_rerank_passed": True,
            "cache_rerank_attempt": "primary",
            "cache_rerank_score": 0.95,
            "cache_rerank_reason": "ok",
            "cache_reranker_reason": "ok",
            "cache_validation_reason": "ok",
            "cache_residual_query": "",
            "cache_written_prompts": ["填错地址了能改吗？"],
            "final_response": "...",
            "llm_calls": {"analysis_llm": 1, "research_llm": 2},
            "llm_usage": {
                "analysis_input_tokens": 100,
                "analysis_output_tokens": 50,
                "analysis_cached_input_tokens": 0,
                "research_input_tokens": 200,
                "research_output_tokens": 80,
                "research_cached_input_tokens": 30,
                "total_input_tokens": 300,
                "total_output_tokens": 130,
                "total_cached_input_tokens": 30,
                "analysis_cost_rmb": 0.001234,
                "research_cost_rmb": 0.005678,
                "total_cost_rmb": 0.006912,
            },
            "metrics": {
                "precheck_latency": 12,
                "cache_latency": 34,
                "rerank_latency": 56,
                "research_latency": 78,
                "supplement_latency": 90,
                "synthesis_latency": 11,
                "total_latency": 281,
            },
        }
        row = build_summary_row(7, result)
        self.assertEqual(tuple(row.keys()), SUMMARY_FIELDNAMES)
        self.assertEqual(row["test_index"], 7)
        self.assertEqual(row["original_query"], "如何修改地址？")
        self.assertEqual(row["total_llm_calls"], 3)
        self.assertEqual(row["analysis_input_tokens"], 100)
        self.assertEqual(row["total_cost_rmb"], "0.006912")
        self.assertEqual(row["total_latency_ms"], "281")
        # cache_written_prompts must be JSON-serialised
        self.assertEqual(json.loads(row["cache_written_prompts"]), ["填错地址了能改吗？"])

    def test_build_row_handles_missing_subdicts(self):
        row = build_summary_row(2, {"query": "x"})
        self.assertEqual(row["analysis_llm_calls"], 0)
        self.assertEqual(row["total_llm_calls"], 0)
        self.assertEqual(row["analysis_input_tokens"], 0)
        self.assertEqual(row["total_cost_rmb"], "0.000000")
        self.assertEqual(row["total_latency_ms"], "0")
        self.assertEqual(row["cache_match_type"], "none")
        self.assertEqual(row["cache_reuse_mode"], "none")
        self.assertEqual(row["cache_written_prompts"], "[]")

    def test_validate_row_keys_rejects_missing(self):
        good = build_summary_row(1, {})
        bad = dict(good)
        bad.pop("final_response")
        with self.assertRaises(ValueError):
            _validate_row_keys(bad)

    def test_validate_row_keys_rejects_extra(self):
        good = build_summary_row(1, {})
        bad = dict(good)
        bad["unexpected_column"] = "boom"
        with self.assertRaises(ValueError):
            _validate_row_keys(bad)


if __name__ == "__main__":
    unittest.main()
