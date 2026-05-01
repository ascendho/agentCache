"""Focused tests for partial-reuse residual extraction and supplement search guardrails."""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from cache.engine import SemanticCacheWrapper  # noqa: E402
import workflow.nodes as nodes_module  # noqa: E402
from workflow.nodes import (  # noqa: E402
    _coerce_tool_args,
    _derive_deterministic_subquery_residual,
    check_cache_node,
    research_supplement_node,
)


def _make_fake_cache_wrapper():
    fake = types.SimpleNamespace()
    fake._seed_id_by_question = {}
    fake._answer_by_question = {}
    fake._normalized_question_map = {}
    fake._near_exact_question_map = {}

    class _FakeVectorCache:
        @staticmethod
        def store(prompt, response):
            return None

        @staticmethod
        def check(query, distance_threshold=None, num_results=1):
            return []

        @staticmethod
        def clear():
            return None

    fake.cache = _FakeVectorCache()
    fake.normalize_query = SemanticCacheWrapper.normalize_query
    fake.normalize_surface_query = SemanticCacheWrapper.normalize_surface_query
    fake.split_query_segments = SemanticCacheWrapper.split_query_segments
    fake.find_subquery_candidate = SemanticCacheWrapper.find_subquery_candidate.__get__(fake)
    fake._levenshtein_distance_with_limit = SemanticCacheWrapper._levenshtein_distance_with_limit
    fake.find_edit_distance_candidate = SemanticCacheWrapper.find_edit_distance_candidate.__get__(fake)
    fake.register_entry = SemanticCacheWrapper.register_entry.__get__(fake)
    fake.check = SemanticCacheWrapper.check.__get__(fake)
    return fake


class DeterministicResidualTests(unittest.TestCase):
    def test_returns_only_uncovered_segment_for_two_part_query(self):
        residual = _derive_deterministic_subquery_residual(
            "怎么联系人工？我买的书缺页了还能退吗？",
            "怎么联系人工？",
        )

        self.assertEqual(residual, "我买的书缺页了还能退吗")

    def test_returns_empty_when_multiple_uncovered_segments_exist(self):
        residual = _derive_deterministic_subquery_residual(
            "怎么联系人工？我买的书缺页了还能退吗？退款几天到账？",
            "怎么联系人工？",
        )

        self.assertEqual(residual, "")


class CoerceToolArgsTests(unittest.TestCase):
    def test_keeps_original_tool_query_when_not_locked(self):
        tool_args = _coerce_tool_args(
            "search_knowledge_base",
            {"query": "联系人工客服 联系方式", "top_k": 2},
        )

        self.assertEqual(tool_args, {"query": "联系人工客服 联系方式", "top_k": 2})

    def test_overrides_supplement_search_query_with_residual(self):
        tool_args = _coerce_tool_args(
            "search_knowledge_base",
            {"query": "联系人工客服 联系方式 客服电话", "top_k": 4},
            locked_search_query="我买的书缺页了还能退吗",
        )

        self.assertEqual(tool_args, {"query": "我买的书缺页了还能退吗", "top_k": 4})

    def test_leaves_other_tools_untouched_even_when_locked(self):
        tool_args = _coerce_tool_args(
            "other_tool",
            {"query": "联系人工客服 联系方式 客服电话"},
            locked_search_query="我买的书缺页了还能退吗",
        )

        self.assertEqual(tool_args, {"query": "联系人工客服 联系方式 客服电话"})


class CheckCacheNodePartialReuseTests(unittest.TestCase):
    def setUp(self):
        self.original_cache_instance = nodes_module._cache_instance
        self.fake_cache = _make_fake_cache_wrapper()
        self.fake_cache.register_entry(
            "怎么联系人工？",
            "您可以拨打官方客服热线 400-820-2026。",
            seed_id=7,
        )
        nodes_module._cache_instance = self.fake_cache

    def tearDown(self):
        nodes_module._cache_instance = self.original_cache_instance

    def test_compound_query_enters_partial_reuse_with_residual_query(self):
        state = {
            "query": "怎么联系人工？我买的书缺页了还能退吗？",
            "cache_enabled": True,
            "execution_path": [],
            "metrics": {},
        }

        result = check_cache_node(state)

        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["cache_match_type"], "subquery_near_exact")
        self.assertEqual(result["cache_reuse_mode"], "partial_reuse")
        self.assertEqual(result["cache_matched_question"], "怎么联系人工？")
        self.assertEqual(result["cache_residual_query"], "我买的书缺页了还能退吗")
        self.assertEqual(result["cache_rerank_attempt"], "skipped")
        self.assertEqual(result["cache_rerank_reason"], "deterministic_subquery_fastpath")


class ResidualCacheShortCircuitTests(unittest.TestCase):
    """When the residual query is itself cached at L1, research_supplement_node
    must skip execute_research entirely (llm_calls['research_llm'] == 0)."""

    def setUp(self):
        self.original_cache_instance = nodes_module._cache_instance
        self.fake_cache = _make_fake_cache_wrapper()
        # "Already answered" part — seeded so the compound check_cache_node could have hit it
        self.fake_cache.register_entry(
            "我买的书缺页了还能退吗？",
            "缺页商品支持七天无理由退换货。",
            seed_id=5,
        )
        # Residual part — also seeded; supplement node should hit this and skip RAG
        self.fake_cache.register_entry(
            "怎么联系人工？",
            "您可以拨打官方客服热线 400-820-2026。",
            seed_id=7,
        )
        nodes_module._cache_instance = self.fake_cache

    def tearDown(self):
        nodes_module._cache_instance = self.original_cache_instance

    def test_residual_hits_cache_skips_rag(self):
        """Reversed compound query: residual 怎么联系人工 is cached → no LLM call."""
        state = {
            "query": "我买的书缺页了还能退吗？怎么联系人工？",
            "cache_enabled": True,
            "cache_hit": False,
            "cache_match_type": "subquery_near_exact",
            "cache_reuse_mode": "partial_reuse",
            "cache_residual_query": "怎么联系人工",
            "cache_reranker_residual_query": "怎么联系人工",
            "cache_base_answer": "缺页商品支持七天无理由退换货。",
            "execution_path": ["cache_checked"],
            "metrics": {},
            "llm_calls": {},
            "llm_usage": None,
            "llm_usage_lock": None,
        }

        result = research_supplement_node(state)

        self.assertEqual(
            result["llm_calls"].get("research_llm", 0),
            0,
            "No LLM call should happen when residual is served from cache",
        )
        self.assertIn("400-820-2026", result["answer"])


class SubqueryAmbiguousResidualTests(unittest.TestCase):
    """When a subquery_* cache hit has an ambiguous residual (0 or >1 uncovered
    segments), check_cache_node must set cache_hit=False / cache_reuse_mode='none'
    so routing bypasses the Reranker and goes straight to full RAG."""

    def setUp(self):
        self.original_cache_instance = nodes_module._cache_instance
        self.fake_cache = _make_fake_cache_wrapper()
        # Seed a single FAQ entry — it will match as a subquery of a 3-part question.
        self.fake_cache.register_entry(
            "怎么联系人工？",
            "您可以拨打官方客服热线 400-820-2026。",
            seed_id=7,
        )
        nodes_module._cache_instance = self.fake_cache

    def tearDown(self):
        nodes_module._cache_instance = self.original_cache_instance

    def test_three_part_query_with_one_subquery_hit_clears_cache_state(self):
        """A 3-segment compound query where only 1 segment is cached leaves 2 uncovered
        segments, so _derive_deterministic_subquery_residual returns ''.
        check_cache_node must then reset cache_hit=False and cache_reuse_mode='none'."""
        state = {
            "query": "怎么联系人工？我买的书缺页了还能退吗？支持海外发货吗？",
            "cache_enabled": True,
            "execution_path": [],
            "metrics": {},
        }

        result = check_cache_node(state)

        self.assertFalse(result["cache_hit"], "cache_hit must be False when residual is ambiguous")
        self.assertEqual(result["cache_reuse_mode"], "none")
        self.assertEqual(result["cache_match_type"], "none",
                         "match_type must be reset so routing skips Reranker")


class DualSubqueryCacheHitLabelTests(unittest.TestCase):
    """When BOTH subquery parts are served from L1 cache (B1 fires), research_supplement_node
    must return cache_reuse_mode='dual_subquery' so the label reads 'Dual Subquery Cache Hit'
    rather than 'Partial Cache Reuse + RAG'."""

    def setUp(self):
        self.original_cache_instance = nodes_module._cache_instance
        self.fake_cache = _make_fake_cache_wrapper()
        # The residual part that B1 will find in cache.
        self.fake_cache.register_entry(
            "我买的书缺页了还能退吗？",
            "缺页商品属于质量问题，支持七天无理由退换货。",
            seed_id=5,
        )
        nodes_module._cache_instance = self.fake_cache

    def tearDown(self):
        nodes_module._cache_instance = self.original_cache_instance

    def test_b1_fires_sets_dual_subquery_mode(self):
        """Reversed compound Q2: residual 我买的书缺页了还能退吗 is now in L1 cache.
        B1 short-circuit fires → cache_reuse_mode must be upgraded to 'dual_subquery'."""
        state = {
            "query": "怎么联系人工？我买的书缺页了还能退吗？",
            "cache_enabled": True,
            "cache_hit": False,
            "cache_match_type": "subquery_near_exact",
            "cache_reuse_mode": "partial_reuse",
            "cache_residual_query": "我买的书缺页了还能退吗",
            "cache_reranker_residual_query": "我买的书缺页了还能退吗",
            "cache_base_answer": "您可以拨打官方客服热线 400-820-2026。",
            "execution_path": ["cache_checked"],
            "metrics": {},
            "llm_calls": {},
            "llm_usage": None,
            "llm_usage_lock": None,
        }

        result = research_supplement_node(state)

        self.assertEqual(
            result["cache_reuse_mode"],
            "dual_subquery",
            "B1 cache hit must upgrade cache_reuse_mode to 'dual_subquery'",
        )
        self.assertEqual(
            result["llm_calls"].get("research_llm", 0),
            0,
            "No RAG LLM call should happen when both subqueries are from cache",
        )
        # merge LLM must also be skipped on the dual_subquery path
        total_llm = sum(result["llm_calls"].get(k, 0) for k in result["llm_calls"])
        self.assertEqual(total_llm, 0, "No LLM calls at all (research or merge) for dual_subquery")

    def test_b1_does_not_fire_keeps_partial_reuse_mode(self):
        """When residual is NOT in cache, B1 doesn't fire → cache_reuse_mode stays 'partial_reuse'."""
        state = {
            "query": "怎么联系人工？支持海外发货吗？",
            "cache_enabled": True,
            "cache_hit": False,
            "cache_match_type": "subquery_near_exact",
            "cache_reuse_mode": "partial_reuse",
            "cache_residual_query": "支持海外发货吗",
            "cache_reranker_residual_query": "支持海外发货吗",
            "cache_base_answer": "您可以拨打官方客服热线 400-820-2026。",
            "execution_path": ["cache_checked"],
            "metrics": {},
            "llm_calls": {},
            "llm_usage": None,
            "llm_usage_lock": None,
        }

        result = research_supplement_node(state)

        self.assertEqual(
            result["cache_reuse_mode"],
            "partial_reuse",
            "cache_reuse_mode must stay 'partial_reuse' when B1 does not fire",
        )


if __name__ == "__main__":
    unittest.main()