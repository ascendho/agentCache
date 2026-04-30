"""Tests for src/workflow/edges.py routers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from workflow.edges import (  # noqa: E402
    RouteTarget,
    cache_rerank_router,
    cache_router,
    pre_check_router,
)


def _state(**overrides):
    base = {
        "query": "demo",
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "none",
        "cache_reuse_mode": "none",
        "cache_rerank_score": 0.0,
    }
    base.update(overrides)
    return base


class PreCheckRouterTests(unittest.TestCase):
    def test_intercepted_goes_to_synthesize(self):
        self.assertEqual(pre_check_router(_state(intercepted=True)), RouteTarget.SYNTHESIZE_RESPONSE)

    def test_normal_goes_to_check_cache(self):
        self.assertEqual(pre_check_router(_state()), RouteTarget.CHECK_CACHE)


class CacheRouterTests(unittest.TestCase):
    def test_partial_reuse_to_supplement(self):
        s = _state(cache_reuse_mode="partial_reuse")
        self.assertEqual(cache_router(s), RouteTarget.RESEARCH_SUPPLEMENT)

    def test_exact_hit_skips_rerank(self):
        s = _state(cache_hit=True, cache_match_type="exact")
        self.assertEqual(cache_router(s), RouteTarget.SYNTHESIZE_RESPONSE)

    def test_near_exact_hit_skips_rerank(self):
        s = _state(cache_hit=True, cache_match_type="near_exact")
        self.assertEqual(cache_router(s), RouteTarget.SYNTHESIZE_RESPONSE)

    def test_edit_distance_hit_skips_rerank(self):
        s = _state(cache_hit=True, cache_match_type="edit_distance")
        self.assertEqual(cache_router(s), RouteTarget.SYNTHESIZE_RESPONSE)

    def test_semantic_candidate_goes_to_rerank(self):
        s = _state(cache_hit=True, cache_match_type="semantic")
        self.assertEqual(cache_router(s), RouteTarget.RERANK_CACHE)

    def test_no_candidate_goes_to_research(self):
        self.assertEqual(cache_router(_state()), RouteTarget.RESEARCH)


class CacheRerankRouterTests(unittest.TestCase):
    def test_full_reuse(self):
        s = _state(cache_reuse_mode="full_reuse", cache_rerank_score=0.93)
        self.assertEqual(cache_rerank_router(s), RouteTarget.SYNTHESIZE_RESPONSE)

    def test_partial_reuse(self):
        s = _state(cache_reuse_mode="partial_reuse", cache_rerank_score=0.85)
        self.assertEqual(cache_rerank_router(s), RouteTarget.RESEARCH_SUPPLEMENT)

    def test_reject_goes_to_research(self):
        s = _state(cache_reuse_mode="reject", cache_rerank_score=0.42)
        self.assertEqual(cache_rerank_router(s), RouteTarget.RESEARCH)

    def test_unknown_reuse_mode_falls_through_to_research(self):
        s = _state(cache_reuse_mode="none")
        self.assertEqual(cache_rerank_router(s), RouteTarget.RESEARCH)


class RouteTargetTests(unittest.TestCase):
    def test_constants_match_node_names(self):
        # Constants must equal the literal node names registered in graph.py.
        self.assertEqual(RouteTarget.PRE_CHECK, "pre_check")
        self.assertEqual(RouteTarget.CHECK_CACHE, "check_cache")
        self.assertEqual(RouteTarget.RERANK_CACHE, "rerank_cache")
        self.assertEqual(RouteTarget.RESEARCH, "research")
        self.assertEqual(RouteTarget.RESEARCH_SUPPLEMENT, "research_supplement")
        self.assertEqual(RouteTarget.SYNTHESIZE_RESPONSE, "synthesize_response")


if __name__ == "__main__":
    unittest.main()
