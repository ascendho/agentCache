"""Tests for the streaming completion contract in src/api/server.py."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from api.server import (  # noqa: E402
    _build_stream_ready_final_event,
    _finalize_total_latency,
    _request_disconnected,
)


def _state(**overrides):
    base = {
        "final_response": "**代理合作**\n\n请发送邮件联系我们。",
        "cache_written_prompts": ["我想做代理商，怎么和你们合作？"],
        "intercepted": False,
        "cache_hit": False,
        "cache_match_type": "none",
        "cache_reuse_mode": "none",
        "metrics": {},
        "background_threads": [object()],
    }
    base.update(overrides)
    return base


class StreamReadyFinalEventTests(unittest.TestCase):
    def test_stream_ready_final_event_does_not_wait_for_background_tasks(self):
        state = _state(cache_reuse_mode="partial_reuse")
        with patch("api.server.wait_for_background_tasks") as mocked_wait:
            with patch("api.server.time.time", return_value=10.25):
                event = _build_stream_ready_final_event(state, 10.0, state["final_response"])

        mocked_wait.assert_not_called()
        self.assertEqual(event["answer"], state["final_response"])
        self.assertEqual(event["latency_ms"], 250.0)
        self.assertEqual(event["cache_written_prompts"], state["cache_written_prompts"])
        self.assertEqual(event["label_key"], "cache_partial_reuse")
        self.assertEqual(event["label_text"], "Partial Cache Reuse + RAG")

    def test_stream_ready_final_event_leaves_total_latency_unset(self):
        state = _state()
        with patch("api.server.time.time", return_value=4.5):
            _build_stream_ready_final_event(state, 4.0, state["final_response"])

        self.assertEqual(state["metrics"], {})
        self.assertEqual(len(state["background_threads"]), 1)


class FinalizeTotalLatencyTests(unittest.TestCase):
    def test_finalize_total_latency_waits_and_persists_metric(self):
        state = _state(metrics={})
        with patch("api.server.wait_for_background_tasks") as mocked_wait:
            with patch("api.server.time.time", return_value=20.75):
                latency = _finalize_total_latency(state, 20.0)

        mocked_wait.assert_called_once_with(state)
        self.assertEqual(latency, 750.0)
        self.assertEqual(state["metrics"]["total_latency"], 750.0)

    def test_finalize_total_latency_initializes_metrics_if_missing(self):
        state = _state(metrics=None)
        with patch("api.server.wait_for_background_tasks"):
            with patch("api.server.time.time", return_value=8.01):
                latency = _finalize_total_latency(state, 8.0)

        self.assertEqual(latency, 10.0)
        self.assertEqual(state["metrics"]["total_latency"], 10.0)


class RequestDisconnectedTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_false_without_request(self):
        self.assertFalse(await _request_disconnected(None))

    async def test_returns_false_when_checker_missing(self):
        self.assertFalse(await _request_disconnected(SimpleNamespace()))

    async def test_returns_checker_value(self):
        request = SimpleNamespace(is_disconnected=AsyncMock(return_value=True))
        self.assertTrue(await _request_disconnected(request))

    async def test_checker_exception_treated_as_disconnected(self):
        request = SimpleNamespace(is_disconnected=AsyncMock(side_effect=RuntimeError("closed")))
        self.assertTrue(await _request_disconnected(request))


if __name__ == "__main__":
    unittest.main()