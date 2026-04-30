"""Tests for tests.scenarios — fail-fast loader behaviour."""

import unittest

from tests import scenarios
from tests.scenarios import SCENARIO_RUNS, _iter_queries, _normalize_query_item


class NormalizeQueryItemTests(unittest.TestCase):
    def test_string_item(self):
        self.assertEqual(_normalize_query_item("  hi  ", 1), "hi")

    def test_dict_item_with_query(self):
        self.assertEqual(_normalize_query_item({"query": "  abc  "}, 1), "abc")

    def test_dict_item_missing_query_returns_empty(self):
        # Empty string is treated as falsy by the iterator and skipped, not raised
        self.assertEqual(_normalize_query_item({"query": ""}, 1), "")

    def test_dict_item_non_string_query_raises(self):
        with self.assertRaises(ValueError):
            _normalize_query_item({"query": 123}, 5)

    def test_unsupported_item_type_raises(self):
        with self.assertRaises(ValueError):
            _normalize_query_item(42, 3)
        with self.assertRaises(ValueError):
            _normalize_query_item(None, 9)


class IterQueriesTests(unittest.TestCase):
    def test_non_list_top_level_raises(self):
        with self.assertRaises(ValueError):
            list(_iter_queries({"items": []}))
        with self.assertRaises(ValueError):
            list(_iter_queries("flat string"))

    def test_skips_empty_strings(self):
        out = list(_iter_queries(["a", "", {"query": "  "}, "b"]))
        self.assertEqual(out, ["a", "b"])

    def test_mixed_string_and_dict(self):
        out = list(_iter_queries(["a", {"query": "b"}]))
        self.assertEqual(out, ["a", "b"])


class ScenarioRunsTests(unittest.TestCase):
    def test_scenario_runs_shape(self):
        # In CI the real data file should be present; verify shape only.
        self.assertIsInstance(SCENARIO_RUNS, list)
        for entry in SCENARIO_RUNS:
            self.assertIn("title", entry)
            self.assertIn("query", entry)
            self.assertTrue(entry["title"].startswith("测试"))
            self.assertIsInstance(entry["query"], str)
            self.assertGreater(len(entry["query"]), 0)

    def test_titles_are_unique_and_sequential(self):
        titles = [entry["title"] for entry in SCENARIO_RUNS]
        self.assertEqual(len(titles), len(set(titles)))
        for index, title in enumerate(titles, 1):
            self.assertEqual(title, f"测试{index:02d}")

    def test_data_filename_constant(self):
        # _data_filename should be one of the two known names
        self.assertIn(scenarios._data_filename, {"test_scenarios.json", "test_scenarios_full.json"})


if __name__ == "__main__":
    unittest.main()
