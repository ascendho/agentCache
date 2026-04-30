"""测试场景加载器：从 data/ 目录读取扁平的 query 数组。"""

import json
import os

# 加载独立的测试数据文件
_profile = os.getenv("TEST_SCENARIO_PROFILE", "debug").strip().lower()
_data_filename = "test_scenarios_full.json" if _profile == "full" else "test_scenarios.json"
_data_path = os.path.join(os.path.dirname(__file__), "..", "data", _data_filename)

try:
    with open(_data_path, "r", encoding="utf-8") as f:
        _test_data = json.load(f)
except FileNotFoundError:
    print(f"⚠️ 测试数据文件缺失: {_data_path}")
    _test_data = []


def _normalize_query_item(item, position: int) -> str:
    """把单条记录规范成 query 字符串；支持纯字符串与 {'query': str} 两种形式。"""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        query = item.get("query", "")
        if isinstance(query, str):
            return query.strip()
        raise ValueError(
            f"{_data_filename} 第 {position} 条记录的 'query' 字段必须是字符串，"
            f"实际为 {type(query).__name__}"
        )
    raise ValueError(
        f"{_data_filename} 第 {position} 条记录格式不支持："
        f"应为字符串或 {{'query': str}}，实际为 {type(item).__name__}"
    )


def _iter_queries(data):
    if not isinstance(data, list):
        raise ValueError(
            f"{_data_filename} 顶层结构必须是数组，实际为 {type(data).__name__}。"
            "请使用扁平的 query 列表格式。"
        )
    for position, item in enumerate(data, 1):
        query = _normalize_query_item(item, position)
        if query:
            yield query


SCENARIO_RUNS = [
    {"title": f"测试{index:02d}", "query": query}
    for index, query in enumerate(_iter_queries(_test_data), 1)
]
