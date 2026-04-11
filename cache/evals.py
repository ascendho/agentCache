import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import panel as pn

from cache.wrapper import CacheResults

try:
    import tiktoken
except ImportError:
    tiktoken = None

from cache.vis import plot_metrics, pprint_confusion_matrix

pn.extension()


def _to_float_env(name: str, default: float) -> float:
    """从环境变量读取浮点数，解析失败时返回默认值。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_display_currency() -> str:
    """获取成本展示币种（当前固定 CNY）。"""
    return "CNY"


def provider_base_currency(provider: str) -> str:
    """返回 provider 计费表的基础币种（当前固定 CNY）。"""
    return "CNY"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> float:
    """币种转换（当前统一 CNY，不做换算）。"""
    if from_currency == to_currency:
        return amount
    return amount


def currency_symbol(currency: str) -> str:
    """返回币种符号。"""
    return "¥" if currency == "CNY" else "$"


def format_cost(value: float, currency: Optional[str] = None) -> str:
    """格式化金额字符串，自动按币种展示符号与精度。"""
    c = (currency or get_display_currency()).upper()
    sym = currency_symbol(c)
    if value < 0.001:
        return f"{sym}{value:.6f}"
    if value < 1:
        return f"{sym}{value:.4f}"
    return f"{sym}{value:.2f}"


def load_model_costs() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    从本地 JSON 文件加载模型计费信息。

    Returns:
        结构为 provider -> model -> {input, output} 的字典。
        其中 input/output 代表“每 1K token 的成本（人民币）”。
    """
    try:
        # 以当前文件路径为基准定位 model_costs.json，避免工作目录变化导致找不到文件。
        current_dir = os.path.dirname(os.path.abspath(__file__))
        costs_file = os.path.join(current_dir, "model_costs.json")

        with open(costs_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # 降级策略：当文件不存在或格式损坏时，返回可用的默认 CNY 费率。
        return {
            "volcengine": {
                "deepseek-v3-2-251201": {"input": 0.003, "output": 0.0045},
                "ep-m-20260411093114-9hftc": {"input": 0.0012, "output": 0.0072},
            }
        }


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    统计文本 token 数量。

    优先使用 tiktoken 精确计数；若环境未安装或编码获取失败，
    则采用“按单词数近似折算”的方式兜底。

    Args:
        text: 待统计文本。
        model: 模型名，用于选择对应编码规则（默认 gpt-4o-mini）。

    Returns:
        文本 token 数。
    """
    if tiktoken is None:
        # 经验近似：英文场景下 token 数通常略高于单词数。
        return len(text.split()) * 1.3

    try:
        # 不同模型家族可能使用不同编码；这里做前缀匹配选择编码。
        model_encodings = {
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base",
            "gpt-4": "cl100k_base",
            "gpt-4-turbo": "cl100k_base",
            "gpt-3.5-turbo": "cl100k_base",
            "claude": "cl100k_base",  # 近似映射，仅用于估算
            "gemini": "cl100k_base",  # 近似映射，仅用于估算
        }

        encoding_name = "o200k_base"  # 对较新模型使用该编码作为默认值
        for model_prefix, enc in model_encodings.items():
            if model_prefix in model.lower():
                encoding_name = enc
                break

        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))

    except Exception:
        # 任何异常都回退近似算法，确保调用方总能拿到可用结果。
        return int(len(text.split()) * 1.3)


def get_model_cost(provider: str, model: str) -> Dict[str, float]:
    """
    获取指定模型的每 1K token 费率。

    Args:
        provider: 提供商名称（如 openai、anthropic）。
        model: 模型名称（如 gpt-4o-mini）。

    Returns:
        含 input/output 费率的字典。
    """
    costs = load_model_costs()

    if provider in costs and model in costs[provider]:
        return costs[provider][model]

    # 兼容某些调用方 provider 传错或为空的情况：尝试跨 provider 查找模型名。
    for p, models in costs.items():
        if model in models:
            return models[model]

    # 最后兜底默认值，避免 KeyError 或返回空结构。
    return {"input": 0.001, "output": 0.002}


def _harmonic_mean(a, b):
    # 调和平均值更强调“短板效应”，适合把两个指标做平衡评分。
    if a + b == 0:
        return 0
    return 2 * a * b / (a + b) if (a + b) > 0 else 0


class CacheEvaluator:
    true_labels: List[bool]
    cache_results: List[CacheResults]

    is_from_full_retrieval: bool

    @classmethod
    def from_full_retrieval(cls, true_labels, cache_results) -> "CacheEvaluator":
        return cls(true_labels, cache_results, is_from_full_retrieval=True)

    def __init__(
        self, true_labels, cache_results, is_from_full_retrieval: bool = False
    ):
        self.true_labels = np.array(true_labels)
        self.cache_results = np.array(cache_results)
        self.is_from_full_retrieval = is_from_full_retrieval

    def matches_df(self) -> pd.DataFrame:
        # 将评估结果扁平化为 DataFrame，方便排查单条 query 的匹配情况。
        query = [r.query for r in self.cache_results]
        match = [
            r.matches[0].prompt if len(r.matches) > 0 else None
            for r in self.cache_results
        ]
        distance = [
            r.matches[0].vector_distance if len(r.matches) > 0 else None
            for r in self.cache_results
        ]
        true_label = self.true_labels.tolist()

        return pd.DataFrame(
            {
                "query": query,
                "match": match,
                "distance": distance,
                "true_label": true_label,
            }
        )

    def get_metrics(self, distance_threshold: Optional[float] = None):
        # 阈值语义：只保留向量距离 < T 的候选。
        # 当 T=None 时使用 1（基本不过滤）。
        T = 1 if distance_threshold is None else distance_threshold

        has_retrieval = np.array(
            [
                len([m for m in it.matches if m.vector_distance < T]) > 0
                for it in self.cache_results
            ]
        )
        true_labels = np.array(self.true_labels)
        if self.is_from_full_retrieval:
            # 在 full retrieval 模式下，若阈值过滤后无召回，会改变标签解释方式：
            # 这一步用于修正“无候选”带来的统计偏差。
            true_labels[~has_retrieval] = ~true_labels[~has_retrieval]

        # 这里采用“是否有检索结果”作为预测正负类的依据来构造混淆矩阵。
        tp = has_retrieval & true_labels
        tn = (~has_retrieval) & true_labels
        fp = has_retrieval & (~true_labels)
        fn = (~has_retrieval) & (~true_labels)

        TP = sum(tp)
        FP = sum(fp)
        FN = sum(fn)
        TN = sum(tn)

        confusion_matrix = np.array([[TN, FP], [FN, TP]])
        # cache_hit_rate 定义为“有返回候选”的比例（不区分真伪命中）。
        cache_hit_rate = (TP + FP) / (TP + FP + FN + TN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1

        return {
            "cache_hit_rate": cache_hit_rate,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * TP / (2 * TP + FP + FN),
            "accuracy": (TP + TN) / (TP + TN + FP + FN),
            "utility": _harmonic_mean(precision, cache_hit_rate),
            # confusion_mask 保存逐样本布尔掩码，便于后续定位误判样本。
            "confusion_matrix": confusion_matrix,
            "confusion_mask": np.array([[tn, fp], [fn, tp]]),
        }

    def report_threshold_sweep(
        self,
        metric_to_maximize="f1_score",
        threshold_span=(0, 1),
        num_samples=100,
        metrics_to_plot=[
            "cache_hit_rate",
            "precision",
            "recall",
            "f1_score",
        ],
    ):
        # 在阈值区间内扫描多个采样点，寻找目标指标最优阈值。
        thresholds = []
        all_metrics = {}
        for threshold in np.linspace(*threshold_span, num_samples):
            threshold = threshold.item()
            metrics = self.get_metrics(threshold)
            thresholds.append(threshold)
            for key, metric in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(metric)

        thresholds = np.array(thresholds)
        all_metrics = {key: np.array(value) for key, value in all_metrics.items()}

        best_index = np.argmax(all_metrics[metric_to_maximize])
        best_threshold = thresholds[best_index].item()

        best_metrics_report = self.report_metrics(
            distance_threshold=best_threshold,
            title=f"Best Threshold Metrics @T={best_threshold:.4f}",
            orientation="vertical",
        )

        return pn.Row(
            best_metrics_report,
            pn.pane.Matplotlib(
                plot_metrics(
                    thresholds,
                    all_metrics,
                    best_threshold,
                    metrics_to_plot=metrics_to_plot,
                ),
                format="svg",
                tight=True,
                height=420,
            ),
        )

    def report_metrics(
        self,
        distance_threshold: Optional[float] = None,
        title="Evaluation report",
        orientation="horizontal",
    ):
        # 统一输出指标表与混淆矩阵，便于横向对比不同阈值。
        metrics = self.get_metrics(distance_threshold)

        metrics_table = (
            pd.DataFrame([metrics])
            .drop(columns=["confusion_matrix", "confusion_mask"])
            .T.rename(columns={0: ""})
        )
        container = pn.Row
        if orientation == "vertical":
            container = pn.Column

        return pn.Column(
            f"### {title}",
            container(
                pn.pane.DataFrame(metrics_table, width=200),
                pprint_confusion_matrix(metrics["confusion_matrix"]),
            ),
        )


class PerfEval:
    def __init__(self):
        # durations 记录每个阶段耗时（秒）。
        self.durations = []
        self.durations_by_label: Dict[str, List[float]] = {}
        self.last_time: Optional[float] = None
        self.total_queries: Optional[int] = None
        # llm_calls 中每项结构：{model, provider, in, out}
        self.llm_calls: List[Dict] = []

    def __enter__(self):
        # 进入上下文时清空旧状态，适合一次完整实验用一次上下文。
        self.last_time = time.time()
        self.durations = []
        self.durations_by_label = {}
        self.llm_calls = []
        return self

    def start(self):
        self.last_time = time.time()

    def tick(self, label: Optional[str] = None):
        # tick 记录“从上一次 start/tick 到当前”的时间差。
        now = time.time()
        if self.last_time is None:
            self.last_time = now
        dt = now - self.last_time
        self.durations.append(dt)
        if label:
            # 通过标签拆分耗时，有助于区分 cache_hit、llm_call 等阶段。
            self.durations_by_label.setdefault(label, []).append(dt)
        self.last_time = now

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def set_total_queries(self, n: int):
        # 记录查询总数，用于计算 avg_cost_per_query 等全局均值指标。
        self.total_queries = n

    def record_llm_call(
        self,
        model: str,
        input_text: str = "",
        output_text: str = "",
        provider: str = "openai",
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ):
        """
        记录一次 LLM 调用，并自动统计输入/输出 token。

        Args:
            model: 模型名称（如 "gpt-4o-mini"）。
            input_text: 输入文本（当未直接给 input_tokens 时用于估算）。
            output_text: 输出文本（当未直接给 output_tokens 时用于估算）。
            provider: 模型提供商名，用于查询费率（默认 "openai"）。
            input_tokens: 直接指定输入 token（优先级高于 input_text 估算）。
            output_tokens: 直接指定输出 token（优先级高于 output_text 估算）。
        """
        if input_tokens is None:
            input_tokens = count_tokens(input_text, model)
        if output_tokens is None:
            output_tokens = count_tokens(output_text, model)

        self.llm_calls.append(
            {
                "model": model,
                "provider": provider,
                "in": input_tokens or 0,
                "out": output_tokens or 0,
            }
        )

    def _stats(self, values: List[float]):
        # 统一统计口径：返回均值、分位数与平均吞吐（qps）。
        if len(values) == 0:
            return {
                "count": 0,
                "average_latency": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "average_throughput": 0.0,
            }
        arr = np.array(values, dtype=float)
        total_ms = arr.sum() * 1000.0
        return {
            "count": int(arr.size),
            "average_latency": float(arr.mean() * 1000.0),
            "p50": float(np.percentile(arr, 50) * 1000.0),
            "p90": float(np.percentile(arr, 90) * 1000.0),
            "p95": float(np.percentile(arr, 95) * 1000.0),
            "p99": float(np.percentile(arr, 99) * 1000.0),
            "average_throughput": float(
                (arr.size / (total_ms / 1000.0)) if total_ms > 0 else 0.0
            )
            * 1000.0,
        }

    def get_metrics(self, labels: Optional[List[str]] = None):
        overall = self._stats(self.durations)
        by_label = {}
        if labels:
            for lbl in labels:
                by_label[lbl] = self._stats(self.durations_by_label.get(lbl, []))
        return {"overall": overall, "by_label": by_label}

    def get_costs(self, target_currency: Optional[str] = None):
        """
        汇总所有 LLM 调用成本，并给出分模型明细。

        Returns:
            成本统计字典（总成本、分模型成本、均次成本等）。
        """
        target_currency = "CNY"
        all_costs = load_model_costs()
        total = 0.0
        by_model: Dict[str, float] = {}

        for call in self.llm_calls:
            model = call["model"]
            provider = call.get("provider", "openai")
            rates = get_model_cost(provider, model)
            base_currency = provider_base_currency(provider)

            # 成本换算：token 数 / 1000 * 单价。
            input_cost_base = (call["in"] / 1000.0) * rates.get("input", 0.0)
            output_cost_base = (call["out"] / 1000.0) * rates.get("output", 0.0)
            call_cost_base = input_cost_base + output_cost_base
            call_cost = convert_currency(call_cost_base, base_currency, target_currency)

            by_model[model] = by_model.get(model, 0.0) + call_cost
            total += call_cost

        result = {
            "total_cost": total,
            "by_model": by_model,
            "calls": len(self.llm_calls),
            "currency": target_currency,
            "display_symbol": currency_symbol(target_currency),
        }

        if self.total_queries:
            result["avg_cost_per_query"] = total / self.total_queries
        if self.llm_calls:
            result["avg_cost_per_call"] = total / len(self.llm_calls)

        return result

    def plot(
        self,
        labels: Optional[List[str]] = None,
        title: str = "Performance Dashboard",
        figsize: tuple = (14, 8),
        show_cost_analysis: bool = True,
    ):
        """
        绘制综合性能看板（命中分布、时延对比、成本、摘要）。

        Args:
            labels: 需要纳入分析的耗时标签；为空时自动从已记录标签推断。
            title: 图表标题。
            figsize: 画布大小（宽, 高）。
            show_cost_analysis: 是否显示成本面板。
        """
        if labels is None:
            labels = list(self.durations_by_label.keys())

        metrics = self.get_metrics(labels=labels)
        costs = self.get_costs()

        # 统一图表风格，保证不同实验图的观感一致。
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10}
        )

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        self._plot_hit_miss_distribution(ax1, labels, metrics)
        self._plot_latency_comparison(ax2, labels, metrics)

        if show_cost_analysis and costs["calls"] > 0:
            self._plot_cost_analysis(ax3, costs)
        else:
            ax3.axis("off")

        self._plot_performance_summary(ax4, labels, metrics, costs)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _plot_hit_miss_distribution(self, ax, labels, metrics):
        """绘制缓存命中/未命中分布饼图。"""
        cache_hits = 0
        cache_misses = 0

        for label in labels:
            count = metrics["by_label"].get(label, {}).get("count", 0)

            # 约定：标签名含 hit 视为命中，含 llm 视为未命中后触发大模型。
            if "hit" in label.lower():
                cache_hits += count
            elif "llm" in label.lower():
                cache_misses += count

        if cache_hits == 0 and cache_misses == 0:
            # 若没有标签可推断，则退化为“全部视作 miss”。
            total_queries = self.total_queries or len(self.durations)
            cache_misses = total_queries

        if cache_hits + cache_misses > 0:
            sizes = [cache_hits, cache_misses]
            labels_pie = [
                f"Cache Hits\n({cache_hits})",
                f"Cache Misses\n({cache_misses})",
            ]
            colors = ["#2ecc71", "#e74c3c"]
            explode = (0.02, 0) if cache_hits > 0 else (0, 0.02)

            wedges, texts, autotexts = ax.pie(
                sizes,
                explode=explode,
                labels=labels_pie,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 10},
            )

            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontweight("bold")

            ax.set_title("Cache Effectiveness", fontweight="bold", pad=20)
        else:
            ax.text(
                0.5,
                0.5,
                "No hit/miss data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Cache Effectiveness", fontweight="bold")

    def _plot_latency_comparison(self, ax, labels, metrics):
        """绘制关键阶段时延对比柱状图。"""
        latency_data = []
        latency_labels = []
        colors = []

        # 只保留有数据的标签，避免画空柱。
        relevant_labels = []
        for label in labels:
            if label in metrics["by_label"] and metrics["by_label"][label]["count"] > 0:
                if "hit" in label.lower() or "llm" in label.lower():
                    relevant_labels.append(label)

        for label in relevant_labels:
            latency = metrics["by_label"][label]["average_latency"]
            latency_data.append(latency)

            if "hit" in label.lower():
                latency_labels.append(f"Cache Hit\n{latency:.1f}ms")
                colors.append("#2ecc71")
            elif "llm" in label.lower():
                latency_labels.append(f"LLM Call\n{latency:.1f}ms")
                colors.append("#e74c3c")
            else:
                latency_labels.append(
                    f'{label.replace("_", " ").title()}\n{latency:.1f}ms'
                )
                colors.append("#95a5a6")

        if latency_data:
            bars = ax.bar(latency_labels, latency_data, color=colors, alpha=0.8)
            ax.set_title("Response Time Comparison", fontweight="bold", pad=20)
            ax.set_ylabel("Latency (ms)", fontweight="bold")

            ax.grid(True, alpha=0.3, axis="y")
            ax.set_axisbelow(True)

            ax.set_ylim(0, max(latency_data) * 1.1)
        else:
            ax.text(
                0.5,
                0.5,
                "No latency data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Response Time Comparison", fontweight="bold")

    def _plot_cost_analysis(self, ax, costs):
        """绘制成本分析柱状图（单次查询与总成本）。"""
        cost_data = [costs.get("avg_cost_per_query", 0), costs.get("total_cost", 0)]

        avg_cost_str = (
            f"${cost_data[0]:.6f}" if cost_data[0] < 0.001 else f"${cost_data[0]:.4f}"
        )
        total_cost_str = (
            f"${cost_data[1]:.6f}" if cost_data[1] < 0.001 else f"${cost_data[1]:.4f}"
        )

        cost_labels = [f"Per Query\n{avg_cost_str}", f"Total Cost\n{total_cost_str}"]

        bars = ax.bar(cost_labels, cost_data, color=["#3498db", "#e74c3c"], alpha=0.8)
        ax.set_title("Cost Analysis", fontweight="bold", pad=20)
        ax.set_ylabel("Cost (USD)", fontweight="bold")

        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        if max(cost_data) > 0:
            ax.set_ylim(0, max(cost_data) * 1.1)

    def _plot_performance_summary(self, ax, labels, metrics, costs):
        """在文本面板中汇总关键性能指标。"""
        ax.axis("off")

        total_queries = self.total_queries or len(self.durations)

        cache_hits = 0
        llm_calls = 0
        cache_hit_latency = 0
        llm_call_latency = 0

        for label in labels:
            label_data = metrics["by_label"].get(label, {})
            count = label_data.get("count", 0)

            if "hit" in label.lower():
                cache_hits += count
                if cache_hit_latency == 0:
                    cache_hit_latency = label_data.get("average_latency", 0)
            elif "llm" in label.lower():
                llm_calls += count
                if llm_call_latency == 0:
                    llm_call_latency = label_data.get("average_latency", 0)

        hit_rate = (cache_hits / total_queries * 100) if total_queries > 0 else 0

        if cache_hits > 0 and llm_calls > 0 and cache_hit_latency > 0:
            speed_improvement = llm_call_latency / cache_hit_latency
            speed_text = f"{speed_improvement:.1f}x faster"
        else:
            speed_text = "N/A"

        summary_text = f"""Performance Summary

Queries: {total_queries}
Cache Hits: {cache_hits} ({hit_rate:.1f}%)
Cache Misses: {llm_calls}

Average Latency: {metrics['overall']['average_latency']:.1f}ms
Cache Speedup: {speed_text}"""

        if costs["calls"] > 0:
            summary_text += f"""

Total Cost: ${costs['total_cost']:.4f}
Cost/Query: ${costs.get('avg_cost_per_query', 0):.6f}"""

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.7",
                facecolor="#f8f9fa",
                edgecolor="#dee2e6",
                linewidth=1,
                alpha=0.9,
            ),
        )
