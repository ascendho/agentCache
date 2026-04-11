import jaro
import numpy as np
import pandas as pd
import seaborn as sns
from fuzzywuzzy import fuzz
from matplotlib import colormaps
from matplotlib import pyplot as plt

def pprint_confusion_matrix(confusion_matrix):
    """
    漂亮地打印混淆矩阵（Confusion Matrix）。
    
    该函数将原始的混淆矩阵数组转换为一个带颜色背景和 HTML 样式的 Pandas 表格，
    非常适合在 Jupyter Notebook 中展示。

    Args:
        confusion_matrix: 一个 2x2 的 numpy 数组，包含 [TN, FP, FN, TP]。
    """
    # 将混淆矩阵展平并解构出四个核心指标
    TN, FP, FN, TP = confusion_matrix.ravel()
    counts = np.array([[TN, FP], [FN, TP]])
    # 计算归一化比例（0.0 到 1.0）
    normalized = counts / counts.sum()
    # 定义标签描述
    desc = [["TN", "FP"], ["FN", "TP"]]

    def format_html(description, count, norm):
        """格式化单个单元格的 HTML 内容：包含标签名、计数值和百分比。"""
        return (
            f"<span style='font-size:smaller'>{description}</span>"
            f"<br><b>{count}</b><br><span style='font-size:smaller'>{norm:.2f}</span>"
        )

    # 生成 HTML 单元格矩阵
    html_cells = [
        [format_html(desc[i][j], counts[i][j], normalized[i][j]) for j in range(2)]
        for i in range(2)
    ]

    # 创建 Pandas DataFrame，设置行索引（真实标签 GT）和列索引（预测标签 Pred）
    df = pd.DataFrame(
        html_cells,
        index=["GT: ❌", "GT: ✅"],
        columns=["Pred: ❌", "Pred: ✅"],
    )

    # --- 样式处理：根据数值大小自动设置背景颜色 ---
    flat_norm = normalized.flatten()
    cmap = colormaps["viridis"]  # 使用 viridis 颜色映射
    color_map = cmap(flat_norm)  # 获取对应的 RGBA 颜色值

    def rgba_to_rgb_str(rgba):
        """将 RGBA 元组转换为 CSS 可用的 rgb 字符串。"""
        r, g, b, _ = rgba
        return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"

    def get_text_color(rgba):
        """
        根据背景颜色的亮度自动决定文字颜色（黑色或白色）。
        使用了 W3C 的亮度计算公式。
        """
        r, g, b, _ = rgba
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.5 else "white"

    # 生成颜色和文字颜色列表
    hex_colors = [rgba_to_rgb_str(c) for c in color_map]
    text_colors = [get_text_color(c) for c in color_map]

    # 重新整理成 2x2 结构以便应用样式
    cell_colors = np.array(hex_colors).reshape(2, 2)
    cell_text_colors = np.array(text_colors).reshape(2, 2)

    def style_func(x):
        """Pandas Styler 的样式应用函数。"""
        df_styles = pd.DataFrame("", index=x.index, columns=x.columns)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                bg_color = cell_colors[i, j]
                txt_color = cell_text_colors[i, j]
                df_styles.iloc[i, j] = (
                    f"background-color: {bg_color}; color: {txt_color}; text-align: center;"
                )
        return df_styles

    # 应用样式并生成最终的 Styler 对象
    styled_df = (
        df.style.set_table_attributes(
            "style='border-collapse:collapse; font-family:sans-serif'"
        )
        .set_properties(**{"text-align": "center"})
        .apply(style_func, axis=None)
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("font-size", "14px"), ("text-align", "center")],
                }
            ]
        )
    )
    return styled_df


def pprint_metrics(threshold, metrics, metrics_to_print):
    """
    以表格形式展示指定阈值下的各项评估指标。

    Args:
        threshold: 语义判定阈值。
        metrics: 包含各项指标数据的字典。
        metrics_to_print: 需要展示的指标名称列表。
    """
    # 构建数据行：阈值 + 各项指标值
    print_metrics = [[threshold] + [metrics[k].item() for k in metrics_to_print]]
    
    # 转换为转置后的 DataFrame，以便垂直展示各项指标
    df = pd.DataFrame(
        print_metrics,
        columns=["Threshold"] + [m.replace("_", " ").title() for m in metrics_to_print],
    ).T
    df.columns = [""] # 隐藏列名
    return df


def plot_metrics(thresholds, metrics, best_threshold, metrics_to_plot=[]):
    """
    绘制阈值扫描曲线图。
    
    展示随着“距离阈值”的变化，精确率、召回率、F1 分数等指标的动态趋势，
    并标注出最优阈值的位置。

    Args:
        thresholds: X 轴数据，一系列距离阈值。
        metrics: Y 轴数据，包含各项指标数值序列的字典。
        best_threshold: 在图中绘制垂直虚线标注的最优阈值。
        metrics_to_plot: 需要绘制在图中的指标名称列表。
    """
    # 创建画布
    fig, ax = plt.subplots(dpi=200, figsize=(8, 4))
    
    # 使用 Seaborn 绘制每一条指标曲线
    for metric_name in metrics_to_plot:
        if metric_name in metrics:
            sns.lineplot(
                x=thresholds,
                y=metrics[metric_name],
                label=metric_name.replace("_", " ").title(),
            )

    # 绘制垂直线标注最优阈值位置
    plt.axvline(
        best_threshold,
        color="tab:blue",
        linewidth=1,
        linestyle="--",  # 添加虚线样式增强辨识度
    )

    # 设置图表元数据
    plt.xlabel("Distance Threshold")
    plt.ylabel("Metric Value")
    plt.title("Threshold Sweep Analysis")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 关闭当前的 pyplot 引用，防止在非交互环境中产生多余输出
    plt.close()

    return fig