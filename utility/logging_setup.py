import logging


def setup_logging():
    """初始化全局日志配置，统一输出格式。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("agentic-workflow")
