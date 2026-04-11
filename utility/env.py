import os


def to_bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}