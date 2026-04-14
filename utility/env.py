import os


def to_bool_env(name: str, default: bool = False) -> bool:
    """读取布尔环境变量。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
import os
import getpass
from dotenv import load_dotenv, find_dotenv

def load_env():
    """加载 .env 文件到进程环境变量中。"""
    load_dotenv(find_dotenv())

def set_ark_key():
    """确保 ARK_API_KEY 可用；若缺失则交互式输入。"""
    load_env()
    if not os.getenv("ARK_API_KEY"):
        os.environ["ARK_API_KEY"] = getpass.getpass("请输入你的 ARK API key: ")
