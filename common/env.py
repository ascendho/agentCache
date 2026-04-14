import os
import getpass
from dotenv import load_dotenv, find_dotenv

# ==========================================
# 环境变量解析工具
# ==========================================

def to_bool_env(name: str, default: bool = False) -> bool:
    """
    读取并安全解析布尔类型的环境变量。
    
    💡 为什么需要这个函数？
    在 .env 文件中，所有的值本质上都是“纯文本字符串”。
    如果你在 .env 里写了 ENABLE_CACHE=False，Python 读取进来其实是字符串 "False"。
    如果你直接用 if os.getenv("ENABLE_CACHE"): 它会判定为 True！（因为非空字符串即为真）。
    这个函数彻底解决了这个坑，它能精准识别各种人类常用的表示“真”的字符。
    
    Args:
        name (str): 环境变量的键名。
        default (bool): 如果没配置该环境变量，默认返回的值。
        
    Returns:
        bool: 解析后的真正布尔值 (True 或 False)。
    """
    raw = os.getenv(name)
    
    # 如果没找到这个环境变量，直接退回备用默认值
    if raw is None:
        return default
        
    # strip() 去除首尾多余空格，lower() 转为小写
    # 只要符合 "1", "true", "yes", "on" 中的任意一个，就判定为 True
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ==========================================
# 核心环境与安全密钥加载工具
# ==========================================

def load_env():
    """
    加载 .env 配置文件到当前进程的系统环境变量中。
    
    内部机制：
    find_dotenv() 会启动“向上冒泡”机制，从当前目录一直向父目录寻找 .env 文件。
    load_dotenv() 负责把找到的文件内容，正式注入到 os.environ 中。
    """
    load_dotenv(find_dotenv())

def set_ark_key():
    """
    确保火山引擎（ARK）大模型的 API Key 可用；若缺失则触发交互式安全输入。
    
    💡 极佳的安全实践：
    1. 绝对不要在代码里硬编码 API Key（防止提交到 GitHub 导致秘钥泄露）。
    2. 如果 .env 里没配，程序不会直接崩溃报错，而是弹出一个密码输入框。
    3. 使用 getpass 而不是 input()，这样用户输入 Key 时屏幕上不会显示字符，防偷窥。
    """
    # 1. 先尝试从 .env 文件加载环境变量
    load_env()
    
    # 2. 检查系统中是否已经存在名为 "ARK_API_KEY" 的变量
    if not os.getenv("ARK_API_KEY"):
        # 3. 如果没找到，触发安全输入机制，并将其临时注入到当前进程环境中
        os.environ["ARK_API_KEY"] = getpass.getpass("请输入你的 ARK API key: ")