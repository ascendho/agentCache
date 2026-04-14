# 从语义缓存配置模块导入预定义的配置对象（包含 Redis 连接、阈值、模型名等）
from semantic_cache.config import config
# 导入 FAQ 数据容器类，用于加载和管理常见问题集（通常从 CSV 或数据库读取）
from semantic_cache.faq_data_container import FAQDataContainer
# 导入语义缓存核心包装类，负责实现基于向量相似度的查询与存储逻辑
from semantic_cache.engine import SemanticCacheWrapper


def setup_semantic_cache():
    """
    初始化语义缓存系统，并使用预定义的 FAQ 数据进行索引预热。
    
    语义缓存不同于传统的键值缓存，它通过计算输入问题的向量相似度，
    在用户提出“意思相近”的问题时即可直接命中缓存。
    
    Returns:
        SemanticCacheWrapper: 初始化完成并加载了数据的缓存实例。
    """
    
    # 1. 根据 config 配置信息创建语义缓存实例
    # 这步通常会初始化向量模型（如嵌入模型）并建立与 Redis 的连接
    cache = SemanticCacheWrapper.from_config(config)
    
    # 2. 实例化 FAQ 数据容器
    # 该容器内部通常维护着一个包含 'question' 和 'answer' 列的 Pandas DataFrame
    data = FAQDataContainer()

    # 3. 将 FAQ 数据“灌入”缓存（Hydrate / 预热）
    # hydrate_from_df 会遍历 DataFrame 中的每一行，将其问题文本转为向量并存入 Redis。
    # 
    # 参数说明：
    # - data.faq_df: 包含原始问答对的数据帧。
    # - clear=True:  在加载前清空 Redis 中旧的缓存数据。
    #                在测试环境或数据更新时设置为 True，确保缓存内容与最新的 FAQ 文档完全同步。
    cache.hydrate_from_df(data.faq_df, clear=True)
    
    # 返回配置妥当的缓存对象，后续可调用 cache.check(query) 来检索
    return cache