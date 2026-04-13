from cache.config import config
from cache.faq_data_container import FAQDataContainer
from cache.wrapper import SemanticCacheWrapper


def setup_semantic_cache():
    """初始化语义缓存，并使用 FAQ 数据进行预热。"""
    cache = SemanticCacheWrapper.from_config(config)
    data = FAQDataContainer()

    # clear 设置为 true 测试前清空旧数据，保证测试环境干净
    cache.hydrate_from_df(data.faq_df, clear=True)
    return cache
