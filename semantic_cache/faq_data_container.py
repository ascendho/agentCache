import pandas as pd

class FAQDataContainer:
    """
    FAQ 数据容器类。
    负责加载 FAQ 种子数据（用于语义缓存预热）。
    """
    
    def __init__(self):
        """
        初始化数据容器，从本地 CSV 文件加载数据。
        """
        # faq_seed.csv: 预装 FAQ 问答对（缓存种子）
        self.faq_df = pd.read_csv("data/faq_seed.csv")
        print(f"已加载 {len(self.faq_df)} 条 FAQ 问答对")