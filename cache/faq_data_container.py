import numpy as np
import pandas as pd

class FAQDataContainer:
    """
    FAQ 数据容器类。
    
    负责加载 FAQ 种子数据（原始库）和测试数据集（变体问题），
    并提供一套逻辑来验证缓存检索结果是否符合预期（Ground Truth）。
    """
    
    def __init__(self):
        """
        初始化数据容器，从本地 CSV 文件加载数据。
        """
        # faq_seed.csv: 原始的 FAQ 问答对（标准库）
        self.faq_df = pd.read_csv("data/faq_seed.csv")
        # test_dataset.csv: 用于测试的各种问题变体，包含预设的命中目标和命中标志
        self.test_df = pd.read_csv("data/test_dataset.csv")

        print(f"Loaded {len(self.faq_df)} FAQ entries")
        print(f"Loaded {len(self.test_df)} test queries")

    def _resolve_question(self, q):
        """
        解析一个问题，确定它在测试集定义中“应该”命中哪一个原始 FAQ 条目。

        Args:
            q: 需要解析的问题字符串。

        Returns:
            expected_hit: 预期命中的原始 FAQ 问题文本。如果预期为不命中，则返回 None。
        """
        # 在测试数据集中精确查找该问题
        matches = self.test_df[self.test_df["question"] == q]
        
        # 断言确保该测试问题在数据集中是唯一的，以便进行准确的 Ground Truth 匹配
        assert len(matches) == 1, "each hit should be matched to a src item"
        
        # 获取该测试问题对应的原始 FAQ 索引
        src_question_id = matches["src_question_id"].iloc[0]
        # 获取该测试问题是否被标记为“应该命中缓存”
        should_hit = matches["cache_hit"].iloc[0]
        
        # 从标准 FAQ 库中根据 ID 提取原始问题文本
        expected_hit = self.faq_df.iloc[src_question_id]["question"]
        
        # 如果标记为不该命中，则预期结果设为 None，否则为对应的 FAQ 文本
        expected_hit = expected_hit if should_hit else None
        return expected_hit

    def label_cache_hits(self, cache_results):
        """
        根据标准答案（Ground Truth）为缓存系统的实际检索结果打标签。
        用于后续计算 Precision、Recall 等评估指标。

        Args:
            cache_results: 语义缓存系统返回的一组 CacheResults 对象。

        Returns:
            Numpy 布尔数组，True 表示实际检索行为符合预期，False 表示不符合。
        """
        results = []
        test_qs = self.test_df["question"].tolist()
        
        for res in cache_results:
            # 1. 查找当前 Query 预期应该命中（或不命中）的原始 FAQ
            expected_hit = self._resolve_question(res.query)
            
            # 2. 获取缓存系统实际返回的最优匹配项的 Prompt 文本
            actual_hit = None if len(res.matches) == 0 else res.matches[0].prompt

            # 3. 处理间接命中逻辑：
            # 如果缓存命中的 Prompt 也是一个测试变体问题（而非原始 FAQ 文本），
            # 则需要将其溯源转换回原始 FAQ 文本，以便与 expected_hit 进行同口径对比。
            if actual_hit is not None and actual_hit in test_qs:
                actual_hit = self._resolve_question(actual_hit)

            # 4. 判断实际命中的目标是否与预期完全一致
            # (如果两者都是 None，代表预期不命中且确实没命中，也是正确的)
            results.append(expected_hit == actual_hit)

        return np.array(results)