from typing import Dict, Optional

import pandas as pd
from fuzzywuzzy import fuzz

from cache.wrapper import CacheResult, CacheResults


class FuzzyCache:
    """
    Fuzzy matching cache implementation.
    模糊匹配缓存实现。
    
    使用 fuzzywuzzy 库计算字符串之间的相似度（基于编辑距离）来匹配缓存结果。
    这种方式在处理拼写错误或细微文字变动时非常有效，但无法处理深层的语义关联。
    """
    def __init__(self):
        # 内部存储：简单的列表结构，存储 [[问题, 答案], ...]
        self.store = []

    def hydrate_from_df(
        self,
        df: pd.DataFrame,
        *,
        q_col: str = "question",
        a_col: str = "answer",
        clear: bool = True,
    ):
        """
        Load cache data from a pandas DataFrame.
        从 pandas DataFrame 加载并填充缓存数据。

        Args:
            df: 包含数据的 DataFrame。
            q_col: 问题列的名称。
            a_col: 答案列的名称。
            clear: 是否在加载新数据前清空现有缓存。
        """
        if clear:
            self.store = []
        
        idx = 0
        # 遍历 DataFrame 的每一行，提取 Q&A 对并存入 self.store
        for row in df[[q_col, a_col]].itertuples(index=False, name=None):
            q, a = row
            self.store.append([q, a])
            idx += 1

    def check_many(self, queries, distance_threshold: Optional[float] = None):
        """
        Check multiple queries against the cache using fuzzy matching.
        使用模糊匹配批量检查查询是否命中缓存。

        Args:
            queries: 待检索的查询字符串列表。
            distance_threshold: 距离阈值（0.0 到 1.0 之间）。
                               数值越小，要求匹配越精确。

        Returns:
            包含检索结果的 CacheResults 对象列表。
        """
        # 如果未设置阈值，默认设为 1（即只要相似度大于 0 就算匹配）
        distance_threshold = 1 if distance_threshold is None else distance_threshold

        results = []
        for query in queries:
            max_ratio = 0
            matched = None
            
            # 线性扫描缓存库：计算输入 query 与库中每一个 q 的相似度
            for q, a in self.store:
                # fuzz.ratio 返回 0-100 之间的整数分数
                ratio = fuzz.ratio(q, query)
                if ratio > max_ratio:
                    max_ratio = ratio
                    matched = [q, a]

            # 解析最高分匹配项
            matched_query, answer = matched
            
            # 将 0-100 的分数转换为系统统一使用的 0.0-1.0 的“距离”表示
            # 距离 = 1 - (分数 / 100)
            # 例如：分数 90 -> 距离 0.1
            calculated_distance = 1 - (max_ratio / 100.0)
            
            # 封装匹配结果
            matches = [
                CacheResult(
                    prompt=matched_query,
                    response=answer,
                    vector_distance=calculated_distance,
                    cosine_similarity=1 - calculated_distance,
                )
            ]
            
            # 如果计算出的距离超过了设定的阈值，则清空匹配项（视为未命中）
            if calculated_distance > distance_threshold:
                matches = []
                
            results.append(CacheResults(query=query, matches=matches))
            
        return results