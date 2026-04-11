from typing import Callable, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CrossEncoder:
    """
    交叉编码器类，用于计算查询（Query）与上下文（Context/Prompt）之间的深度语义相关性。
    
    与传统的双编码器（Bi-Encoder，如向量检索）不同，交叉编码器将两个句子同时输入模型进行交互计算，
    虽然计算开销较大，但其捕捉语义匹配的精度远高于单纯的向量距离。
    """
    def __init__(self, model_name_or_path="Alibaba-NLP/gte-reranker-modernbert-base"):
        """
        初始化交叉编码器模型。
        
        Args:
            model_name_or_path: 预训练模型的路径或名称。默认使用 Alibaba 提供的 GTE 重排序模型。
        """
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # 加载序列分类模型（用于回归打分）
        # 使用 torch.float16 以减少显存占用并加速推理
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
        )
        self.model = model.eval() # 设置为评估模式，关闭 Dropout 等

    def pair_distance(self, query: str, context: str) -> float:
        """
        计算一对文本之间的语义距离（1 - 相关性分数）。
        分数越小，代表语义越接近。
        """
        return 1 - self.predict([query], [context])[0]

    def predict(self, queries: List[str], contexts: List[str]) -> List[float]:
        """
        对查询-上下文对进行直接的交叉编码预测。

        Args:
            queries: 查询字符串列表。
            contexts: 上下文字符串列表（长度必须与 queries 相同）。

        Returns:
            每个对之间的相似度分数列表 [0.0-1.0]。
        """
        # 将查询和上下文配对
        pairs = list(zip(queries, contexts))
        # 对文本对进行编码，设置截断和填充以适应模型输入
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        
        # 禁用梯度计算以提高推理性能
        with torch.no_grad():
            # 获取模型输出（Logits）并转为 float32
            outputs = self.model(**inputs, return_dict=True).logits.view(-1).float()
        
        # 使用 Sigmoid 函数将 Logits 映射到 0.0 - 1.0 的概率空间（代表相关度）
        probs = torch.sigmoid(outputs).numpy()
        return probs.tolist()

    def create_reranker(self):
        """工厂方法：创建一个专门用于语义缓存集成重排序的对象。"""
        return CrossEncoderReranker(self)

class CrossEncoderReranker:
    """
    交叉编码器重排序器，专门用于集成到语义缓存的工作流中。
    """
    def __init__(self, cross_encoder: CrossEncoder):
        """
        初始化重排序器。
        
        Args:
            cross_encoder: 基础 CrossEncoder 实例。
        """
        self.cross_encoder = cross_encoder

    def __call__(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        用于语义缓存集成的重排序函数。
        当向量检索给出多个候选缓存结果时，使用交叉编码器对这些结果进行精确定分并排序。

        Args:
            query: 用户的搜索查询。
            candidates: 缓存候选结果字典列表（通常包含 'prompt' 字段）。

        Returns:
            经过过滤、重排序并附加了 Cross-Encoder 元数据的候选结果列表。
        """
        if not candidates:
            return []

        # 提取所有候选缓存的原始 Prompt
        prompts = [c.get("prompt", "") for c in candidates]

        # 将当前 query 与所有候选 Prompt 配对进行预测
        # 例如 query="A" 而候选有 ["B", "C"]，则预测 ["A", "B"] 和 ["A", "C"] 的得分
        scores = self.cross_encoder.predict([query] * len(prompts), prompts)

        # 构建验证后的候选结果列表，并注入重排序相关的指标
        validated_candidates = [
            (
                {
                    **candidate,
                    "reranker_type": "cross_encoder",      # 标记使用的重排序类型
                    "reranker_score": float(score),        # 原始分数
                    "reranker_distance": 1 - float(score), # 转换后的语义距离
                },
                score,
            )
            for candidate, score in zip(candidates, scores)
        ]
        
        # 按得分从高到低排序（分数越高表示与用户当前问题越相似）
        validated_candidates.sort(key=lambda x: x[1], reverse=True)

        # 返回排序后的结果列表
        return [candidate for candidate, _ in validated_candidates]