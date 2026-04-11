import getpass
import logging
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from cache.config import config

# 配置日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv, find_dotenv                                  
def load_env():
    """加载 .env 文件到进程环境变量中，统一管理 API Key 等环境变量。"""
    _ = load_dotenv(find_dotenv())

def set_ark_key():
    """确保 ARK_API_KEY（火山引擎/方舟）可用；若缺失则交互式输入。"""
    load_env()
    if not os.getenv("ARK_API_KEY"):
        api_key = getpass.getpass("Enter your ARK API key: ")
        os.environ["ARK_API_KEY"] = api_key

class SimilarityResult(BaseModel):
    """定义 LLM 返回的结构化 schema 数据模型。"""
    reason: str = Field(description="解释为什么两个句子意思相同或不同")
    is_similar: bool = Field(
        description="如果两个句子意思相同则为 True，否则为 False"
    )

def batch_iterable(iterable, batch_size):
    """按批次切分可迭代对象，避免一次性请求数据量过大导致 OOM 或超时。"""
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]

@dataclass(frozen=True)
class LLMEvaluationResult:
    """包装评估结果列表，并提供转换为 Pandas DataFrame 的便捷方法。"""
    resulting_items: List[SimilarityResult]

    @property
    def df(self):
        """将结果列表转换为 DataFrame，便于分析与导出。"""
        return pd.DataFrame([dict(it) for it in self.resulting_items])


# 默认的判断提示词模板
DEFAULT_COMPARE_PROMPT_TEMPLATE = """
你是一个能够判断两个句子意思是否相同的助手。
你会收到两个句子，你需要判断它们的语义是否一致。
你需要返回一个包含以下字段的 JSON 对象：
- is_similar: 如果意思相同则为 True，否则为 False
- reason: 解释判断逻辑
待判断的句子是：
- sentence1: {sentence1}
- sentence2: {sentence2}
"""

class LLMEvaluator:
    """LLM 评估器类，负责调用大模型进行语义比对。"""

    _reranker_llm_cache: Dict[str, Runnable] = {}

    @staticmethod
    def get_reranker_llm(model: str = None) -> Runnable:
        """获取（并缓存）用于 reranker 的 LLM。"""
        load_env()
        selected_model = model or config.get("llm_reranker_model", "doubao-seed-lite")
        if selected_model not in LLMEvaluator._reranker_llm_cache:
            LLMEvaluator._reranker_llm_cache[selected_model] = ChatOpenAI(
                model=selected_model,
                api_key=os.getenv("ARK_API_KEY"),
                base_url="https://ark.cn-beijing.volces.com/api/v3",
            ).with_structured_output(SimilarityResult)
        return LLMEvaluator._reranker_llm_cache[selected_model]
    
    @staticmethod
    def construct_with_ark(
            prompt=DEFAULT_COMPARE_PROMPT_TEMPLATE, model: str = None
        ):
            """使用火山引擎 (Ark) API 构建评估器（默认使用 llm_reranker_model）。"""
            llm = LLMEvaluator.get_reranker_llm(model=model)
            return LLMEvaluator(llm, prompt)

    def __init__(self, llm: Runnable, prompt: str):
        """初始化评估器，构建 LangChain 运行链。"""
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["sentence1", "sentence2"],
        )
        self.prompt = prompt_template
        # 将提示词和模型连接成一个 Runnable 链
        self.chain: Runnable = prompt_template | llm

    def predict(
        self,
        dataset: List[Tuple[str, str]],
        batch_size: int,
        show_progress: bool = True,
    ) -> LLMEvaluationResult:
        """
        批量评估句对的语义一致性。
        
        Args:
            dataset: 包含 (sentence1, sentence2) 元组的列表。
            batch_size: 每批次处理的数量。
            show_progress: 是否显示进度条。
            
        Returns:
            LLMEvaluationResult 对象，包含所有评估结果。
        """
        all_results = []
        dataset = list(dataset)
        num_batches = math.ceil(len(dataset) / batch_size)
    
        # 遍历每个批次进行请求
        for batch in tqdm(
            batch_iterable(dataset, batch_size),
            total=num_batches,
            disable=not show_progress,
        ):
            # 准备 LangChain batch 请求的输入数据
            batch_payload = [{"sentence1": s1, "sentence2": s2} for s1, s2 in batch]
            try:
                # 调用模型链的批量处理方法
                batch_results = self.chain.batch(batch_payload)
                for r in batch_results:
                    if isinstance(r, SimilarityResult):
                        all_results.append(r)
                    else:
                        # 兜底：如果返回的是 dict 则尝试校验为 SimilarityResult
                        all_results.append(SimilarityResult.model_validate(r))
            except Exception as e:
                # 异常处理：记录日志并在该批次填充失败结果，保证程序不中断
                logger.error(f"Error in batch: {e}")
                for _ in batch_payload:
                    all_results.append(SimilarityResult(is_similar=False, reason=f"Exception: {str(e)}"))
    
        return LLMEvaluationResult(all_results)

    def create_reranker(self, batch_size: int = 5) -> "LLMReranker":
        """基于当前评估器创建一个重排序器。"""
        return LLMReranker(self, batch_size=batch_size)

class LLMReranker:
    """利用 LLM 评估结果对候选列表进行筛选和打分的重排序器。"""
    
    def __init__(self, llm_evaluator: LLMEvaluator, batch_size: int = 5):
        self.llm_evaluator = llm_evaluator
        self.batch_size = batch_size

    def __call__(self, query: str, candidates: List[Dict]):
        """
        执行重排序逻辑。
        
        query 是“当前问题”，prompt 是“候选历史问题（缓存键）”，
        evaluator 的任务就是判断 query 和每个 prompt 是否语义一致。
        
        Args:
            query: 查询语句。
            candidates: 候选对象列表，每个对象应包含 "prompt" 键。
            
        Returns:
            仅包含语义相似（is_similar=True）的候选对象列表，并附带分值。
        """
        if not candidates:
            return []

        # 将 query 与每个候选句组成待预测对
        validation_pairs = []
        for candidate in candidates:
            prompt = candidate.get("prompt", "")
            validation_pairs.append((query, prompt))

        # 运行 LLM 批量预测
        llm_result = self.llm_evaluator.predict(
            validation_pairs, batch_size=self.batch_size, show_progress=False
        )

        # 过滤并补充元数据
        validated_candidates = [
            {
                **candidate,
                "reranker_type": "llm",
                # 如果相似，设置 score=0 (通常用于距离计算，0表示最贴近)，反之为 1
                "reranker_score": 0.0 if validation.is_similar else 1.0,
                "reranker_distance": 1.0 if validation.is_similar else 0.0,
                "reranker_reason": validation.reason,
            }
            for candidate, validation in zip(candidates, llm_result.resulting_items)
            if validation.is_similar # 仅保留 LLM 认为语义相同的项
        ]
        return validated_candidates