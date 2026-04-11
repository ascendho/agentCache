# 运行指令：python main.py

import warnings
from cache.llm_evaluator import set_ark_key
from agent.graph import build_workflow
from app.bootstrap import create_knowledge_base
from app.workflow_runner import run_workflow_scenarios
from cache.bootstrap import setup_semantic_cache
from utility.logging_setup import setup_logging
from utility.env import to_bool_env

def main():
    """程序入口：完成初始化、执行场景测试并统计缓存表现。"""
    warnings.simplefilter("ignore")
    set_ark_key()
    logger = setup_logging()

    # 1) 创建知识库：把原始文本写入向量索引，供后续检索。
    logger.info("初始化知识库...")

    # kb_index 是 Redis 向量索引对象（SearchIndex 实例），
    # 代表“已经建好的知识库索引”，真正使用时在检索工具里做 query 查询
    # embeddings 是向量化模型对象（HFTextVectorizer），
    # 负责把文本转成向量，检索时把用户 query 向量化
    # 注意：当前建库时，create_knowledge_base_from_texts 内部也会再创建一个默认向量器用于入库向量生成
    # （同模型名，两个实例），主流程返回的 embeddings 主要用于“在线查询向量化”。
    kb_index, embeddings = create_knowledge_base()

    # 2) 初始化语义缓存：用于拦截相似问题，减少重复推理成本。
    logger.info("初始化语义缓存...")
    cache = setup_semantic_cache()

    # 3) 组装工作流：节点、路由、工具在这里被编排成可执行图。
    logger.info("构建 LangGraph 工作流...")
    workflow_app = build_workflow(cache, kb_index, embeddings)

    # 4) 依次执行三个场景，观察命中率随对话推进是否提升。
    show_console_results = to_bool_env("SHOW_CONSOLE_RESULTS", False)
    run_workflow_scenarios(
        workflow_app=workflow_app,
        logger=logger,
        show_console_results=show_console_results,
    )

if __name__ == "__main__":
    main()
