"""
Deep Research Agent 的简洁 Gradio 演示界面。
该模块用于演示“网页内容抽取 + 知识库构建 + 语义缓存问答”的完整流程，
依赖 Tavily API 完成网页正文抽取。
"""

import gradio as gr
import time
import logging
import os
from datetime import datetime
from typing import Dict, Any
from getpass import getpass

import redis
from tavily import TavilyClient
from .knowledge_base_utils import KnowledgeBaseManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")


def _get_tavily_api_key() -> str:
    """
    获取 Tavily API Key。
    优先从环境变量读取，如果不存在则提示用户手动输入。
    """
    api_key = os.getenv("TAVILY_API_KEY")
    
    if api_key:
        logger.info("Using Tavily API key from environment")
        return api_key
    
    logger.info("Tavily API key not found in environment, requesting from user...")
    # 在某些环境下（如本地终端）使用 getpass 隐藏输入
    api_key = getpass("Please enter your Tavily API key: ")
    
    if not api_key.strip():
        raise ValueError("Tavily API key is required to run the demo")
    
    return api_key.strip()


class ResearchDemo:
    """
    演示状态管理器：负责协调 UI 交互与后端逻辑。
    维护当前正在处理的 URL、本地知识库以及连续提问的历史性能日志。
    """
    
    def __init__(self, workflow_app, semantic_cache, tavily_api_key, redis_url="redis://localhost:6379"):
        """
        初始化演示程序。
        
        Args:
            workflow_app: 已编译的 LangGraph 工作流实例。
            semantic_cache: 语义缓存实例。
            tavily_api_key: 用于抓取网页内容的 API Key。
            redis_url: Redis 连接地址，用于存储向量索引和缓存。
        """
        self.workflow_app = workflow_app
        self.semantic_cache = semantic_cache
        # 初始化 Tavily 客户端，支持通过环境变量自定义 Base URL（适配 DLAI 等特殊环境）
        self.tavily_client = TavilyClient(api_key=tavily_api_key, api_base_url=os.getenv("DLAI_TAVILY_BASE_URL"))
        self.current_url = None
        # run_log 用于按时间顺序累积每次提问的性能摘要（耗时、命中率、Token估算等）
        self.run_log = []
        
        # 初始化知识库管理器
        redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.kb_manager = KnowledgeBaseManager(redis_client)
    
    def _reset_all_data(self) -> str:
        """
        重置演示状态：清空 Redis 中的向量索引以及内存/Redis 中的语义缓存。
        确保新一轮演示从零开始。
        """
        try:
            # 清除向量知识库
            kb_status = self.kb_manager.clear_knowledge_base()
            
            # 清除语义缓存内容
            self.semantic_cache.cache.clear()
            
            self.current_url = None
            self.run_log = []
            
            logger.info("🧹 Complete reset: cleared knowledge bases and semantic cache")
            return f"🧹 Reset complete: {kb_status}, semantic cache cleared"
            
        except Exception as e:
            error_msg = f"❌ Reset failed: {e}"
            logger.error(error_msg)
            return error_msg
    
    def process_url(self, url: str) -> str:
        """
        抓取 URL 内容、提取正文并建立向量索引。
        """
        if not url.strip():
            return "❌ Please enter a URL"
        
        try:
            # 在处理新 URL 前先重置旧数据，防止知识污染
            reset_status = self._reset_all_data()
            logger.info(f"Reset status: {reset_status}")
            
            # 使用 Tavily 的 extract 功能获取网页正文
            # extract_depth="advanced" 会尝试绕过部分反爬并提取更干净的文本
            result = self.tavily_client.extract(
                urls=[url], 
                extract_depth="advanced"
            )
            
            if not result or 'results' not in result or not result['results']:
                return "❌ No content extracted from URL"
            
            # 提取原始内容或清洗后的正文
            extracted_content = result['results'][0].get('raw_content', '') or result['results'][0].get('content', '')
            if not extracted_content:
                return "❌ No readable content found"
            
            # 将提取的内容切片并存入 Redis 向量数据库
            success, message, kb_index = self.kb_manager.create_knowledge_base(
                source_id=url,
                content=extracted_content,
                chunk_size=4500,     # 每个切片约 4500 字符
                chunk_overlap=250    # 切片间的重叠部分，保证上下文不丢失
            )
            
            if success:
                self.current_url = url
                # 重要：URL 变化后，必须重新初始化工具，使其指向新的 Redis 向量索引名
                from . import initialize_tools
                initialize_tools(kb_index, self.kb_manager.embeddings)
                return f"✅ Extracted {len(extracted_content)} chars, {message}"
            else:
                return f"❌ {message}"
                
        except Exception as e:
            return f"❌ Processing failed: {e}"
    
    def ask_question(self, question: str) -> tuple[str, str]:
        """
        执行一次问答流程。
        1. 绑定当前知识库索引。
        2. 运行 Agent 工作流。
        3. 计算并格式化性能指标日志。
        """
        if not question.strip():
            return "❌ Please enter a question", ""
        
        if not self.current_url:
            return "❌ Please process a URL first", ""
        
        try:
            from . import run_agent, initialize_agent
            
            # 获取当前 URL 对应的 Redis 索引名
            kb_index = self.kb_manager.get_index_for_source(self.current_url)
            # 确保 Agent 节点的依赖是最新的
            initialize_agent(self.semantic_cache, kb_index, self.kb_manager.embeddings)
            
            start_time = time.perf_counter()
            # 执行工作流：开启语义缓存
            result = run_agent(self.workflow_app, question, enable_caching=True)
            total_time = (time.perf_counter() - start_time) * 1000
            
            answer = result.get('final_response', 'No response generated')
            
            logger.info(f"答案预览: {answer[:100]}...")
            
            # 统计性能指标
            cache_hits = len([h for h in result.get('cache_hits', {}).values() if h])
            total_questions = len(result.get('sub_questions', []))
            total_llm_calls = sum(result.get('llm_calls', {}).values()) if result.get('llm_calls') else 0
            # 粗略估算 Token 消耗（假设每次调用平均 150 tokens）
            estimated_tokens = total_llm_calls * 150
            
            # 构建一行日志：显示轮次、时间、耗时、缓存命中情况、LLM调用数和Token估算
            run_number = len(self.run_log) + 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            question_preview = question[:50] + "..." if len(question) > 50 else question
            
            log_entry = f"#{run_number:2d} | {timestamp} | {total_time:4.0f}ms | {cache_hits}/{total_questions} hits | {total_llm_calls:2d} LLM | ~{estimated_tokens:4d} tokens | Q: {question_preview}"
            
            # 将最新日志放在最前面
            self.run_log.append(log_entry)
            metrics_display = "\n".join(reversed(self.run_log))
            
            return str(answer), str(metrics_display)
            
        except Exception as e:
            return f"❌ Error: {e}", ""
    
    
    def get_status(self) -> str:
        """
        获取当前系统的运行状态。
        """
        if self.current_url:
            kb_status = self.kb_manager.get_status()
            kb_count = kb_status['total_indexes']
            return f"✅ Ready | URL: {self.current_url} | KB: {kb_count} loaded"
        else:
            return "⏳ No content loaded"


def create_demo(workflow_app, semantic_cache, tavily_api_key):
    """
    使用 Gradio Blocks 构建演示界面。
    """
    
    demo_state = ResearchDemo(workflow_app, semantic_cache, tavily_api_key)
    
    with gr.Blocks(title="Deep Research Agent Demo") as demo:
        gr.Markdown("# Deep Research Agent with Semantic Caching")
        gr.Markdown("语义缓存能够随着提问次数的增加，通过复用历史研究成果，逐步提高回答速度并降低成本。")

        # 第一部分：网页加载
        gr.Markdown("## 1. 加载网页内容")
        url_input = gr.Textbox(label="输入网址", placeholder="https://example.com/article", lines=1)
        process_btn = gr.Button("抓取并构建知识库")
        status_output = gr.Textbox(label="状态信息", interactive=False, lines=2, max_lines=6)

        # 第二部分：问答环节
        gr.Markdown("## 2. 深度研究提问")
        question_input = gr.Textbox(
            label="研究问题",
            placeholder="例如：本文讨论的核心观点有哪些？",
            lines=2, max_lines=6
        )
        ask_btn = gr.Button("发送提问")

        # 第三部分：结果输出
        gr.Markdown("## 3. 回答结果")
        answer_output = gr.Textbox(interactive=False, lines=10, max_lines=1000, show_label=False)

        # 第四部分：性能监控日志
        gr.Markdown("## 4. 性能日志 (观察缓存如何起作用)")
        metrics_output = gr.Textbox(interactive=False, lines=10, max_lines=1000, show_label=False)

        # 绑定按钮点击事件
        process_btn.click(fn=demo_state.process_url, inputs=[url_input], outputs=[status_output])
        ask_btn.click(fn=demo_state.ask_question, inputs=[question_input], outputs=[answer_output, metrics_output])
        
        # 页面加载时自动获取一次状态
        demo.load(fn=demo_state.get_status, outputs=[status_output])
    
    return demo


def launch_demo(workflow_app, semantic_cache, tavily_api_key=None, **kwargs):
    """
    启动 Gradio 演示应用。
    """
    if tavily_api_key is None:
        tavily_api_key = _get_tavily_api_key()
    
    demo = create_demo(workflow_app, semantic_cache, tavily_api_key)
    
    # 设置默认高度
    if 'height' not in kwargs:
        kwargs['height'] = 1000
    
    return demo.launch(**kwargs)