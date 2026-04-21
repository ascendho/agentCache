import logging
import sys
from pathlib import Path

# 设置路径以从 src 目录导入
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 屏蔽大面积日志干扰
logging.basicConfig(level=logging.WARNING)

from knowledge.builder import init_app_knowledge_base
from workflow.tools import initialize_tools, search_knowledge_base

print("🚀 正在加载本地知识库与 BGE 向量模型 (约需几秒钟)...")
kb_index, embeddings = init_app_knowledge_base()
initialize_tools(kb_index, embeddings)

print("\n" + "="*60)
print(" 🔬 混合检索 (双重漏斗) 效果演示")
print("="*60)

# 测试案例 1：纯向量擅长的“模糊语义”
query_semantic = "我把东西退回去，主要是通过什么方式给钱？"
print(f"\n🏷️ 测试 1 (测语义): '{query_semantic}'")
print("   [思考]: 这句话没有明确的‘退款’、‘支付’等专业词汇，仅靠字词匹配几乎找不到，必须靠向量懂语义！")
res_semantic = search_knowledge_base.invoke(query_semantic)
print(f"\n=> 检索结果:\n{res_semantic}\n")

# 测试案例 2：BM25擅长的“孤僻/确切专有名词”
query_literal = "我的退换货 RMA Number 应该写在哪里？"
print(f"\n🏷️ 测试 2 (测字面): '{query_literal}'")
print("   [思考]: 'RMA Number' 属于极其冷门的专业英文缩写。向量模型因为不认识它，很容易把它映射成噪音从而找不到；但 BM25 因为是纯看字面词频，能做到一击必杀！")
res_literal = search_knowledge_base.invoke(query_literal)
print(f"\n=> 检索结果:\n{res_literal}\n")

print("="*60)
print("✅ 演示完成！混合检索让系统变成了：既懂人话的模糊暗示（向量），又绝不漏掉核心的专业名词（BM25）。")
