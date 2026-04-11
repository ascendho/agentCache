"""
测试场景模块 (Scenarios Module)
这里定义了不同的测试问题，用于验证 Agent 工作流和语义缓存的能力。

这些场景模拟了一个典型的企业采购流程，通过在不同提问中包含交叠的子问题，
来观察语义缓存如何减少重复的 LLM 调用和搜索。
"""

# 场景 1：企业评估阶段
# 涵盖：企业版限额、2GB数据导出、安全合规标准、ACH支付支持。
SCENARIO_1_QUERY = """
We are evaluating your platform for our enterprise. We need to know the specific
API rate limits for the Enterprise plan, your data export options for a 2GB migration,
the security compliance standards you meet, and if you support ACH payments.
"""

# 场景 2：实施规划阶段
# 涵盖：Pro与企业版限额对比（包含重复项）、Salesforce集成、数据导出（重复项）、ACH支付（重复项）。
SCENARIO_2_QUERY = """
We're moving forward with implementation planning. I need to compare API rate limits 
between Pro and Enterprise plans to decide on our tier, confirm the Salesforce 
integration capabilities we discussed, understand what data export options you provide 
for our migration needs, and verify the payment methods including ACH since our 
accounting team prefers that for monthly billing.
"""

# 场景 3：最终确认阶段
# 涵盖：安全合规/SOC2（包含重复项）、Pro版限额（重复项）、Salesforce集成（重复项）、ACH支付（重复项）、未来迁移的数据导出（重复项）。
SCENARIO_3_QUERY = """
Before finalizing our Pro plan purchase, I need complete validation on: your security 
compliance framework including SOC2 requirements, the exact API rate limits for the 
Pro plan we're purchasing, confirmation of the Salesforce integration features, all 
supported payment methods since we want to use ACH transfers, and your data export 
capabilities for our future migration planning.
"""

# 统一维护：场景标题 + 场景问题。
# main.py 可直接消费该列表，避免在入口脚本里重复维护标题与 query 的映射关系。
SCENARIO_RUNS = [
	{
		"title": "Scenario 1: Enterprise Platform Evaluation",
		"query": SCENARIO_1_QUERY,
	},
	{
		"title": "Scenario 2: Implementation Planning",
		"query": SCENARIO_2_QUERY,
	},
	{
		"title": "Scenario 3: Pre-Purchase Comprehensive Review",
		"query": SCENARIO_3_QUERY,
	},
]