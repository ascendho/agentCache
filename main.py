# 运行指令：python main.py

import logging
import warnings
from cache.llm_evaluator import set_ark_key
from redisvl.utils.vectorize import HFTextVectorizer
from agent import create_knowledge_base_from_texts
from cache.wrapper import SemanticCacheWrapper
from cache.config import config
from cache.faq_data_container import FAQDataContainer
from langgraph.graph import StateGraph, END
from agent import (
    WorkflowState,
    initialize_agent,
    decompose_query_node,
    check_cache_node,
    research_node,
    evaluate_quality_node,
    synthesize_response_node,
    route_after_cache_check,
    route_after_quality_evaluation,
    run_agent,
    display_results,
    analyze_agent_results
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("agentic-workflow")

def create_knowledge_base():
    embeddings = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    raw_docs = [
        "Our premium support plan includes 24/7 phone support, priority email response within 2 hours, and dedicated account management. Premium support costs $49/month.",
        "Account upgrade process: Go to Account Settings → Plan & Billing → Select Upgrade. Available plans: Basic $9/month, Pro $29/month, Enterprise $99/month.",
        "API rate limits by plan: Free tier 100 requests/hour, Basic 1,000 requests/hour, Pro 10,000 requests/hour, Enterprise unlimited with fair-use policy.",
        "Data export options: CSV, JSON, XML formats supported. Large exports (>1GB) may take up to 24 hours to process.",
        "Third-party integrations: Native support for Slack, Microsoft Teams, Zoom, Salesforce, HubSpot. 200+ additional integrations available via Zapier.",
        "Security features: SOC2 compliance, end-to-end encryption, GDPR compliance, SSO integration, audit logs, IP whitelisting.",
        "Billing and payments: We accept all major credit cards, PayPal, and ACH transfers. Enterprise customers can pay by invoice with NET30 terms.",
        "Account recovery: Use forgot password link, verify email, or contact support with account verification details. Response within 4 hours.",
    ]

    success, message, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,
        source_id="customer_support_docs",
        redis_url="redis://localhost:6379",
        skip_chunking=True
    )
    
    return kb_index, embeddings

def setup_semantic_cache():
    cache = SemanticCacheWrapper.from_config(config)
    data = FAQDataContainer()
    cache.hydrate_from_df(data.faq_df, clear=True)
    return cache

def build_workflow(cache, kb_index, embeddings):
    initialize_agent(cache, kb_index, embeddings)
    workflow = StateGraph(WorkflowState)

    workflow.add_node("decompose_query", decompose_query_node)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("research", research_node)
    workflow.add_node("evaluate_quality", evaluate_quality_node)
    workflow.add_node("synthesize", synthesize_response_node)

    workflow.set_entry_point("decompose_query")

    workflow.add_edge("decompose_query", "check_cache")
    workflow.add_conditional_edges(
        "check_cache",
        route_after_cache_check,
        {
            "research": "research",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("research", "evaluate_quality")
    workflow.add_conditional_edges(
        "evaluate_quality",
        route_after_quality_evaluation,
        {
            "research": "research",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()

def main():
    warnings.simplefilter("ignore")
    set_ark_key()
    logger = setup_logging()

    logger.info("Initializing Knowledge Base...")
    kb_index, embeddings = create_knowledge_base()

    logger.info("Setting up Semantic Cache...")
    cache = setup_semantic_cache()

    logger.info("Building LangGraph Workflow...")
    workflow_app = build_workflow(cache, kb_index, embeddings)

    # ------------------ Scenarios ------------------
    logger.info("Running Scenario 1: Enterprise Platform Evaluation")
    scenario1_query = """
    We are evaluating your platform for our enterprise. We need to know the specific
    API rate limits for the Enterprise plan, your data export options for a 2GB migration,
    the security compliance standards you meet, and if you support ACH payments.
    """
    result1 = run_agent(workflow_app, scenario1_query)
    display_results(result1)

    logger.info("Running Scenario 2: Implementation Planning")
    scenario2_query = """
    We're moving forward with implementation planning. I need to compare API rate limits 
    between Pro and Enterprise plans to decide on our tier, confirm the Salesforce 
    integration capabilities we discussed, understand what data export options you provide 
    for our migration needs, and verify the payment methods including ACH since our 
    accounting team prefers that for monthly billing.
    """
    result2 = run_agent(workflow_app, scenario2_query)
    display_results(result2)

    logger.info("Running Scenario 3: Pre-Purchase Comprehensive Review")
    scenario3_query = """
    Before finalizing our Pro plan purchase, I need complete validation on: your security 
    compliance framework including SOC2 requirements, the exact API rate limits for the 
    Pro plan we're purchasing, confirmation of the Salesforce integration features, all 
    supported payment methods since we want to use ACH transfers, and your data export 
    capabilities for our future migration planning.
    """
    result3 = run_agent(workflow_app, scenario3_query)
    display_results(result3)

    logger.info("Analyzing Agent Performance...")
    total_questions, total_cache_hits = analyze_agent_results(
        [result1, result2, result3]
    )
    logger.info(f"Total Sub-Questions Evaluated: {total_questions}, Total Cache Hits: {total_cache_hits}")

if __name__ == "__main__":
    main()
