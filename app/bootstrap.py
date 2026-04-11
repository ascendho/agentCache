from redisvl.utils.vectorize import HFTextVectorizer

from agent import create_knowledge_base_from_texts


def create_knowledge_base():
    """构建演示用知识库并返回索引与向量模型。"""
    embeddings = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    raw_docs = [
        "Our premium support plan includes 24/7 phone support, priority email response within 2 hours, and dedicated account management. Premium support costs $49/month.",
        "Account upgrade process: Go to Account Settings -> Plan & Billing -> Select Upgrade. Available plans: Basic $9/month, Pro $29/month, Enterprise $99/month.",
        "API rate limits by plan: Free tier 100 requests/hour, Basic 1,000 requests/hour, Pro 10,000 requests/hour, Enterprise unlimited with fair-use policy.",
        "Data export options: CSV, JSON, XML formats supported. Large exports (>1GB) may take up to 24 hours to process.",
        "Third-party integrations: Native support for Slack, Microsoft Teams, Zoom, Salesforce, HubSpot. 200+ additional integrations available via Zapier.",
        "Security features: SOC2 compliance, end-to-end encryption, GDPR compliance, SSO integration, audit logs, IP whitelisting.",
        "Billing and payments: We accept all major credit cards, PayPal, and ACH transfers. Enterprise customers can pay by invoice with NET30 terms.",
        "Account recovery: Use forgot password link, verify email, or contact support with account verification details. Response within 4 hours.",
    ]

    _, _, kb_index = create_knowledge_base_from_texts(
        texts=raw_docs,
        source_id="customer_support_docs",
        redis_url="redis://localhost:6379",
        skip_chunking=True,
    )
    return kb_index, embeddings
