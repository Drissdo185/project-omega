import os
from app.utils.azure_openai_client import AzureOpenAIClient

_client = AzureOpenAIClient()

# Simple prompt templates with deterministic temperature=0
_DOC_CLASS_PROMPT = """
You are an AI assistant that classifies text into one of three categories: HR, IT, or Other.

Rules:
- HR: related to employees, leave, salary, benefits, HR policies, recruitment, onboarding, staff performance, training, or personnel management.
- IT: related to technology, software, servers, APIs, networks, cybersecurity, systems, or data management.
- Other: unrelated to HR or IT.

Return JSON only:
{{"label": "HR" | "IT" | "Other", "reason": "short reason"}}

Examples:
Input: "How many days of annual leave does a staff member receive?" -> HR
Input: "Database backup policy for the internal network." -> IT
Input: "Procurement procedure for external vendors." -> Other

Classify this text:
\"\"\"{excerpt}\"\"\"
"""


def classify_text_short(excerpt: str):
    if not _client.client:
        text = excerpt.lower()
        if any(k in text for k in ["salary", "leave", "recruit", "onboarding", "performance", "hr", "benefit", "vacation", "holiday", "policy", "staff", "employee"]):
            return "HR", 1.0
        if any(k in text for k in ["server", "database", "kubernetes", "ssh", "api", "it", "network", "deploy", "software", "system", "infrastructure", "cybersecurity"]):
            return "IT", 1.0
        return "Other", 0.5

    prompt = _DOC_CLASS_PROMPT.format(excerpt=excerpt[:4000])
    resp = _client.call_completion(
        prompt,
        temperature=0.0,
        max_tokens=100,
        system_message="You are a classifier that returns strict JSON only, no explanation outside JSON."
    )
    text = resp.get("text", "").strip()
    # basic parse: try to extract JSON
    import json
    label = "Other"
    score = 0.0
    try:
        # if model returned pure text, attempt JSON load
        parsed = json.loads(text)
        label = parsed.get("label", "Other")
        reason = parsed.get("reason", "")
        score = 1.0
    except Exception:
        # fallback: simple heuristics
        txt = text.lower()
        if "hr" in txt or "salary" in txt:
            label = "HR"
            score = 0.8
        elif "it" in txt or "server" in txt:
            label = "IT"
            score = 0.8
        else:
            label = "Other"
            score = 0.5
    return label, score
