# ============================================================
# agent/ticket_summary.py
#
# Generates a structured JSON ticket summary from a conversation.
# Called on-demand (e.g. UI "Generate ticket" button), not on
# every turn.
# ============================================================

import re
import json


def _default_summary(query: str, urgency: str) -> dict:
    return {
        "issue_summary": query[:150],
        "category": "General Inquiry",
        "urgency": urgency,
        "resolution_status": "Unresolved",
        "recommended_next_step": "Review conversation manually — automatic summary failed.",
    }


def generate_ticket_summary(groq_client, conversation: list[dict],
                             fallback_query: str, fallback_urgency: str,
                             model_name: str = "llama-3.1-8b-instant") -> dict:
    """
    conversation: list of {"user": ..., "agent": ...} turns for this session.
    fallback_query / fallback_urgency: used only if the conversation list
    is empty or the LLM call fails entirely.
    """
    if not conversation:
        return _default_summary(fallback_query, fallback_urgency)

    convo_text = "\n".join(
        f"Customer: {t['user']}\nAgent: {t['agent']}" for t in conversation
    )

    prompt = f"""Summarize this customer support conversation into a structured ticket.
Respond with ONLY valid JSON, no markdown, no explanation. Use exactly this structure:

{{
  "issue_summary": "one sentence describing the customer's core issue",
  "category": "one or two words, e.g. Order Cancellation, Refund, Account Access",
  "urgency": "Low, Medium, or High",
  "resolution_status": "Resolved, Partially Resolved, or Unresolved",
  "recommended_next_step": "one sentence for the human agent picking this up"
}}

Conversation:
{convo_text}"""

    default_summary = _default_summary(fallback_query, fallback_urgency)

    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=250,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model added them anyway.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())

        # Extract just the {...} block in case there's extra text.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return default_summary

        parsed = json.loads(match.group())

        # Validate every expected field individually — fill gaps with
        # defaults field-by-field rather than discarding everything.
        required_fields = [
            "issue_summary", "category", "urgency",
            "resolution_status", "recommended_next_step"
        ]
        for field in required_fields:
            if field not in parsed or not parsed[field]:
                parsed[field] = default_summary[field]

        return parsed

    except (json.JSONDecodeError, Exception):
        return default_summary