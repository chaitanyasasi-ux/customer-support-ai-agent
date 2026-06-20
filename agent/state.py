# ============================================================
# agent/state.py
#
# The shared state object that flows through every node in the
# LangGraph. Each node reads what it needs from this dict and
# writes its output back into it.
# ============================================================

from typing import TypedDict, Optional


class AgentState(TypedDict):
    session_id:        str
    query:             str
    route:             Optional[str]            # small_talk / rag_search / escalate
    retrieved_docs:    Optional[list]            # FAISS chunks, if RAG path taken
    confidence:        Optional[float]           # retrieval confidence score
    urgency:           Optional[str]             # Low / Medium / High
    answer:            Optional[str]             # final generated response
    needs_escalation:  Optional[bool]
    escalation_reason: Optional[str]
    metrics:           Optional[dict]            # latency tracking per node