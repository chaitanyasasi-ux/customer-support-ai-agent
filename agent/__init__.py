# ============================================================
# agent/__init__.py
#
# Exposes the public API of the agent package so app.py can do:
#   from agent import build_knowledge_base, build_agent_graph, run_agent, ...
# instead of reaching into individual submodules.
# ============================================================

from .knowledge_base import build_knowledge_base
from .memory import SlidingWindowMemory, FullHistoryStore
from .graph import build_agent_graph, run_agent
from .ticket_summary import generate_ticket_summary

__all__ = [
    "build_knowledge_base",
    "SlidingWindowMemory",
    "FullHistoryStore",
    "build_agent_graph",
    "run_agent",
    "generate_ticket_summary",
]