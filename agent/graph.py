# ============================================================
# agent/graph.py
#
# Builds and compiles the LangGraph from the node functions in
# nodes.py. This is the only file that knows the SHAPE of the
# agent (which node leads to which).
# ============================================================

from typing import Literal
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    make_router_node,
    make_urgency_node,
    make_rag_search_node,
    make_generation_node,
    make_escalation_check_node,
)


def route_decision(state: AgentState) -> Literal["rag_search", "generation"]:
    """Conditional edge: after urgency classification, decide whether
    the rag_search node is needed or we can skip straight to generation."""
    if state["route"] == "rag_search":
        return "rag_search"
    return "generation"


def build_agent_graph(groq_client, vectorstore, sliding_memory, full_history):
    """
    Wires together all five nodes into the compiled agent graph.

    Graph shape:
        router → urgency → (conditional) → rag_search → generation → escalation_check → END
                                          ↘ generation ──────────────↗
    """
    router_node           = make_router_node(groq_client)
    urgency_node           = make_urgency_node(groq_client)
    rag_search_node        = make_rag_search_node(vectorstore)
    generation_node        = make_generation_node(groq_client, sliding_memory, full_history)
    escalation_check_node  = make_escalation_check_node(full_history)

    graph = StateGraph(AgentState)

    graph.add_node("router",           router_node)
    graph.add_node("urgency",          urgency_node)
    graph.add_node("rag_search",       rag_search_node)
    graph.add_node("generation",       generation_node)
    graph.add_node("escalation_check", escalation_check_node)

    graph.set_entry_point("router")
    graph.add_edge("router", "urgency")

    graph.add_conditional_edges(
        "urgency",
        route_decision,
        {"rag_search": "rag_search", "generation": "generation"}
    )

    graph.add_edge("rag_search", "generation")
    graph.add_edge("generation", "escalation_check")
    graph.add_edge("escalation_check", END)

    return graph.compile()


def run_agent(agent, query: str, session_id: str) -> AgentState:
    """Convenience wrapper to invoke the agent with a fresh initial state."""
    initial_state: AgentState = {
        "session_id": session_id,
        "query": query,
        "route": None,
        "retrieved_docs": None,
        "confidence": None,
        "urgency": None,
        "answer": None,
        "needs_escalation": None,
        "escalation_reason": None,
        "metrics": {},
    }
    return agent.invoke(initial_state)