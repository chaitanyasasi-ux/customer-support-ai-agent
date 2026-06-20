# ============================================================
# app.py
#
# Streamlit UI layer ONLY. All RAG, memory, graph, and ticket
# logic lives in the agent/ package and is imported here.
# ============================================================

import os
import uuid
import streamlit as st

from agent import (
    build_knowledge_base,
    SlidingWindowMemory,
    FullHistoryStore,
    build_agent_graph,
    run_agent,
    generate_ticket_summary,
)


st.set_page_config(
    page_title="Customer Support AI Agent",
    page_icon="🎧",
    layout="wide"
)


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_knowledge_base():
    return build_knowledge_base()


@st.cache_resource(show_spinner=False)
def load_groq_client():
    from groq import Groq
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
    if not api_key:
        st.error(
            "No Groq API key found. Add GROQ_API_KEY to Streamlit secrets "
            "(Settings → Secrets) or as an environment variable."
        )
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner=False)
def load_memory_stores():
    return SlidingWindowMemory(max_turns=3), FullHistoryStore()


@st.cache_resource(show_spinner=False)
def load_agent(_groq_client, _vectorstore, _sliding_memory, _full_history):
    return build_agent_graph(_groq_client, _vectorstore, _sliding_memory, _full_history)


vectorstore, num_docs, num_chunks = load_knowledge_base()
groq_client = load_groq_client()
sliding_memory, full_history = load_memory_stores()
agent = load_agent(groq_client, vectorstore, sliding_memory, full_history)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "last_state" not in st.session_state:
    st.session_state.last_state = None


st.title("🎧 Customer Support AI Agent")
st.caption(
    "Multi-tool agent: RAG knowledge base · urgency classifier · "
    "conversation memory · escalation detection · ticket summaries"
)

chat_col, sidebar_col = st.columns([2, 1])

with chat_col:
    st.subheader("Chat")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your support question...")

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_agent(agent, user_input, st.session_state.session_id)
            st.write(result["answer"])

        st.session_state.chat_messages.append({"role": "assistant", "content": result["answer"]})
        st.session_state.last_state = result
        st.rerun()


with sidebar_col:
    st.subheader("Agent reasoning")

    state = st.session_state.last_state

    if state is None:
        st.info(
            "Send a message to see the agent's routing decision, "
            "urgency classification, and confidence score here."
        )
    else:
        route_icons = {"small_talk": "🟢", "rag_search": "🔵", "escalate": "🟠"}
        st.markdown(f"**Route:** {route_icons.get(state['route'], '⚪')} `{state['route']}`")

        urgency_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        st.markdown(f"**Urgency:** {urgency_icons.get(state['urgency'], '⚪')} {state['urgency']}")

        if state.get("confidence") is not None:
            st.markdown(f"**Retrieval confidence:** {state['confidence']:.2f}")
            st.progress(min(max(state["confidence"], 0.0), 1.0))

        if state.get("needs_escalation"):
            st.error(f"Escalation flagged\n\n{state.get('escalation_reason', '')}")
        else:
            st.success("No escalation needed")

        with st.expander("Raw latency metrics"):
            st.json(state.get("metrics", {}))

    st.divider()
    st.subheader("Ticket summary")

    if st.button("Generate ticket for this conversation", use_container_width=True):
        if state is None:
            st.warning("Start a conversation first.")
        else:
            conversation = full_history.get(st.session_state.session_id)
            with st.spinner("Summarizing..."):
                summary = generate_ticket_summary(
                    groq_client, conversation,
                    fallback_query=state["query"],
                    fallback_urgency=state.get("urgency", "Medium"),
                )
            st.markdown(f"**Category:** {summary['category']}")
            st.markdown(f"**Urgency:** {summary['urgency']}")
            st.markdown(f"**Status:** {summary['resolution_status']}")
            st.markdown(f"**Issue:** {summary['issue_summary']}")
            st.markdown(f"**Next step:** {summary['recommended_next_step']}")

    st.divider()
    st.subheader("Knowledge base")
    st.markdown(f"**Documents indexed:** {num_docs}")
    st.markdown(f"**Chunks:** {num_chunks}")
    st.caption("Built from the Bitext customer support dataset (free, public)")

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.last_state = None
        sliding_memory.clear(st.session_state.session_id)
        full_history.clear(st.session_state.session_id)
        st.rerun()