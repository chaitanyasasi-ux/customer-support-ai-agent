# ============================================================
# agent/nodes.py
#
# Every node function the LangGraph calls. Each node has a single
# responsibility. Nodes take groq_client/vectorstore as factory
# arguments (make_*_node functions) rather than importing them as
# globals — this means nodes.py has zero hidden dependencies and
# can be unit tested with mock objects instead of real ones.
# ============================================================

import time
from .state import AgentState


# ── Router node ──────────────────────────────────────────────
def make_router_node(groq_client, model_name: str = "llama-3.1-8b-instant"):
    def router_node(state: AgentState) -> AgentState:
        t0 = time.time()
        prompt = f"""Classify this customer support message into exactly one category.

Categories:
- "small_talk"  : greetings, thanks, goodbyes, casual chat — no support need
- "rag_search"  : a real support question needing knowledge lookup
- "escalate"    : message shows strong frustration/anger, mentions legal action,
                   or is clearly abusive/off-topic for customer support

Message: "{state['query']}"

Respond with ONLY one word: small_talk, rag_search, or escalate"""

        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip().lower()
            valid_routes = ["small_talk", "rag_search", "escalate"]
            route = next((r for r in valid_routes if r in raw), None) or "rag_search"
        except Exception:
            # FAILURE MODE: Groq API down / rate limited. Default to
            # rag_search — a wasted search is cheaper than silently
            # ignoring a real support question.
            route = "rag_search"

        state["route"] = route
        state["metrics"] = state.get("metrics", {})
        state["metrics"]["routing_ms"] = round((time.time() - t0) * 1000, 1)
        return state

    return router_node


# ── Urgency node ──────────────────────────────────────────────
def make_urgency_node(groq_client, model_name: str = "llama-3.1-8b-instant"):
    def urgency_node(state: AgentState) -> AgentState:
        t0 = time.time()
        prompt = f"""Classify the urgency of this customer support message.

Levels:
- "Low"    : general question, no time pressure, casual tone
- "Medium" : a real issue but not urgent
- "High"   : money at risk, account locked/compromised, angry tone, blocked from service

Message: "{state['query']}"

Respond with ONLY one word: Low, Medium, or High"""

        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip()
            valid_levels = ["Low", "Medium", "High"]
            urgency = next(
                (lvl for lvl in valid_levels if lvl.lower() in raw.lower()), None
            ) or "Medium"
        except Exception:
            # Default to Medium — never assume Low (could miss a real
            # urgent issue), never assume High (over-alerts on every
            # ambiguous message, making the signal useless).
            urgency = "Medium"

        state["urgency"] = urgency
        state["metrics"]["urgency_ms"] = round((time.time() - t0) * 1000, 1)
        return state

    return urgency_node


# ── RAG search node ──────────────────────────────────────────
def make_rag_search_node(vectorstore, k: int = 3):
    def rag_search_node(state: AgentState) -> AgentState:
        t0 = time.time()
        results = vectorstore.similarity_search_with_score(state["query"], k=k)
        docs      = [r[0] for r in results]
        scores    = [r[1] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 999

        # Confidence heuristic: lower FAISS distance = higher confidence.
        confidence = max(0.0, 1.0 - (avg_score / 2.0))

        state["retrieved_docs"] = [
            {"content": d.page_content, "intent": d.metadata.get("intent", "?")}
            for d in docs
        ]
        state["confidence"] = round(float(confidence), 2)
        state["metrics"]["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        return state

    return rag_search_node


# ── Generation node ──────────────────────────────────────────
def make_generation_node(groq_client, sliding_memory, full_history,
                          model_name: str = "llama-3.1-8b-instant"):
    def generation_node(state: AgentState) -> AgentState:
        t0 = time.time()
        route   = state["route"]
        urgency = state.get("urgency", "Medium")

        history_block   = sliding_memory.format_for_prompt(state["session_id"])
        history_section = f"\nPrevious conversation:\n{history_block}\n" if history_block else ""

        if route == "small_talk":
            system_prompt = (
                "You are a friendly customer support agent. Respond briefly and warmly. "
                "Use the previous conversation for context if relevant."
            )
            user_content = f"{history_section}\nCustomer: {state['query']}"

        elif route == "escalate":
            urgency_tone = (
                "This is HIGH urgency — acknowledge immediacy and act fast."
                if urgency == "High" else
                "Acknowledge their concern and explain a human will assist soon."
            )
            system_prompt = (
                f"You are a customer support agent. This requires human attention. "
                f"{urgency_tone}"
            )
            user_content = f"{history_section}\nCustomer: {state['query']}"

        else:  # rag_search
            docs       = state.get("retrieved_docs", [])
            confidence = state.get("confidence", 0)
            context    = "\n\n---\n\n".join([d["content"] for d in docs])

            confidence_note = ""
            if confidence < 0.3:
                confidence_note = (
                    "\n\nNOTE: Retrieved info may not match the question well. "
                    "If it doesn't answer the question, say so."
                )

            system_prompt = (
                "You are a helpful customer support agent. Answer using the provided "
                "support information and previous conversation. Be concise and friendly."
                + confidence_note
            )
            user_content = (
                f"{history_section}\nSupport information:\n{context}\n\n"
                f"Customer: {state['query']}"
            )

        try:
            response = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content}
                ],
                temperature=0.3,
                max_tokens=300,
            )
            answer = response.choices[0].message.content.strip()
        except Exception:
            answer = (
                "I'm having trouble processing your request right now. "
                "Let me connect you to a human agent."
            )

        state["answer"] = answer
        state["metrics"]["generation_ms"] = round((time.time() - t0) * 1000, 1)

        # Save this turn to BOTH memory stores after generating an answer
        # (including the fallback message, if generation failed above).
        sliding_memory.add_turn(state["session_id"], state["query"], answer)
        full_history.add_turn(state["session_id"], state["query"], answer, route, urgency)

        return state

    return generation_node


# ── Escalation check node ────────────────────────────────────
def make_escalation_check_node(full_history):
    def escalation_check_node(state: AgentState) -> AgentState:
        t0 = time.time()
        reasons = []

        if state["route"] == "escalate":
            reasons.append("Router flagged message tone as requiring escalation")

        confidence = state.get("confidence")
        if confidence is not None and confidence < 0.3:
            reasons.append(
                f"Low retrieval confidence ({confidence}) — knowledge base may not cover this"
            )

        # NOTE: generation_node already wrote the CURRENT turn into
        # full_history before this node runs (see graph.py edge order).
        # Slice off the last entry to look only at turns BEFORE this one.
        all_turns  = full_history.get(state["session_id"])
        past_turns = all_turns[:-1] if all_turns else []
        had_high_urgency_before = any(t["urgency"] == "High" for t in past_turns)
        is_repeat_contact = len(past_turns) >= 1

        if (state.get("urgency") == "High" or had_high_urgency_before) and is_repeat_contact:
            reasons.append(
                f"High urgency present in this session across {len(all_turns)} turns — "
                f"issue may not be resolving"
            )

        state["needs_escalation"]  = len(reasons) > 0
        state["escalation_reason"] = "; ".join(reasons) if reasons else None
        state["metrics"]["escalation_check_ms"] = round((time.time() - t0) * 1000, 1)
        return state

    return escalation_check_node