# ============================================================
# agent/memory.py
#
# Two separate memory stores, each with a distinct purpose:
#
# 1. SlidingWindowMemory — last N turns, injected into LLM
#    prompts. Kept short to control token cost and latency.
#
# 2. FullHistoryStore — every turn for a session, used for
#    ticket summaries and the escalation detector's "has this
#    been a recurring issue" check.
#
# Both are keyed by session_id so multiple concurrent users of
# the deployed app never share or leak memory between sessions.
# ============================================================


class SlidingWindowMemory:
    def __init__(self, max_turns: int = 3):
        self.max_turns = max_turns
        self.sessions: dict[str, list[dict]] = {}

    def add_turn(self, session_id: str, user_msg: str, agent_msg: str):
        self.sessions.setdefault(session_id, [])
        self.sessions[session_id].append({"user": user_msg, "agent": agent_msg})
        self.sessions[session_id] = self.sessions[session_id][-self.max_turns:]

    def get_history(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def format_for_prompt(self, session_id: str) -> str:
        """Turn stored history into a readable block for the LLM prompt."""
        history = self.get_history(session_id)
        if not history:
            return ""
        lines = []
        for turn in history:
            lines.append(f"Customer: {turn['user']}")
            lines.append(f"Agent: {turn['agent']}")
        return "\n".join(lines)

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)


class FullHistoryStore:
    def __init__(self):
        self.sessions: dict[str, list[dict]] = {}

    def add_turn(self, session_id: str, user_msg: str, agent_msg: str,
                 route: str, urgency: str):
        self.sessions.setdefault(session_id, [])
        self.sessions[session_id].append({
            "user": user_msg,
            "agent": agent_msg,
            "route": route,
            "urgency": urgency
        })

    def get(self, session_id: str) -> list[dict]:
        return self.sessions.get(session_id, [])

    def clear(self, session_id: str):
        self.sessions.pop(session_id, None)