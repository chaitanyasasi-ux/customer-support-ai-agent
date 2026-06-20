# Multi-Tool Customer Support AI Agent

A production-style agentic RAG system for customer support — built with **LangGraph**, **FAISS**, **HuggingFace embeddings**, and **Groq (Llama 3.1)**. The agent doesn't just answer questions from a knowledge base; it routes messages intelligently, classifies urgency, remembers conversation context, detects when a human needs to step in, and generates structured tickets for handoff.

**[Live Demo →](#)** *(add your deployed Streamlit/HuggingFace Spaces link here)*

---

## Why this exists

Most "RAG chatbot" portfolio projects are a single retrieval → generation loop. This project intentionally goes further: it's an **agent with decision-making**, not a fixed pipeline. Every architectural choice below was driven by a real failure mode discovered during testing — not theoretical best practices copied from a tutorial.

---

## Architecture

```
User message
     │
     ▼
┌─────────┐      ┌─────────┐
│ Router  │ ───▶ │ Urgency │   (always runs — even small-talk-looking
└─────────┘      └─────────┘    messages can carry real urgency)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
 ┌─────────────┐              ┌──────────────┐
 │ RAG search   │              │ small_talk / │
 │ (FAISS+MiniLM)│             │ escalate     │
 └─────────────┘              └──────────────┘
        │                             │
        └──────────────┬──────────────┘
                        ▼
                ┌─────────────┐
                │ Generation   │  (Groq llama-3.1-8b-instant,
                │ (+ memory)   │   reads sliding-window history)
                └─────────────┘
                        │
                        ▼
                ┌──────────────────┐
                │ Escalation check  │  (rule-based, not an LLM call —
                └──────────────────┘   reasons over confidence + urgency
                        │               + session history)
                        ▼
                  Response + optional
                  structured ticket (on demand)
```

---

## What it does

| Capability | How |
|---|---|
| **Knowledge retrieval** | FAISS vector search over the Bitext customer support dataset, embedded with `all-MiniLM-L6-v2` |
| **Intelligent routing** | An LLM classifies each message as `small_talk`, `rag_search`, or `escalate` *before* any retrieval happens — avoiding wasted search calls on greetings or angry messages |
| **Urgency classification** | A second, independent LLM call tags every message Low/Medium/High — run separately from routing because single-purpose prompts are more reliable than one prompt asking for two judgments |
| **Conversation memory** | A sliding window of the last 3 turns is injected into every generation prompt, so follow-up questions ("how long will *that* take?") resolve correctly |
| **Escalation detection** | Rule-based (not LLM-based) — checks router output, retrieval confidence, and **session-level urgency history**, not just the current message's tone |
| **Ticket summarization** | Generates structured JSON (issue, category, urgency, status, next step) for human handoff, with defensive parsing for malformed LLM output |

---

## Tech stack

- **Orchestration:** LangGraph (StateGraph with conditional routing)
- **Retrieval:** FAISS + `sentence-transformers/all-MiniLM-L6-v2`
- **Generation:** Groq API, `llama-3.1-8b-instant`
- **UI:** Streamlit
- **Dataset:** [Bitext Customer Support LLM Chatbot Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (27 intents, free, public)

100% free to run — no paid API tiers required.

---

## Measured performance

All numbers below are from actual test runs, not estimates:

| Metric | Value |
|---|---|
| Avg retrieval latency (FAISS) | ~8–30ms |
| Avg routing latency | ~110–300ms |
| Avg urgency classification latency | ~105–250ms |
| Avg generation latency (Groq) | ~170–520ms |
| Avg ticket summary generation | ~180–290ms |
| Precision@3 (retrieval accuracy) | 100% on 10 labeled test queries |
| Knowledge base size | 27 intents → 240 chunks |
| Index build time | ~0.2–2.5s |

---

## Design decisions and known limitations

Honest engineering means documenting tradeoffs, not just features. These are the real decisions made (and bugs caught) while building this:

**Escalation runs as deterministic rules, not a 4th LLM call.** By the time escalation check runs, the graph already has `confidence`, `urgency`, `route`, and turn count in state. Adding another LLM call to re-judge information already computed would cost latency for no benefit — sometimes the right call is *not* to reach for an LLM.

**Two separate memory stores exist, deliberately not merged.** A sliding window (last 3 turns) optimizes for cheap, focused generation prompts. A separate full-history store optimizes for human handoff — a support agent picking up an escalated ticket needs the whole conversation, not a truncated window. Conflating these would force a bad tradeoff in one direction.

**Escalation logic checks session history, not just the current message's tone.** An earlier version only checked whether the *current* turn was marked "High" urgency. Testing revealed a real failure mode: a customer's second message ("it's still locked, second time asking") often reads calmer in tone than their first, even though the underlying issue hasn't been resolved. The rule now checks whether *any* turn in the session was High urgency, combined with repeat contact — catching persisting issues that tone alone would miss.

**Session-state ordering matters in stateful graphs.** While fixing the bug above, a second, subtler bug surfaced: the generation node writes the current turn into full-history *before* the escalation-check node runs. A naive history check would therefore include the current turn when computing "past" turns, making "is this a repeat contact" trivially true on a user's very first message. The fix slices off the last entry before checking history — a reminder that node ordering and shared mutable state in agentic graphs needs to be reasoned about explicitly, not assumed.

**Knowledge base gaps are handled by retrieval confidence, not denial.** The Bitext dataset doesn't cover every possible support topic (e.g. it has no "account locked" or "return policy" intent). Rather than letting the LLM hallucinate an answer from irrelevant retrieved chunks, a confidence score (derived from FAISS distance) below 0.3 triggers an explicit instruction to the model to hedge or escalate instead of guessing.

**Generator model choice was revised mid-project.** `flan-t5-large` was the original generator (free, local, no API key needed) but proved too weak at open-ended instruction-following — it frequently echoed prompts back or returned generic fallbacks even when given relevant context. Switched to Groq's `llama-3.1-8b-instant`, which is instruction-tuned and dramatically more reliable at extracting and rephrasing context, while still being free and fast (Groq's LPU hardware keeps latency under 1 second even on the free tier).

**Bitext's template placeholders required explicit cleanup.** The dataset contains unfilled variables like `{{Customer Support Hours}}` meant to be populated by a real company's backend. Left unhandled, these leaked verbatim into agent responses. A mapping of known placeholders to realistic defaults, plus a regex safety net for any unmapped pattern, ensures no broken-looking text reaches a user.

---

## Project structure

```
support_agent/
├── app.py                      # Streamlit UI only — no agent logic
├── requirements.txt
├── .streamlit/
│   ├── secrets.toml            # Groq API key (gitignored)
│   └── config.toml             # disables file-watcher noise
└── agent/
    ├── __init__.py              # public API exports
    ├── knowledge_base.py        # dataset loading, cleaning, FAISS indexing
    ├── memory.py                 # sliding-window + full-history stores
    ├── state.py                  # shared AgentState TypedDict
    ├── nodes.py                  # router, urgency, RAG, generation, escalation
    ├── graph.py                  # LangGraph wiring
    └── ticket_summary.py         # structured JSON summary + defensive parsing
```

The UI layer (`app.py`) contains zero RAG/agent logic — everything is imported from `agent/`. This means the entire backend is reusable as-is behind a different frontend (e.g. FastAPI + React) without touching a single line of agent code.

---

## Running locally

```bash
git clone <your-repo-url>
cd support_agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add your free [Groq API key](https://console.groq.com) to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_your_key_here"
```

Then run:
```bash
streamlit run app.py
```

---

## What's next

- Reranking retrieved chunks before generation (currently top-k by raw FAISS distance only)
- Swapping the rule-based confidence threshold for a learned/calibrated score
- Persisting conversation + ticket history to a real database instead of in-memory dicts
- Adding a small eval suite (precision@k across more labeled queries, latency regression tests)

---

Built as a portfolio project demonstrating production-style agentic RAG patterns: routing, memory, escalation, and structured output — not just a retrieval demo.
