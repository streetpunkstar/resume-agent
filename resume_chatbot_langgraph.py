"""
LangGraph scaffold for the resume chatbot's intake + gating pipeline.

This replaces a linear function-call pipeline with an explicit state graph.
Plug your existing Ollama / ChromaDB / email logic into the marked TODOs.

Install:
    pip install langgraph langchain-core --break-system-packages
"""

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# 1. State schema — everything a node might need to read or write.
#    This is the "shared memory" that flows through the graph.
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    visitor_id: str
    message: str
    intent: Optional[Literal["professional", "personal", "job_opportunity", "spam"]]
    has_gdpr_consent: bool
    access_level: Optional[Literal["professional_only", "full"]]
    retrieved_context: Optional[str]
    response: Optional[str]
    should_notify_coach: bool


# ---------------------------------------------------------------------------
# 2. Nodes — each one is a plain function: state in, partial state out.
# ---------------------------------------------------------------------------

def classify_intent(state: ChatState) -> dict:
    """Classify the incoming message. Swap in a small Gemma call or a
    cheap classifier — this doesn't need your main generation model."""
    # TODO: replace with your Ollama classification call
    message = state["message"].lower()
    if any(k in message for k in ["viagra", "http://", "click here"]):
        intent = "spam"
    elif any(k in message for k in ["hire", "opportunity", "role", "position"]):
        intent = "job_opportunity"
    elif any(k in message for k in ["family", "hobbies", "personal"]):
        intent = "personal"
    else:
        intent = "professional"
    return {"intent": intent}


def log_and_deflect(state: ChatState) -> dict:
    """Spam/suspicious path — log to admin dashboard, skip generation entirely."""
    # TODO: write to your admin analytics store
    return {"response": "Thanks for reaching out — this message has been flagged for review."}


def request_consent(state: ChatState) -> dict:
    """GDPR gate — return a consent prompt instead of proceeding."""
    return {"response": "Before I can help further, could you confirm you consent to data processing under GDPR?"}


def gate_content(state: ChatState) -> dict:
    """Decide which ChromaDB collection(s) this visitor can query."""
    if state["intent"] in ("job_opportunity", "professional"):
        level = "full" if state["has_gdpr_consent"] else "professional_only"
    else:
        level = "professional_only"
    return {"access_level": level}


def retrieve_context(state: ChatState) -> dict:
    """ChromaDB retrieval, scoped to the access level decided above."""
    # TODO: replace with your actual ChromaDB query, filtered by collection
    # collection = "personal_docs" if state["access_level"] == "full" else "professional_docs"
    context = f"[retrieved context for access_level={state['access_level']}]"
    return {"retrieved_context": context}


def generate_response(state: ChatState) -> dict:
    """Main generation call to Gemma via Ollama."""
    # TODO: replace with your actual Ollama call, e.g.
    # response = ollama_client.generate(model="gemma:4e4b", prompt=..., context=state["retrieved_context"])
    response = f"[generated answer using: {state['retrieved_context']}]"
    should_notify = state["intent"] == "job_opportunity"
    return {"response": response, "should_notify_coach": should_notify}


def log_analytics(state: ChatState) -> dict:
    # TODO: write visitor_id, intent, access_level to your admin dashboard DB
    return {}


def notify_career_coach(state: ChatState) -> dict:
    if state.get("should_notify_coach"):
        # TODO: trigger your existing email notification
        pass
    return {}


def send_visitor_email(state: ChatState) -> dict:
    # TODO: your GDPR-compliant visitor receipt email
    return {}


# ---------------------------------------------------------------------------
# 3. Conditional routing functions — decide which edge to take next.
# ---------------------------------------------------------------------------

def route_after_classify(state: ChatState) -> str:
    if state["intent"] == "spam":
        return "log_and_deflect"
    if not state["has_gdpr_consent"] and state["intent"] != "spam":
        return "request_consent"
    return "gate_content"


# ---------------------------------------------------------------------------
# 4. Build the graph.
# ---------------------------------------------------------------------------
graph = StateGraph(ChatState)

graph.add_node("classify_intent", classify_intent)
graph.add_node("log_and_deflect", log_and_deflect)
graph.add_node("request_consent", request_consent)
graph.add_node("gate_content", gate_content)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("generate_response", generate_response)
graph.add_node("log_analytics", log_analytics)
graph.add_node("notify_career_coach", notify_career_coach)
graph.add_node("send_visitor_email", send_visitor_email)

graph.set_entry_point("classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    route_after_classify,
    {
        "log_and_deflect": "log_and_deflect",
        "request_consent": "request_consent",
        "gate_content": "gate_content",
    },
)

graph.add_edge("log_and_deflect", END)
graph.add_edge("request_consent", END)  # ends turn; resumes once consent is given
graph.add_edge("gate_content", "retrieve_context")
graph.add_edge("retrieve_context", "generate_response")
graph.add_edge("generate_response", "log_analytics")
graph.add_edge("log_analytics", "notify_career_coach")
graph.add_edge("notify_career_coach", "send_visitor_email")
graph.add_edge("send_visitor_email", END)

# MemorySaver gives you per-visitor checkpointing "for free" — swap for
# PostgresSaver/SqliteSaver if you want it to survive server restarts.
app = graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 5. Example call — this is what your Streamlit frontend would invoke.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "visitor-123"}}
    result = app.invoke(
        {
            "visitor_id": "visitor-123",
            "message": "Hi, I'm hiring for a data science role — can you tell me about your ML experience?",
            "has_gdpr_consent": True,
            "intent": None,
            "access_level": None,
            "retrieved_context": None,
            "response": None,
            "should_notify_coach": False,
        },
        config=config,
    )
    print(result["response"])
