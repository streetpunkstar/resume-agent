"""
LangGraph scaffold for the resume chatbot's intake + gating pipeline.

This replaces a linear function-call pipeline with an explicit state graph.
Plug your existing Ollama / ChromaDB / email logic into the marked TODOs.

Install:
    pip install langgraph langchain-core better-profanity --break-system-packages
"""

import os
import tomllib  # stdlib since Python 3.11 — no extra dependency needed
from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from better_profanity import profanity

profanity.load_censor_words()  # loads the library's maintained wordlist once at import time


def _load_langfuse_credentials():
    """Loads Langfuse credentials into os.environ if not already set there.

    Priority: (1) env vars already set — covers app.py, which sets them from
    st.secrets before importing this module, and covers a manual `export` if
    you ever want one. (2) Falls back to reading .streamlit/secrets.toml
    directly — the same file Streamlit already uses — so standalone scripts
    (this file's __main__ block, eval_classify_intent.py) pick up credentials
    automatically without you exporting them by hand every session.

    secrets.toml stays the single source of truth either way; nothing here
    duplicates the actual key values anywhere else.
    """
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        return

    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.isfile(secrets_path):
        return  # no secrets file found from this working directory — tracing stays off

    with open(secrets_path, "rb") as f:
        secrets = tomllib.load(f)

    if secrets.get("langfuse_public_key"):
        os.environ["LANGFUSE_PUBLIC_KEY"] = secrets["langfuse_public_key"]
        os.environ["LANGFUSE_SECRET_KEY"] = secrets["langfuse_secret_key"]
        os.environ["LANGFUSE_HOST"] = secrets.get("langfuse_host", "http://localhost:3000")


_load_langfuse_credentials()

# Optional Langfuse tracing — only activates if credentials were found above.
# Kept as a no-op otherwise so `python -m graph.pipeline` still works without
# Langfuse running.
langfuse_handler = None
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()

# `collection` is your existing ChromaDB collection — now defined once in
# chromadb_setup.py and shared between app.py and this pipeline, instead of
# being set up separately in each place. get_collection() is cached
# (st.cache_resource under the hood), so calling it per-message is cheap.
from chromadb_setup import get_collection


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
    links_text: Optional[str]
    response: Optional[str]
    should_notify_coach: bool


# ---------------------------------------------------------------------------
# 2. Nodes — each one is a plain function: state in, partial state out.
# ---------------------------------------------------------------------------

# Kept for reference / potential reuse elsewhere (e.g. admin-side tagging),
# but no longer used by classify_intent directly — see classify_intent_with_llm
# below. Keyword lists couldn't generalize past English; the eval dataset
# (eval_classify_intent.py) caught this concretely: 3/4 German test messages
# misclassified, including German profanity slipping past better-profanity's
# English-focused wordlist entirely.
PERSONAL_TOPIC_KEYWORDS = [
    "hobby", "hobbies", "favorite", "favourite", "fun fact", "fun facts",
    "outside of work", "outside work", "free time", "spare time", "weekend",
    "weekends", "leisure", "pastime", "pastimes", "personal life", "for fun",
    "food", "movie", "movies", "show", "shows", "book", "books", "music",
    "band", "bands", "game", "games", "sport", "sports", "travel", "pet",
    "pets", "vacation", "holiday",
]

# Still used as a cheap, deterministic first-pass filter — no need to spend
# an Ollama call on obviously-spam patterns like raw URLs.
SPAM_KEYWORDS = ["viagra", "http://", "click here"]

JOB_OPPORTUNITY_KEYWORDS = [
    "hire", "hiring", "opportunity", "role", "position", "interview",
    "job offer", "openings", "open role", "join our team", "join the team",
    "recruiting", "recruiter", "would you be interested", "available to talk",
    "schedule a call", "next steps",
]

# Same model your main chatbot runs — duplicated from app.py's MODEL_NAME
# constant since the two files don't share a common config module yet.
OLLAMA_MODEL = "gemma4:e4b"

CLASSIFICATION_PROMPT = """You are a strict message classifier for a professional resume chatbot. Classify the visitor's message below into exactly ONE of these categories, regardless of what language the message is written in:

- spam: contains profanity, hate speech, harassment, or is clearly junk/abusive (in any language)
- job_opportunity: the VISITOR is offering the candidate a job, an interview, or wants to discuss next steps in a hiring process (e.g. "would you like an interview", "we have an opening", "can we schedule a call")
- personal: asking about hobbies, personal life, or interests outside of work
- professional: anything else related to skills, background, or work experience — including the visitor simply asking what the candidate's current role, title, or job is

Important: a visitor asking ABOUT the candidate's role/title (e.g. "what is your role?", "what's your current position?") is 'professional', NOT 'job_opportunity' — only classify as job_opportunity when the visitor is the one offering something.

Respond with ONLY the single category word — no punctuation, no explanation.

Message: {message}
Category:"""


def classify_intent_with_llm(message: str) -> str:
    """Language-agnostic classification via a local Ollama call. Falls back
    to 'professional' on any parsing failure or Ollama error — the safest
    default, since it neither blocks a real visitor nor wrongly fires the
    career-coach notification.

    Also logs a 'classification_fallback' score to Langfuse whenever that
    fallback triggers — otherwise this failure mode is invisible (previously
    just a print statement to a terminal nobody watches). This is what a
    health-check script can alert on later.
    """
    import ollama

    valid = {"spam", "job_opportunity", "personal", "professional"}
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}],
            options={"temperature": 0},
        )
        raw = response["message"]["content"].strip().lower()
        # Model sometimes adds punctuation or a short explanation despite
        # instructions — take the first valid category word found.
        for category in valid:
            if category in raw:
                _log_fallback_score(0.0)
                return category
        _log_fallback(f"unparseable response: {raw!r}")
    except Exception as e:
        _log_fallback(f"Ollama call failed: {e}")
    return "professional"


def _log_fallback_score(value: float, comment: str = None):
    """Logs the classification_fallback score onto the currently-active
    trace. Relies on the OTel context the LangChain callback handler already
    sets up during a traced graph run — score_current_trace() attaches to
    whatever trace is active in that context, no trace_id needed explicitly."""
    if not langfuse_handler:
        return
    try:
        from langfuse import get_client
        get_client().score_current_trace(
            name="classification_fallback", value=value, data_type="NUMERIC", comment=comment
        )
    except Exception as e:
        print(f"[Warning] Failed to log classification_fallback score to Langfuse: {e}")


def _log_fallback(reason: str):
    print(f"[Warning] classify_intent_with_llm falling back to 'professional': {reason}")
    _log_fallback_score(1.0, comment=reason)


def classify_intent(state: ChatState) -> dict:
    """Cheap keyword/profanity pre-filter catches obvious English spam
    without spending an Ollama call. Everything else goes through the LLM
    classifier, which — unlike the old keyword lists — generalizes to any
    language. Confirmed against eval_classify_intent.py after the swap."""
    message = state["message"]
    lowered = message.lower()
    if any(k in lowered for k in SPAM_KEYWORDS) or profanity.contains_profanity(message):
        intent = "spam"
    else:
        intent = classify_intent_with_llm(message)
    return {"intent": intent}


def log_and_deflect(state: ChatState) -> dict:
    """Spam/suspicious path — log to admin dashboard, skip generation entirely,
    and close out the session (app.py sets a 'blocked' flag after this)."""
    # TODO: write to your admin analytics store
    return {
        "response": (
            "Thanks for stopping by, but this doesn't look like a genuine "
            "question about my background, so I'll close things out here. "
            "Take care!"
        )
    }


def request_consent(state: ChatState) -> dict:
    """GDPR gate — return a consent prompt instead of proceeding."""
    return {"response": "Before I can help further, could you confirm you consent to data processing under GDPR?"}


def gate_content(state: ChatState) -> dict:
    """Decide which ChromaDB collection(s) this visitor can query.
    Mirrors app.py's original rule exactly: personal-category chunks are
    only allowed in when the message itself reads as a personal-topic
    question — not based on GDPR consent or professional/job-opp framing."""
    level = "full" if state["intent"] == "personal" else "professional_only"
    return {"access_level": level}


def retrieve_context(state: ChatState) -> dict:
    """ChromaDB retrieval — mirrors app.py's existing filter logic exactly,
    but driven off the graph's intent classification instead of a standalone
    is_personal_query() call inline in the UI code.

    Wrapped in error handling because an Ollama outage here previously
    crashed the whole graph.invoke() call unhandled — classify_intent's own
    LLM call has a try/except and degrades gracefully, but this one didn't,
    so a visitor got a blank response and a raw Python traceback instead of
    any answer at all. Caught here now for the same reason.
    """
    import ollama

    collection = get_collection()

    try:
        query_vector = ollama.embeddings(model="nomic-embed-text", prompt=state["message"])["embedding"]
    except Exception as e:
        print(f"[Warning] retrieve_context: Ollama embeddings call failed, returning empty context: {e}")
        return {
            "retrieved_context": "I'm having trouble accessing my background information right now — please try again in a moment.",
            "links_text": "",
        }

    if state["access_level"] == "full":
        query_filter = {"type": "content"}
    else:
        query_filter = {"$and": [{"type": "content"}, {"category": "professional"}]}

    results = collection.query(query_embeddings=[query_vector], n_results=3, where=query_filter)
    context_chunks = results.get("documents", [[]])[0]
    context_text = "\n\n".join(context_chunks) if context_chunks else "No matching data found."

    links_result = collection.get(where={"type": "links"})
    links_chunks = links_result.get("documents", [])
    links_text = "\n\n".join(links_chunks) if links_chunks else ""

    return {"retrieved_context": context_text, "links_text": links_text}


def prepare_generation(state: ChatState) -> dict:
    """Graph's job ends here — it hands back the routing decisions and
    retrieved content. app.py still owns the actual Ollama streaming call
    so the Streamlit response_placeholder keeps working exactly as before."""
    should_notify = state["intent"] == "job_opportunity"
    return {"should_notify_coach": should_notify}


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
graph.add_node("prepare_generation", prepare_generation)

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
graph.add_edge("retrieve_context", "prepare_generation")
graph.add_edge("prepare_generation", END)

# NOTE: log_analytics / notify_career_coach / send_visitor_email nodes from the
# original scaffold are removed here — app.py already handles logging via
# log_message()/log_visitor() and coach notification via notify_of_new_visitor().
# Call those from app.py using should_notify_coach from the graph result instead
# of duplicating that logic inside the graph.

# MemorySaver gives you per-visitor checkpointing "for free" — swap for
# PostgresSaver/SqliteSaver if you want it to survive server restarts.
app = graph.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# 5. Example call — this is what your Streamlit frontend would invoke.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "visitor-123"}}
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]
        print("Langfuse tracing enabled for this run.")
    result = app.invoke(
        {
            "visitor_id": "visitor-123",
            "message": "Hi, I'm hiring for a data science role — can you tell me about your ML experience?",
            "has_gdpr_consent": True,
            "intent": None,
            "access_level": None,
            "retrieved_context": None,
            "links_text": None,
            "response": None,
            "should_notify_coach": False,
        },
        config=config,
    )
    print("intent:", result["intent"])
    print("access_level:", result["access_level"])
    print("should_notify_coach:", result["should_notify_coach"])
    print("retrieved_context:", result["retrieved_context"])
