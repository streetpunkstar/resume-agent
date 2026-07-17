# resume-agent

A personal resume chatbot and job search agent.

- `app.py` — Streamlit chatbot that answers questions about the candidate's background using a resume-backed vector store.
- `job_search_agent.py` / `job_search_mcp_server.py` — multi-source job search and lead tracking.
- `build_embeddings.py` / `chromadb_setup.py` — build and manage the resume vector store.
- `check_health.py` — periodic health checks with email alerting.

## Setup

Secrets (Langfuse keys, SMTP credentials, app passwords) are read from `.streamlit/secrets.toml`, which is gitignored. Personal data such as the resume PDF, parsed background JSON, and the job search database are also gitignored and never committed.
