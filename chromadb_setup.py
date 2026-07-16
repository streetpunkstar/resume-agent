"""
Shared ChromaDB access for both app.py and graph/pipeline.py.

This is extracted straight out of app.py's original get_db_version()/get_vector_db()
functions — same caching behavior (auto-invalidates when the DB file's mtime
changes), just in one place so app.py and the LangGraph pipeline can't drift
into querying different collections.
"""
import os
import chromadb
import streamlit as st

DB_FOLDER = "./resume_vectors_db"
COLLECTION_NAME = "resume_data"


def get_db_version():
    """Returns the ChromaDB file's last-modified time, used to auto-invalidate the cache below."""
    db_file = os.path.join(DB_FOLDER, "chroma.sqlite3")
    if os.path.isfile(db_file):
        return os.path.getmtime(db_file)
    return None


@st.cache_resource
def _load_collection(_db_version):
    client = chromadb.PersistentClient(path=DB_FOLDER)
    return client.get_collection(name=COLLECTION_NAME)


def get_collection():
    """Call this rather than importing a pre-built `collection` object.
    Keeps the original cache-invalidation-on-file-change behavior, and lets
    each caller (app.py at startup, graph/pipeline.py per-message) handle
    load failures in its own way instead of one import blowing up the other."""
    return _load_collection(get_db_version())
