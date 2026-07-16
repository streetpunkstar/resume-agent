"""
Exports professional-category chunks from ChromaDB to a plain JSON file.

Run this from your MAIN environment (same one as app.py, NOT crewai-env) —
it uses chromadb_setup.py, which is already known-good for this project's
chromadb version. job_assistant.py (in crewai-env) reads the JSON output
instead of connecting to ChromaDB itself, so the two environments never
need matching chromadb versions.

Run manually whenever your resume data changes:
    python export_background.py

Or wire this into a cron job / systemd timer if you want it to stay fresh
automatically after you update your resume vectors.
"""
import json
from chromadb_setup import get_collection

OUTPUT_FILE = "./professional_background.json"


def export_professional_background():
    collection = get_collection()
    results = collection.get(where={"$and": [{"type": "content"}, {"category": "professional"}]})
    chunks = results.get("documents", [])

    if not chunks:
        print("No professional-category chunks found — check your ChromaDB collection.")
        return

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"chunks": chunks}, f, indent=2)

    print(f"Exported {len(chunks)} chunks to {OUTPUT_FILE}.")


if __name__ == "__main__":
    export_professional_background()
