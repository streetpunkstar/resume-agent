"""
Persistent memory for the job search agent — SQLite-backed, survives across
runs. Two kinds of memory, both missing from the original agent:

1. Episodic: which specific postings have already been shown, so re-running
   the agent doesn't re-surface the same jobs every time.
2. Procedural: which requirement patterns (e.g. "C1 German") the user has
   marked as dealbreakers, so future rankings can deprioritize similar
   postings automatically instead of you re-noticing the same problem
   every single run.
"""
import sqlite3
from datetime import datetime

DB_PATH = "job_search_memory.db"


def init_memory():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_jobs (
            url TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            first_seen TEXT,
            feedback TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_dealbreakers (
            pattern TEXT PRIMARY KEY,
            reason TEXT,
            added TEXT
        )
    """)
    conn.commit()
    conn.close()


def filter_unseen(jobs: list[dict]) -> list[dict]:
    """Episodic memory in action: don't re-surface postings already shown
    in a previous run. Operates on RAW Adzuna results (before rank_jobs
    renames fields), so this checks 'redirect_url', not 'url'."""
    conn = sqlite3.connect(DB_PATH)
    seen_urls = {row[0] for row in conn.execute("SELECT url FROM seen_jobs")}
    conn.close()
    return [job for job in jobs if job.get("redirect_url") not in seen_urls]


def record_shown(jobs: list[dict]):
    """Call this after presenting results, so next run knows what's already
    been shown."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    for job in jobs:
        conn.execute(
            "INSERT OR IGNORE INTO seen_jobs (url, title, company, first_seen, feedback) VALUES (?, ?, ?, ?, NULL)",
            (job["url"], job["title"], job["company"], now),
        )
    conn.commit()
    conn.close()


def record_feedback(url: str, feedback: str):
    """feedback: 'good_fit' or 'bad_fit'. This is how a human correction
    (like 'that one needed C1 German') becomes something the agent can
    actually learn from, rather than you noticing the same problem every
    single run."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE seen_jobs SET feedback = ? WHERE url = ?", (feedback, url))
    conn.commit()
    conn.close()


def add_learned_dealbreaker(pattern: str, reason: str):
    """Procedural memory: a pattern (e.g. 'C1 German', 'security clearance')
    the agent should deprioritize in future rankings, learned from past
    feedback rather than hardcoded upfront."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO learned_dealbreakers (pattern, reason, added) VALUES (?, ?, ?)",
        (pattern.lower(), reason, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_learned_dealbreakers() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    patterns = [row[0] for row in conn.execute("SELECT pattern FROM learned_dealbreakers")]
    conn.close()
    return patterns


def check_learned_dealbreakers(description: str) -> bool:
    """Like the hardcoded LANGUAGE_FLAG_PATTERNS check from earlier, but
    the pattern list now grows over time from real feedback instead of
    being fixed at write-time."""
    lowered = description.lower()
    return any(pattern in lowered for pattern in get_learned_dealbreakers())
