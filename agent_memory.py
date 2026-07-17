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
    # Migrations for databases created before these columns existed —
    # CREATE TABLE IF NOT EXISTS doesn't add columns to an already-existing
    # table, so this handles upgrading a DB that's already on disk.
    for column_def in ["description TEXT", "location TEXT", "similarity REAL"]:
        try:
            conn.execute(f"ALTER TABLE seen_jobs ADD COLUMN {column_def}")
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_dealbreakers (
            pattern TEXT PRIMARY KEY,
            reason TEXT,
            added TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_queue (
            url TEXT PRIMARY KEY,
            title TEXT,
            company TEXT,
            run_id TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_to_review_queue(url: str, title: str, company: str, run_id: str):
    """Called once a full application packet (cover letter + ATS resume +
    fact check) has actually been generated for a lead — status starts as
    'pending', meaning a human still needs to review it before anything
    gets submitted anywhere."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO review_queue (url, title, company, run_id, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
        (url, title, company, run_id, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_review_queue(status: str = None) -> list:
    conn = sqlite3.connect(DB_PATH)
    if status:
        rows = conn.execute(
            "SELECT url, title, company, run_id, status, created_at FROM review_queue WHERE status = ? ORDER BY created_at DESC",
            (status,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT url, title, company, run_id, status, created_at FROM review_queue ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [
        {"url": r[0], "title": r[1], "company": r[2], "run_id": r[3], "status": r[4], "created_at": r[5]}
        for r in rows
    ]


def update_review_status(url: str, status: str):
    """status: 'pending', 'approved', or 'rejected'. Approval never triggers
    an actual submission anywhere — it just means a human has reviewed the
    packet and judged it ready. The human still does the actual clicking on
    the job site themselves."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE review_queue SET status = ? WHERE url = ?", (status, url))
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
    been shown. Uses an upsert (INSERT ... ON CONFLICT DO UPDATE) rather than
    INSERT OR IGNORE — a job already in the table gets its description,
    location, and similarity refreshed if a fresher value is available,
    instead of being permanently stuck with whatever (or nothing) was known
    the first time it was ever seen. feedback is deliberately excluded from
    the update clause so re-processing a job never wipes out a human's
    earlier good_fit/bad_fit judgment."""
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    for job in jobs:
        conn.execute("""
            INSERT INTO seen_jobs (url, title, company, first_seen, feedback, description, location, similarity)
            VALUES (?, ?, ?, ?, NULL, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                description = excluded.description,
                location = excluded.location,
                similarity = excluded.similarity
        """, (
            job["url"], job["title"], job["company"], now,
            job.get("description", ""), job.get("location", ""), job.get("similarity"),
        ))
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


def seed_default_dealbreakers():
    """Pre-populates dealbreakers that are already known upfront, rather than
    requiring the user to reactively 'teach' them one at a time after seeing
    a bad match. Uses INSERT OR IGNORE so it's safe to call on every startup —
    never overwrites a pattern the user has since edited or intentionally
    removed. German-fluency patterns aren't included here since they're
    already covered by job_search_agent.py's hardcoded LANGUAGE_FLAG_PATTERNS
    list — this is specifically for constraints that list doesn't cover."""
    defaults = [
        ("no visa sponsorship", "Requires own work authorization, no employer sponsorship offered"),
        ("unable to sponsor", "Requires own work authorization, no employer sponsorship offered"),
        ("does not sponsor", "Requires own work authorization, no employer sponsorship offered"),
        ("eu citizenship required", "Requires EU citizenship, not just EU residency/relocation willingness"),
        ("eu work authorization required", "Requires already having EU work authorization"),
        ("must be eligible to work in the eu", "Requires already having EU work eligibility"),
    ]
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now().isoformat()
    for pattern, reason in defaults:
        conn.execute(
            "INSERT OR IGNORE INTO learned_dealbreakers (pattern, reason, added) VALUES (?, ?, ?)",
            (pattern, reason, now),
        )
    conn.commit()
    conn.close()


def get_learned_dealbreakers() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    patterns = [row[0] for row in conn.execute("SELECT pattern FROM learned_dealbreakers")]
    conn.close()
    return patterns


def check_learned_dealbreakers(description: str) -> list[dict]:
    """Returns every learned dealbreaker that actually matched, with its
    reason — not just a bare boolean. Previously this returned True/False,
    which got folded into a flag mislabeled 'possible_language_requirement'
    regardless of whether the match was actually about language, visa
    status, or anything else. Callers now get enough detail to show an
    honest, specific flag instead of a generic mislabeled one."""
    lowered = description.lower()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT pattern, reason FROM learned_dealbreakers").fetchall()
    conn.close()
    return [{"pattern": pattern, "reason": reason} for pattern, reason in rows if pattern in lowered]


def get_job_tracker() -> list[dict]:
    """The persistent, shared 'what's the state of every job we've found'
    view — addresses a real gap: seen_jobs only existed to avoid re-showing
    postings, with no way to browse jobs found in past runs or see which
    ones still need action. Joins seen_jobs with review_queue so a single
    query gives the full lifecycle status of every lead ever found, visible
    identically to both the admin and coach logins since this data was
    always shared, just never displayed persistently before now."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT
            seen_jobs.url,
            seen_jobs.title,
            seen_jobs.company,
            seen_jobs.first_seen,
            seen_jobs.feedback,
            seen_jobs.description,
            seen_jobs.location,
            seen_jobs.similarity,
            review_queue.status,
            review_queue.run_id
        FROM seen_jobs
        LEFT JOIN review_queue ON seen_jobs.url = review_queue.url
        ORDER BY seen_jobs.first_seen DESC
    """).fetchall()
    conn.close()

    tracker = []
    for url, title, company, first_seen, feedback, description, location, similarity, rq_status, run_id in rows:
        if rq_status:
            derived_status = rq_status  # 'pending', 'approved', or 'rejected'
        elif feedback:
            derived_status = feedback  # 'good_fit' or 'bad_fit', no packet generated yet
        else:
            derived_status = "new"  # found, but genuinely no action taken at all

        tracker.append({
            "url": url,
            "title": title,
            "company": company,
            "first_seen": first_seen,
            "description": description or "",
            "location": location or "",
            "similarity": similarity,
            "status": derived_status,
            "run_id": run_id,
        })
    return tracker
