"""
Health check for the resume chatbot's LLM-based intent classifier.

Queries Langfuse for 'classification_fallback' scores from the last hour.
If more than FALLBACK_THRESHOLD of them are 1.0 (meaning classify_intent_with_llm
had to fall back to 'professional' due to an Ollama error or unparseable
response), sends an email alert using the same SMTP pattern as app.py's
existing notify_of_new_visitor()/notify_career_coach() functions.

Run on a schedule via cron, e.g. every 30 minutes:
    */30 * * * * cd /home/liberty/resume-bot && /usr/bin/python3 check_health.py >> /home/liberty/resume-bot/health_check.log 2>&1

Reads Langfuse + SMTP credentials from .streamlit/secrets.toml — same file
app.py already uses, so nothing new to configure.

NOTE: this hits Langfuse's stable public REST API (GET /api/public/scores)
directly with requests + HTTP Basic Auth, rather than going through the SDK's
langfuse.api.* wrapper. That wrapper's newer high-performance query methods
(api.scores, api.observations, api.metrics) are Cloud-only per Langfuse's own
docs — self-hosted instances may not expose that surface at all, so chasing
the right SDK method name for a self-hosted deployment is the wrong fight.
The plain REST endpoint below is the long-standing, version-stable one.
"""
import tomllib
import smtplib
import requests
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

FALLBACK_THRESHOLD = 0.10  # alert if more than 10% of classifications fell back
LOOKBACK_HOURS = 1


def load_secrets():
    with open(".streamlit/secrets.toml", "rb") as f:
        return tomllib.load(f)


def check_fallback_rate(secrets) -> tuple[float, int]:
    """Returns (fallback_rate, total_checked). Returns (0.0, 0) if there's
    no data in the window — treated as healthy, not alarming, since low
    traffic on a personal site is normal, not a failure."""
    public_key = secrets["langfuse_public_key"]
    secret_key = secrets["langfuse_secret_key"]
    host = secrets.get("langfuse_host", "http://localhost:3000")

    since = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    response = requests.get(
        f"{host}/api/public/scores",
        params={"name": "classification_fallback", "fromTimestamp": since.isoformat()},
        auth=(public_key, secret_key),
        timeout=10,
    )
    response.raise_for_status()
    scores = response.json().get("data", [])
    values = [s["value"] for s in scores if s.get("value") is not None]

    if not values:
        return 0.0, 0

    fallback_rate = sum(values) / len(values)
    return fallback_rate, len(values)


def send_alert_email(secrets, fallback_rate, total_checked):
    smtp_user = secrets.get("smtp_user")
    smtp_password = secrets.get("smtp_password")
    notify_to = secrets.get("notify_email")

    if not (smtp_user and smtp_password and notify_to):
        print("[Warning] Can't send alert — smtp_user/smtp_password/notify_email missing from secrets.toml")
        return

    notify_to_list = [addr.strip() for addr in notify_to.split(",") if addr.strip()]
    body = (
        f"Your resume chatbot's LLM-based intent classifier is falling back "
        f"to 'professional' more than expected.\n\n"
        f"  Fallback rate: {fallback_rate:.0%}\n"
        f"  Messages checked (last {LOOKBACK_HOURS}h): {total_checked}\n"
        f"  Threshold: {FALLBACK_THRESHOLD:.0%}\n\n"
        f"This usually means Ollama is erroring or returning unparseable responses. "
        f"Check the Langfuse dashboard for traces tagged 'classification_fallback' "
        f"to see the actual error messages, and confirm Ollama is running:\n"
        f"  systemctl status ollama   (or however you're running it)\n\n"
        f"— Resume Bot Health Check"
    )
    msg = MIMEText(body)
    msg["Subject"] = f"⚠️ Resume bot classifier fallback rate: {fallback_rate:.0%}"
    msg["From"] = smtp_user
    msg["To"] = ", ".join(notify_to_list)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, notify_to_list, msg.as_string())

    print(f"[+] Alert email sent — fallback rate {fallback_rate:.0%} over {total_checked} messages.")


if __name__ == "__main__":
    secrets = load_secrets()
    fallback_rate, total_checked = check_fallback_rate(secrets)

    print(f"{datetime.now().isoformat()} — fallback rate: {fallback_rate:.0%} ({total_checked} messages checked)")

    if total_checked > 0 and fallback_rate > FALLBACK_THRESHOLD:
        send_alert_email(secrets, fallback_rate, total_checked)
