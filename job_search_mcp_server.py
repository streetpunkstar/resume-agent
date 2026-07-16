"""
The Adzuna job search tool from job_search_agent.py, wrapped as a real MCP
server — usable by Claude Desktop, Cursor, or any other MCP-compatible
client, instead of being hardcoded into one script.

Install:
    pip install mcp requests

Run standalone to test with the official inspector (no LLM involved,
just a UI for listing tools and testing invocations):
    npx @modelcontextprotocol/inspector python3 job_search_mcp_server.py

To use it from Claude Desktop, add to your MCP config
(~/Library/Application Support/Claude/claude_desktop_config.json on Mac):
    {
      "mcpServers": {
        "job-search": {
          "command": "python3",
          "args": ["/home/liberty/resume-bot/job_search_mcp_server.py"]
        }
      }
    }
"""
import tomllib
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("job-search")


def load_secrets():
    with open(".streamlit/secrets.toml", "rb") as f:
        return tomllib.load(f)


@mcp.tool()
def search_jobs(query: str, country: str = "us", results: int = 10) -> list[dict]:
    """Search for jobs using Adzuna's job aggregator API.

    Args:
        query: Job title and/or key skills to search for (e.g. "Data Scientist Python")
        country: Two-letter country code — us, gb, de, fr, nl, etc.
        results: Number of results to return (max ~50)
    """
    secrets = load_secrets()
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {
        "app_id": secrets["adzuna_app_id"],
        "app_key": secrets["adzuna_app_key"],
        "what": query,  # the parameter name we spent an hour finding earlier today
        "results_per_page": results,
    }
    response = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=15)
    response.raise_for_status()

    jobs = response.json().get("results", [])
    return [
        {
            "title": job.get("title"),
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "contract_type": job.get("contract_type", "not specified"),
            "description": job.get("description", ""),
            "url": job.get("redirect_url"),
        }
        for job in jobs
    ]


if __name__ == "__main__":
    mcp.run()
