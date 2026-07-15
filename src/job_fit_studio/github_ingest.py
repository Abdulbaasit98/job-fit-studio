"""
Pulls your public GitHub repos directly via the GitHub REST API and turns
each one into a CapabilityChunk -- this is what lets the corpus stay
current automatically as you push new/updated projects, instead of you
hand-editing cv_corpus.py every time.

A real bug found while building this (documented here on purpose, same
honesty as every other project): GitHub's API returns rate-limit errors
as a JSON OBJECT with a "message" key, not a list of repos. Naive code
that does `len(response.json())` without checking for this will silently
"succeed" with a nonsense small number (an error dict with 2 keys reads
as "length 2"), rather than failing loudly. fetch_user_repos() below
checks explicitly for this instead of trusting the response shape.
"""
import base64
from dataclasses import dataclass

import requests

from .cv_corpus import CapabilityChunk

GITHUB_API = "https://api.github.com"


class GitHubFetchError(Exception):
    pass


def _get(url: str, token: str = None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=15)
    data = response.json()

    # GitHub returns errors as a JSON object with a "message" key, whether
    # it's a 403 rate limit, a 404 user-not-found, or anything else. A
    # successful repo-list response is always a JSON ARRAY. Checking the
    # type, not just "did the request not throw," is what catches this.
    if isinstance(data, dict) and "message" in data:
        raise GitHubFetchError(
            f"GitHub API error for {url}: {data['message']}"
            + (f" ({data.get('documentation_url')})" if data.get("documentation_url") else "")
        )

    return data


def fetch_user_repos(username: str, token: str = None, include_forks: bool = False,
                       max_repos: int = None) -> list:
    """Returns the raw list of repo dicts from GitHub's API.

    token: optional GitHub Personal Access Token. Unauthenticated requests
    are limited to 60/hour PER SOURCE IP -- fine for occasional personal
    use, but easy to exhaust if you're testing repeatedly or sharing an
    IP (e.g. a shared dev sandbox). A token raises this to 5000/hour.

    max_repos: if set, limits results to your N most RECENTLY PUSHED
    repos (sort=pushed on the API call), not an arbitrary alphabetical
    or creation-order cut. This matters: truncating to "the first 10
    however the API happens to order them" could easily keep old
    abandoned scratch repos and drop your most current, most relevant
    work. Sorting by recent push activity first means a limit favors
    what you're actually working on now.
    """
    repos = _get(f"{GITHUB_API}/users/{username}/repos?per_page=100&sort=pushed&direction=desc",
                 token=token)

    if not include_forks:
        repos = [r for r in repos if not r.get("fork", False)]

    if max_repos is not None:
        repos = repos[:max_repos]

    return repos


def fetch_readme_text(username: str, repo_name: str, token: str = None) -> str:
    """Returns the repo's README content as plain text, or empty string
    if there's no README (not every repo has one -- this is expected,
    not an error)."""
    try:
        data = _get(f"{GITHUB_API}/repos/{username}/{repo_name}/readme", token=token)
    except GitHubFetchError:
        return ""  # no README -- not a failure, just nothing to add

    content_b64 = data.get("content", "")
    try:
        return base64.b64decode(content_b64).decode("utf-8", errors="replace")
    except Exception:
        return ""


def repo_to_chunk(repo: dict, readme_text: str = "", max_readme_chars: int = 800) -> CapabilityChunk:
    """Converts one GitHub repo (+ optional README text) into a
    CapabilityChunk, matching the same structure as the hand-authored
    entries in cv_corpus.py -- so repo-derived chunks and hand-written
    ones are usable identically by the matcher."""
    name = repo["name"]
    description = repo.get("description") or ""
    language = repo.get("language") or ""
    stars = repo.get("stargazers_count", 0)

    text_parts = [description] if description else []
    if readme_text:
        # Truncate long READMEs -- we want a representative summary for
        # matching purposes, not the entire document embedded as one
        # chunk (same "don't blend five subtopics into one vector"
        # reasoning as the papers RAG project's chunking).
        text_parts.append(readme_text[:max_readme_chars])
    if language:
        text_parts.append(f"Primary language: {language}.")

    full_text = " ".join(text_parts).strip() or f"GitHub repository: {name}"

    return CapabilityChunk(
        chunk_id=f"github-{name}",
        kind="project",
        title=name,
        text=full_text,
        evidence=f"github.com/{repo['owner']['login']}/{name}"
                  + (f" ({stars} stars)" if stars else ""),
        tags=[language.lower()] if language else [],
    )


def build_github_corpus(username: str, token: str = None, fetch_readmes: bool = True,
                          max_repos: int = None) -> list:
    """Full pipeline: fetch repos (optionally limited to your max_repos
    most recently active), optionally fetch each README, convert all of
    them to CapabilityChunks. Returns a list ready to merge into the
    overall CV corpus."""
    repos = fetch_user_repos(username, token=token, max_repos=max_repos)
    chunks = []

    for repo in repos:
        readme_text = ""
        if fetch_readmes:
            readme_text = fetch_readme_text(username, repo["name"], token=token)
        chunks.append(repo_to_chunk(repo, readme_text))

    return chunks
