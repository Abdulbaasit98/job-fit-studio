import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from job_fit_studio.github_ingest import (GitHubFetchError, _get, build_github_corpus,
                                            fetch_readme_text, fetch_user_repos, repo_to_chunk)


def make_response(json_data):
    mock_resp = MagicMock()
    mock_resp.json.return_value = json_data
    return mock_resp


# The EXACT error body real GitHub API returned when we hit this for real
# while building this module -- used as a regression test so this
# specific failure mode can never silently regress.
REAL_RATE_LIMIT_RESPONSE = {
    "message": "API rate limit exceeded for 34.60.77.245. (But here's the good news: "
               "Authenticated requests get a higher rate limit. Check out the documentation for more details.)",
    "documentation_url": "https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting",
}


def test_get_raises_clear_error_on_real_rate_limit_response():
    """Regression test for the actual bug hit during development: a
    naive `len(response.json())` on this exact payload silently returns
    2 (the dict has 2 keys), misread as '2 repos found.' This test
    proves _get() raises instead."""
    with patch("job_fit_studio.github_ingest.requests.get",
               return_value=make_response(REAL_RATE_LIMIT_RESPONSE)):
        with pytest.raises(GitHubFetchError, match="rate limit exceeded"):
            _get("https://api.github.com/users/someone/repos")


def test_get_raises_clear_error_on_user_not_found():
    not_found = {"message": "Not Found", "documentation_url": "https://docs.github.com/rest"}
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(not_found)):
        with pytest.raises(GitHubFetchError, match="Not Found"):
            _get("https://api.github.com/users/nonexistent-user-xyz/repos")


def test_get_returns_data_on_successful_list_response():
    repos = [{"name": "repo1"}, {"name": "repo2"}]
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(repos)):
        result = _get("https://api.github.com/users/someone/repos")
    assert result == repos


def test_fetch_user_repos_filters_out_forks_by_default():
    repos = [
        {"name": "my-real-project", "fork": False},
        {"name": "someone-elses-project", "fork": True},
    ]
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(repos)):
        result = fetch_user_repos("someone")
    assert len(result) == 1
    assert result[0]["name"] == "my-real-project"


def test_fetch_user_repos_can_include_forks():
    repos = [{"name": "a", "fork": False}, {"name": "b", "fork": True}]
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(repos)):
        result = fetch_user_repos("someone", include_forks=True)
    assert len(result) == 2


def test_fetch_user_repos_respects_max_repos_limit():
    repos = [{"name": f"repo-{i}", "fork": False} for i in range(20)]
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(repos)):
        result = fetch_user_repos("someone", max_repos=10)
    assert len(result) == 10


def test_fetch_user_repos_no_limit_returns_all():
    repos = [{"name": f"repo-{i}", "fork": False} for i in range(20)]
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(repos)):
        result = fetch_user_repos("someone", max_repos=None)
    assert len(result) == 20


def test_fetch_user_repos_requests_sort_by_pushed():
    """The limit should favor recently-active repos, not an arbitrary
    API-default order -- verify the actual request URL asks the API to
    sort by push date, descending (most recent first)."""
    repos = [{"name": "a", "fork": False}]
    with patch("job_fit_studio.github_ingest.requests.get",
               return_value=make_response(repos)) as mock_get:
        fetch_user_repos("someone", max_repos=10)

    called_url = mock_get.call_args.args[0]
    assert "sort=pushed" in called_url
    assert "direction=desc" in called_url


def test_max_repos_limit_applied_after_fork_filtering():
    """max_repos should limit the FINAL, fork-filtered list to N, not
    slice the raw API response to N and then filter forks out of that
    (which could return fewer than N real repos even when more exist)."""
    # 15 real repos + 10 forks, all in one raw response -- if forks were
    # filtered AFTER slicing to max_repos=10, some real repos could be
    # lost to forks that happened to sort earlier.
    repos = ([{"name": f"real-{i}", "fork": False} for i in range(15)]
             + [{"name": f"forked-{i}", "fork": True} for i in range(10)])
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(repos)):
        result = fetch_user_repos("someone", max_repos=10)

    assert len(result) == 10
    assert all(not r["fork"] for r in result)  # no forks leaked through despite ordering


def test_fetch_readme_text_returns_empty_string_on_missing_readme():
    """A repo with no README isn't an error condition -- should return
    "" gracefully, not raise."""
    not_found = {"message": "Not Found"}
    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(not_found)):
        result = fetch_readme_text("someone", "repo-with-no-readme")
    assert result == ""


def test_fetch_readme_text_decodes_base64_content():
    import base64
    readme_content = "# My Project\n\nThis does something useful."
    encoded = base64.b64encode(readme_content.encode()).decode()
    response = {"content": encoded}

    with patch("job_fit_studio.github_ingest.requests.get", return_value=make_response(response)):
        result = fetch_readme_text("someone", "some-repo")

    assert result == readme_content


def test_repo_to_chunk_builds_correct_chunk():
    repo = {
        "name": "traffic-switch",
        "description": "Density-aware traffic congestion detection",
        "language": "Python",
        "stargazers_count": 3,
        "owner": {"login": "Abdulbaasit98"},
    }
    chunk = repo_to_chunk(repo, readme_text="Full README content about calibration and YOLO.")

    assert chunk.chunk_id == "github-traffic-switch"
    assert chunk.title == "traffic-switch"
    assert "Density-aware traffic congestion" in chunk.text
    assert "calibration" in chunk.text
    assert "github.com/Abdulbaasit98/traffic-switch" in chunk.evidence
    assert "3 stars" in chunk.evidence
    assert chunk.kind == "project"


def test_repo_to_chunk_handles_missing_description_and_readme():
    """A bare repo with no description and no README shouldn't crash or
    produce an empty/garbage chunk -- should fall back to a minimal but
    valid chunk."""
    repo = {"name": "empty-repo", "owner": {"login": "someone"}}
    chunk = repo_to_chunk(repo)
    assert "empty-repo" in chunk.text
    assert chunk.chunk_id == "github-empty-repo"


def test_repo_to_chunk_truncates_long_readme():
    repo = {"name": "big-repo", "owner": {"login": "someone"}}
    long_readme = "word " * 1000  # far longer than max_readme_chars
    chunk = repo_to_chunk(repo, readme_text=long_readme, max_readme_chars=100)
    # the README portion of the text should be capped, not the whole 5000 chars
    assert len(chunk.text) < 200


def test_build_github_corpus_full_pipeline():
    repos_response = [
        {"name": "proj-a", "description": "First project", "fork": False,
         "language": "Python", "owner": {"login": "someone"}},
        {"name": "proj-b", "description": "Second project", "fork": False,
         "language": "Python", "owner": {"login": "someone"}},
    ]
    readme_response = {"content": ""}

    call_count = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        call_count["n"] += 1
        if "readme" in url:
            return make_response(readme_response)
        return make_response(repos_response)

    with patch("job_fit_studio.github_ingest.requests.get", side_effect=fake_get):
        chunks = build_github_corpus("someone")

    assert len(chunks) == 2
    assert {c.title for c in chunks} == {"proj-a", "proj-b"}
