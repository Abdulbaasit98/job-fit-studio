import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from job_fit_studio.corpus_builder import build_full_corpus
from job_fit_studio.cv_corpus import CV_CORPUS, CapabilityChunk


def test_build_full_corpus_with_no_sources_returns_hand_authored_only():
    corpus = build_full_corpus()
    assert len(corpus) == len(CV_CORPUS)


def test_build_full_corpus_can_exclude_hand_authored():
    corpus = build_full_corpus(include_hand_authored=False)
    assert corpus == []


def test_build_full_corpus_includes_uploaded_documents(tmp_path):
    (tmp_path / "resume.txt").write_text(
        "A real paragraph describing a project I built and shipped with tests."
    )
    corpus = build_full_corpus(uploaded_docs_dir=str(tmp_path))
    assert len(corpus) == len(CV_CORPUS) + 1
    uploaded = [c for c in corpus if c.kind == "uploaded"]
    assert len(uploaded) == 1


def test_build_full_corpus_includes_github_repos():
    fake_repos = [{"name": "some-repo", "description": "A real project", "fork": False,
                   "owner": {"login": "someone"}}]
    readme_response = {"content": ""}

    def fake_get(url, headers=None, timeout=None):
        mock_resp = MagicMock()
        mock_resp.json.return_value = readme_response if "readme" in url else fake_repos
        return mock_resp

    with patch("job_fit_studio.github_ingest.requests.get", side_effect=fake_get):
        corpus = build_full_corpus(github_username="someone")

    github_chunks = [c for c in corpus if c.chunk_id.startswith("github-")]
    assert len(github_chunks) == 1
    assert github_chunks[0].title == "some-repo"


def test_build_full_corpus_combines_all_three_sources(tmp_path):
    (tmp_path / "notes.txt").write_text(
        "A real paragraph about additional project context worth including."
    )
    fake_repos = [{"name": "repo-x", "description": "desc", "fork": False, "owner": {"login": "u"}}]

    def fake_get(url, headers=None, timeout=None):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": ""} if "readme" in url else fake_repos
        return mock_resp

    with patch("job_fit_studio.github_ingest.requests.get", side_effect=fake_get):
        corpus = build_full_corpus(uploaded_docs_dir=str(tmp_path), github_username="u")

    assert len(corpus) == len(CV_CORPUS) + 1 + 1


def test_build_full_corpus_raises_on_duplicate_chunk_ids(monkeypatch):
    """Simulates two sources accidentally producing the same chunk_id --
    should raise loudly rather than silently keeping only one."""
    duplicate = CapabilityChunk(chunk_id="proj-diabetes", kind="uploaded",
                                 title="dup", text="dup text")

    import job_fit_studio.corpus_builder as cb
    monkeypatch.setattr(cb, "ingest_directory", lambda *a, **k: [duplicate])

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        build_full_corpus(uploaded_docs_dir="/fake/path")
