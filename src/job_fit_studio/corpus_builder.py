"""
Merges all three corpus sources into one list the matcher can use:
  1. Hand-authored chunks (cv_corpus.CV_CORPUS) -- your core projects,
     written once with careful evidence fields.
  2. Uploaded documents (document_ingest.ingest_directory) -- your resume
     file, portfolio write-ups, past cover letters, anything you drop in
     a folder.
  3. GitHub repos (github_ingest.build_github_corpus) -- pulled live,
     stays current automatically as you push new work.

This is the "upload once, reuse for every job" piece: run build_full_corpus()
once (or whenever you've added new work) to produce the current corpus,
rather than hand-editing cv_corpus.py by hand every time.
"""
from .cv_corpus import CV_CORPUS, CapabilityChunk
from .document_ingest import ingest_directory
from .github_ingest import build_github_corpus


def build_full_corpus(uploaded_docs_dir: str = None, github_username: str = None,
                       github_token: str = None, max_github_repos: int = None,
                       include_hand_authored: bool = True) -> list:
    """Assembles the complete corpus from whichever sources you provide.
    All arguments are optional -- call with none of them and you get just
    the hand-authored CV_CORPUS; add uploaded_docs_dir and/or
    github_username to pull in more.

    max_github_repos: limits GitHub ingestion to your N most recently
    pushed repos (see github_ingest.fetch_user_repos for why this sorts
    by recent activity rather than an arbitrary cut).
    """
    corpus = []

    if include_hand_authored:
        corpus.extend(CV_CORPUS)

    if uploaded_docs_dir:
        corpus.extend(ingest_directory(uploaded_docs_dir, kind="uploaded"))

    if github_username:
        corpus.extend(build_github_corpus(github_username, token=github_token,
                                            max_repos=max_github_repos))

    _warn_on_duplicate_ids(corpus)
    return corpus


def _warn_on_duplicate_ids(corpus: list):
    """Duplicate chunk_ids would silently confuse anything that looks up
    a chunk by id later (e.g. citing evidence back to the user). This
    can genuinely happen -- e.g. a hand-authored 'proj-traffic' chunk AND
    a GitHub-derived 'github-traffic-switch' chunk both describing the
    same real project under different ids isn't a collision, but two
    sources producing the literal same id would be. Fail loudly rather
    than silently dropping one."""
    seen = set()
    for chunk in corpus:
        if chunk.chunk_id in seen:
            raise ValueError(
                f"Duplicate chunk_id '{chunk.chunk_id}' found while building corpus -- "
                "this would cause silent data loss in downstream matching. "
                "Check for the same file/repo being ingested twice."
            )
        seen.add(chunk.chunk_id)
