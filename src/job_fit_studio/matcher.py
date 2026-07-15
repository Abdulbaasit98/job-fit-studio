"""
The core of job-fit-studio: for each parsed job requirement, find the
best-matching CV chunk (same embed-and-compare idea as the papers RAG
project, just with your CV as the corpus and a job requirement as the
query), and aggregate into an overall fit report.

Two numbers matter, not one:
  - COVERAGE: what fraction of requirements had a reasonably confident
    match somewhere in your CV. This is the number the capability gate
    uses -- it answers "how much of what they're asking for do you
    actually have evidence for?"
  - mean similarity: a secondary confidence signal (how STRONG the
    matches were, not just how many).

Reporting coverage as "7 of 10 requirements matched" rather than a single
blended score is deliberate: it's interpretable, and it directly produces
the list of UNMATCHED requirements the gap-recommendation branch needs.
"""
from dataclasses import dataclass

import numpy as np

from .cv_corpus import CapabilityChunk
from .embeddings import EmbeddingModel


@dataclass
class RequirementMatch:
    requirement: str
    best_chunk: CapabilityChunk
    similarity: float
    matched: bool  # similarity >= threshold


@dataclass
class FitReport:
    matches: list          # list of RequirementMatch, one per requirement
    coverage: float          # fraction of requirements matched
    mean_similarity: float    # average best-match similarity across all requirements
    threshold: float

    @property
    def matched_requirements(self):
        return [m for m in self.matches if m.matched]

    @property
    def unmatched_requirements(self):
        return [m for m in self.matches if not m.matched]


def _cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_fit(requirements: list, corpus: list, embedding_model: EmbeddingModel,
              threshold: float = 0.15) -> FitReport:
    """requirements: list of requirement strings (from requirements.py).
    corpus: list of CapabilityChunk (from cv_corpus.py).
    threshold: minimum cosine similarity to count as "matched". NOTE:
    this needs calibration per embedding backend -- TF-IDF and neural
    embeddings produce similarity scores on different scales, same
    lesson as traffic-switch's per-camera calibration. 0.15 is a
    starting point for TF-IDF; expect to need a different value for
    SentenceTransformerEmbedding once you switch to it.
    """
    if not requirements:
        raise ValueError("No requirements to score -- check that parse_requirements() "
                          "actually extracted lines from the posting text.")

    corpus_texts = [c.text for c in corpus]
    corpus_embeddings = embedding_model.embed(corpus_texts)
    requirement_embeddings = embedding_model.embed(requirements)

    matches = []
    for req_text, req_emb in zip(requirements, requirement_embeddings):
        similarities = [_cosine_similarity(req_emb, chunk_emb) for chunk_emb in corpus_embeddings]
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        matches.append(RequirementMatch(
            requirement=req_text,
            best_chunk=corpus[best_idx],
            similarity=best_score,
            matched=best_score >= threshold,
        ))

    coverage = sum(m.matched for m in matches) / len(matches)
    mean_similarity = sum(m.similarity for m in matches) / len(matches)

    return FitReport(matches=matches, coverage=coverage, mean_similarity=mean_similarity, threshold=threshold)
