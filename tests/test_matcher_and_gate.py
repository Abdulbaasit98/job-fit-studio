import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_fit_studio.cv_corpus import CV_CORPUS
from job_fit_studio.embeddings import TfidfEmbedding
from job_fit_studio.gate import evaluate_gate
from job_fit_studio.matcher import score_fit
from job_fit_studio.requirements import parse_requirements


def make_fitted_embedder():
    """TfidfEmbedding must be fit on the combined vocabulary of BOTH the
    CV corpus and whatever requirements we'll score -- otherwise words
    that appear only in a requirement (never in the CV) would be unknown
    to the vectorizer and silently ignored, which could make a real
    match score artificially low. Real requirement text is included in
    the fit corpus in each test for this reason."""
    return TfidfEmbedding()


# A posting genuinely well-matched to this CV -- built directly from
# skills/projects that ARE in CV_CORPUS, so we know what a good match
# should look like.
STRONG_MATCH_POSTING = """
- Experience with Python and PyTorch for deep learning
- Familiarity with FastAPI and Docker for model deployment
- Experience with computer vision and object detection (YOLO)
- Familiarity with MLflow for experiment tracking
- Experience with RAG systems and vector databases
"""

# A posting requiring things genuinely absent from this CV (Kubernetes,
# Go, iOS development) -- deliberately chosen to have almost no overlap.
WEAK_MATCH_POSTING = """
- 5+ years of production Kubernetes cluster administration
- Expert-level Go backend development
- Native iOS development with Swift and SwiftUI
- Experience managing large-scale Kafka streaming infrastructure
"""


def test_strong_match_posting_scores_high_coverage():
    embedder = make_fitted_embedder()
    requirements = parse_requirements(STRONG_MATCH_POSTING)
    corpus_texts = [c.text for c in CV_CORPUS]
    embedder.fit(corpus_texts + requirements)

    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)

    assert report.coverage >= 0.6, (
        f"Expected high coverage for a well-matched posting, got {report.coverage:.0%}. "
        f"Unmatched: {[m.requirement for m in report.unmatched_requirements]}"
    )


def test_weak_match_posting_scores_low_coverage():
    embedder = make_fitted_embedder()
    requirements = parse_requirements(WEAK_MATCH_POSTING)
    corpus_texts = [c.text for c in CV_CORPUS]
    embedder.fit(corpus_texts + requirements)

    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)

    assert report.coverage <= 0.4, (
        f"Expected low coverage for a genuinely unmatched posting, got {report.coverage:.0%}"
    )


def test_matcher_returns_one_match_per_requirement():
    embedder = make_fitted_embedder()
    requirements = parse_requirements(STRONG_MATCH_POSTING)
    embedder.fit([c.text for c in CV_CORPUS] + requirements)
    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)
    assert len(report.matches) == len(requirements)


def test_matcher_raises_on_empty_requirements():
    embedder = make_fitted_embedder()
    embedder.fit([c.text for c in CV_CORPUS])
    try:
        score_fit([], CV_CORPUS, embedder)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pytorch_requirement_matches_a_pytorch_project_specifically():
    """Not just 'coverage is high' -- verify the SPECIFIC match makes
    sense. A PyTorch requirement should match one of the PyTorch-tagged
    chunks, not an unrelated one like the car classifier's evidence text
    matching purely by coincidence."""
    embedder = make_fitted_embedder()
    requirements = ["Strong experience with PyTorch for deep learning model development"]
    embedder.fit([c.text for c in CV_CORPUS] + requirements)
    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)

    matched_chunk_id = report.matches[0].best_chunk.chunk_id
    assert matched_chunk_id in {"skill-core-ml", "proj-rnn-mlflow", "proj-carclassifier",
                                 "exp-graduate-researcher"}, (
        f"PyTorch requirement matched an unexpected chunk: {matched_chunk_id}"
    )


# --- Capability gate tests ---

def test_gate_proceeds_to_draft_on_strong_match():
    embedder = make_fitted_embedder()
    requirements = parse_requirements(STRONG_MATCH_POSTING)
    embedder.fit([c.text for c in CV_CORPUS] + requirements)
    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)

    decision = evaluate_gate(report, min_coverage=0.6)

    assert decision.proceed_to_draft is True
    assert decision.unmatched_requirements == []


def test_gate_recommends_gaps_on_weak_match():
    embedder = make_fitted_embedder()
    requirements = parse_requirements(WEAK_MATCH_POSTING)
    embedder.fit([c.text for c in CV_CORPUS] + requirements)
    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)

    decision = evaluate_gate(report, min_coverage=0.6)

    assert decision.proceed_to_draft is False
    assert len(decision.unmatched_requirements) > 0


def test_gate_reason_is_human_readable():
    embedder = make_fitted_embedder()
    requirements = parse_requirements(STRONG_MATCH_POSTING)
    embedder.fit([c.text for c in CV_CORPUS] + requirements)
    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)
    decision = evaluate_gate(report, min_coverage=0.6)
    assert "%" in decision.reason
    assert isinstance(decision.reason, str) and len(decision.reason) > 10
