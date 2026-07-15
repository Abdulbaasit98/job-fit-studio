import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from job_fit_studio.cv_corpus import CV_CORPUS
from job_fit_studio.embeddings import TfidfEmbedding
from job_fit_studio.gate import evaluate_gate
from job_fit_studio.letter_writer import write_cover_letter, write_resume_summary
from job_fit_studio.llm_client import FakeLLMClient
from job_fit_studio.matcher import score_fit
from job_fit_studio.requirements import parse_requirements

STRONG_MATCH_POSTING = """
- Strong experience with Python and PyTorch for deep learning
- Experience deploying models via FastAPI or similar REST frameworks
- Experience with computer vision and object detection (YOLO)
- Familiarity with MLflow for experiment tracking
"""

WEAK_MATCH_POSTING = """
- 5+ years of production Kubernetes cluster administration
- Expert-level Go backend development
- Native iOS development with Swift and SwiftUI
"""


def build_report_and_decision(posting_text, threshold=0.1, min_coverage=0.6):
    embedder = TfidfEmbedding()
    requirements = parse_requirements(posting_text)
    embedder.fit([c.text for c in CV_CORPUS] + requirements)
    report = score_fit(requirements, CV_CORPUS, embedder, threshold=threshold)
    decision = evaluate_gate(report, min_coverage=min_coverage)
    return report, decision


def test_write_cover_letter_refuses_when_gate_not_approved():
    """The core safety test: this function must be UNCALLABLE for a
    weak-fit posting, not just discouraged."""
    report, decision = build_report_and_decision(WEAK_MATCH_POSTING)
    assert decision.proceed_to_draft is False  # sanity check on the fixture

    llm = FakeLLMClient()
    with pytest.raises(RuntimeError, match="Refusing to draft"):
        write_cover_letter(decision, report, "some posting", llm, "Test Applicant")

    # and critically: the LLM must never even have been CALLED
    assert len(llm.calls) == 0


def test_write_resume_summary_also_refuses_when_gate_not_approved():
    report, decision = build_report_and_decision(WEAK_MATCH_POSTING)
    llm = FakeLLMClient()
    with pytest.raises(RuntimeError, match="Refusing to draft"):
        write_resume_summary(decision, report, "some posting", llm)
    assert len(llm.calls) == 0


def test_write_cover_letter_succeeds_on_strong_match():
    report, decision = build_report_and_decision(STRONG_MATCH_POSTING)
    assert decision.proceed_to_draft is True  # sanity check

    llm = FakeLLMClient(canned_response="Dear Hiring Manager, ...")
    result = write_cover_letter(decision, report, "AI Engineer role", llm, "Abdulvosit")

    assert result == "Dear Hiring Manager, ..."
    assert len(llm.calls) == 1


def test_prompt_includes_only_matched_chunk_content():
    """The actual grounding test: verify the prompt sent to the LLM
    contains ONLY evidence from chunks that matched a real requirement --
    not the entire CV corpus, and specifically not chunks unrelated to
    this posting (e.g. the car classifier project shouldn't appear in a
    prompt about FastAPI/MLflow/YOLO, unless it happened to match)."""
    report, decision = build_report_and_decision(STRONG_MATCH_POSTING)
    llm = FakeLLMClient()
    write_cover_letter(decision, report, "posting text", llm, "Applicant")

    system_prompt, user_prompt = llm.calls[0]

    matched_titles = {m.best_chunk.title for m in report.matched_requirements}
    unmatched_chunk_ids_in_corpus = {
        c.chunk_id for c in CV_CORPUS
    } - {m.best_chunk.chunk_id for m in report.matched_requirements}

    # every matched chunk's title should appear in the prompt
    for title in matched_titles:
        assert title in user_prompt, f"Matched chunk '{title}' missing from prompt"

    # chunks that were NOT matched to any requirement should not appear,
    # by title, anywhere in the evidence block
    unmatched_titles = [c.title for c in CV_CORPUS if c.chunk_id in unmatched_chunk_ids_in_corpus]
    for title in unmatched_titles:
        assert title not in user_prompt, (
            f"Unmatched chunk '{title}' leaked into the grounding prompt -- "
            "this would let the LLM reference capability that was never verified "
            "as relevant to this specific posting."
        )


def test_prompt_deduplicates_repeated_chunk_matches():
    """If two different requirements both best-match the SAME chunk
    (e.g. two Python-related requirements both matching 'Core ML/DL
    Stack'), that chunk's text should appear only ONCE in the prompt,
    not twice."""
    posting = "- Experience with Python\n- Experience with Python programming for data work\n"
    report, decision = build_report_and_decision(posting, threshold=0.05, min_coverage=0.0)

    llm = FakeLLMClient()
    write_cover_letter(decision, report, "posting", llm, "Applicant")
    _, user_prompt = llm.calls[0]

    # Whatever chunk both requirements matched, its title should appear
    # exactly once in the evidence block, not once per matching requirement
    matched_titles = [m.best_chunk.title for m in report.matched_requirements]
    if len(matched_titles) == 2 and matched_titles[0] == matched_titles[1]:
        assert user_prompt.count(matched_titles[0]) == 1


def test_system_prompt_contains_grounding_instruction():
    """Even though the real safeguard is omission (untested content
    literally isn't in the prompt), the instruction-level safeguard
    should also be present as defense in depth."""
    report, decision = build_report_and_decision(STRONG_MATCH_POSTING)
    llm = FakeLLMClient()
    write_cover_letter(decision, report, "posting", llm, "Applicant")
    system_prompt, _ = llm.calls[0]
    assert "ONLY" in system_prompt
    assert "evidence" in system_prompt.lower()


def test_write_email_improvements_is_a_separate_llm_call():
    """No gate check needed here -- critiquing an already-written letter
    doesn't touch the CV corpus or matching at all, it's a pure text
    review. Verify it calls the LLM exactly once with the letter text."""
    from job_fit_studio.letter_writer import write_email_improvements

    llm = FakeLLMClient(canned_response="Consider a stronger opening sentence.")
    letter_text = "Dear Hiring Manager, I am writing to apply..."

    result = write_email_improvements(letter_text, llm)

    assert result == "Consider a stronger opening sentence."
    assert len(llm.calls) == 1
    _, user_prompt = llm.calls[0]
    assert letter_text in user_prompt


def test_write_email_improvements_system_prompt_instructs_critical_review():
    """The prompt should explicitly push AGAINST simple self-praise --
    verify the instruction to be critical/specific is actually present,
    not just implied."""
    from job_fit_studio.letter_writer import write_email_improvements

    llm = FakeLLMClient()
    write_email_improvements("Some letter text.", llm)
    system_prompt, _ = llm.calls[0]
    assert "critical" in system_prompt.lower() or "weaknesses" in system_prompt.lower()
    assert "specific" in system_prompt.lower()
