"""
The full pipeline, as one function -- but deliberately with no input()
calls anywhere in this file. Same principle as letter_writer.py being
separate from gmail_draft.py: keep the actual LOGIC testable without
needing a real terminal, a real GitHub account, or a real LLM/Gmail
connection. cli.py (the interactive entry point) is a thin wrapper around
this that handles the actual asking-questions part.
"""
from dataclasses import dataclass

from .corpus_builder import build_full_corpus
from .email_extractor import pick_primary_email
from .embeddings import EmbeddingModel
from .gate import GateDecision, evaluate_gate
from .gmail_draft import create_application_draft
from .letter_writer import write_cover_letter, write_email_improvements, write_resume_summary
from .llm_client import LLMClient
from .matcher import FitReport, score_fit
from .requirements import parse_requirements


@dataclass
class PipelineResult:
    decision: GateDecision
    report: FitReport
    cover_letter: str = None
    resume_summary: str = None
    improvement_suggestions: str = None
    recipient_email: str = None
    gmail_draft: dict = None  # None if Gmail step was skipped or no service provided


def run_pipeline(
    posting_text: str,
    embedding_model: EmbeddingModel,
    applicant_name: str,
    github_username: str = None,
    github_token: str = None,
    max_github_repos: int = None,
    uploaded_docs_dir: str = None,
    llm_client: LLMClient = None,
    gmail_service=None,
    resume_attachment: tuple = None,   # (filename, bytes, mime_subtype) or None
    min_coverage: float = 0.6,
    match_threshold: float = 0.1,
) -> PipelineResult:
    """Runs the complete flow: build corpus -> parse requirements -> match
    -> gate -> (if approved) draft cover letter + resume summary +
    improvement suggestions -> extract recipient email -> (if a Gmail
    service and LLM were provided) create the Gmail draft.

    llm_client and gmail_service are optional: pass None for either to
    run the match/gate step alone without needing real credentials --
    useful for a quick "am I even close" check before setting up the
    LLM/Gmail pieces.
    """
    corpus = build_full_corpus(
        uploaded_docs_dir=uploaded_docs_dir,
        github_username=github_username,
        github_token=github_token,
        max_github_repos=max_github_repos,
    )

    requirements = parse_requirements(posting_text)
    embedding_model.fit([c.text for c in corpus] + requirements)
    report = score_fit(requirements, corpus, embedding_model, threshold=match_threshold)
    decision = evaluate_gate(report, min_coverage=min_coverage)

    result = PipelineResult(decision=decision, report=report)

    if not decision.proceed_to_draft:
        return result  # gate refused -- caller shows the gap list, nothing more to do

    result.recipient_email = pick_primary_email(posting_text)

    if llm_client is None:
        return result  # matched, but no LLM provided -- caller can draft manually later

    result.cover_letter = write_cover_letter(decision, report, posting_text, llm_client, applicant_name)
    result.resume_summary = write_resume_summary(decision, report, posting_text, llm_client)
    result.improvement_suggestions = write_email_improvements(result.cover_letter, llm_client)

    if gmail_service is not None and result.recipient_email:
        attachments = [resume_attachment] if resume_attachment else None
        result.gmail_draft = create_application_draft(
            gmail_service,
            to=result.recipient_email,
            subject=f"Application from {applicant_name}",
            body_text=result.cover_letter,  # the email body IS the cover letter, nothing separate
            attachments=attachments,
        )

    return result
