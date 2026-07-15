import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_fit_studio.embeddings import TfidfEmbedding
from job_fit_studio.llm_client import FakeLLMClient
from job_fit_studio.pipeline import run_pipeline

STRONG_MATCH_POSTING_WITH_EMAIL = """
- Strong experience with Python and PyTorch for deep learning
- Experience deploying models via FastAPI or Docker
- Experience with computer vision and object detection (YOLO)
- Familiarity with MLflow for experiment tracking

To apply, please send your resume to careers@company.com.
"""

STRONG_MATCH_POSTING_NO_EMAIL = """
- Strong experience with Python and PyTorch for deep learning
- Experience deploying models via FastAPI or Docker
- Experience with computer vision and object detection (YOLO)
- Familiarity with MLflow for experiment tracking
"""

WEAK_MATCH_POSTING = """
- 5+ years of production Kubernetes cluster administration
- Expert-level Go backend development
- Native iOS development with Swift and SwiftUI
"""


def make_fake_gmail_service(returned_draft=None):
    service = MagicMock()
    service.users.return_value.drafts.return_value.create.return_value.execute.return_value = (
        returned_draft or {"id": "draft123"}
    )
    return service


def test_weak_match_stops_at_gate_no_llm_no_gmail_called():
    llm = FakeLLMClient()
    gmail = make_fake_gmail_service()

    result = run_pipeline(
        posting_text=WEAK_MATCH_POSTING,
        embedding_model=TfidfEmbedding(),
        applicant_name="Test Applicant",
        llm_client=llm,
        gmail_service=gmail,
    )

    assert result.decision.proceed_to_draft is False
    assert result.cover_letter is None
    assert len(llm.calls) == 0
    gmail.users.return_value.drafts.return_value.create.assert_not_called()


def test_strong_match_without_llm_client_stops_after_matching():
    """Should be usable as a quick 'am I close?' check without needing
    an LLM or Gmail connection at all."""
    result = run_pipeline(
        posting_text=STRONG_MATCH_POSTING_WITH_EMAIL,
        embedding_model=TfidfEmbedding(),
        applicant_name="Test Applicant",
        llm_client=None,
    )

    assert result.decision.proceed_to_draft is True
    assert result.recipient_email == "careers@company.com"  # extraction still runs
    assert result.cover_letter is None  # but no LLM was given, so no draft text


def test_strong_match_with_llm_but_no_gmail_produces_letter_only():
    llm = FakeLLMClient(canned_response="Dear Hiring Manager, ...")

    result = run_pipeline(
        posting_text=STRONG_MATCH_POSTING_WITH_EMAIL,
        embedding_model=TfidfEmbedding(),
        applicant_name="Test Applicant",
        llm_client=llm,
        gmail_service=None,
    )

    assert result.cover_letter == "Dear Hiring Manager, ..."
    assert result.resume_summary is not None
    assert result.improvement_suggestions is not None
    assert result.gmail_draft is None  # no service given, so no draft created


def test_full_flow_with_llm_and_gmail_creates_draft_with_cover_letter_as_body():
    llm = FakeLLMClient(canned_response="Dear Hiring Manager, I am applying...")
    gmail = make_fake_gmail_service({"id": "draft789"})

    result = run_pipeline(
        posting_text=STRONG_MATCH_POSTING_WITH_EMAIL,
        embedding_model=TfidfEmbedding(),
        applicant_name="Test Applicant",
        llm_client=llm,
        gmail_service=gmail,
        resume_attachment=("resume.docx", b"fake resume bytes",
                            "vnd.openxmlformats-officedocument.wordprocessingml.document"),
    )

    assert result.gmail_draft == {"id": "draft789"}

    call_kwargs = gmail.users.return_value.drafts.return_value.create.call_args.kwargs
    # the draft body must be the cover letter -- not some separate generic email text
    import base64
    raw = call_kwargs["body"]["message"]["raw"]
    decoded = base64.urlsafe_b64decode(raw)
    assert b"Dear Hiring Manager, I am applying..." in decoded
    assert b"resume.docx" in decoded  # attachment filename present too


def test_matched_but_no_email_found_skips_gmail_even_with_service_available():
    """Real edge case: the gate approves, an LLM is available, a Gmail
    service is available -- but the posting simply has no email address
    in it. Should draft the letter but NOT attempt to create a Gmail
    draft with no recipient."""
    llm = FakeLLMClient(canned_response="Dear Hiring Manager, ...")
    gmail = make_fake_gmail_service()

    result = run_pipeline(
        posting_text=STRONG_MATCH_POSTING_NO_EMAIL,
        embedding_model=TfidfEmbedding(),
        applicant_name="Test Applicant",
        llm_client=llm,
        gmail_service=gmail,
    )

    assert result.recipient_email is None
    assert result.cover_letter is not None  # letter still gets written
    assert result.gmail_draft is None
    gmail.users.return_value.drafts.return_value.create.assert_not_called()
