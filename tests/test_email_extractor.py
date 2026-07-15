import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_fit_studio.email_extractor import extract_emails, pick_primary_email


def test_extract_emails_finds_single_email():
    text = "Please send your resume to hr@company.com for consideration."
    assert extract_emails(text) == ["hr@company.com"]


def test_extract_emails_finds_multiple_emails():
    text = "Contact recruiter@company.com or apply via jobs@company.com"
    emails = extract_emails(text)
    assert set(emails) == {"recruiter@company.com", "jobs@company.com"}


def test_extract_emails_deduplicates_case_insensitively():
    text = "Email HR@Company.com or hr@company.com with questions."
    emails = extract_emails(text)
    assert len(emails) == 1


def test_extract_emails_returns_empty_list_when_none_found():
    text = "This posting has no contact information at all."
    assert extract_emails(text) == []


def test_extract_emails_handles_korean_posting_text():
    text = "이력서 제출: careers@company.co.kr 로 보내주세요."
    emails = extract_emails(text)
    assert emails == ["careers@company.co.kr"]


def test_pick_primary_email_returns_none_when_no_email():
    assert pick_primary_email("No contact info here.") is None


def test_pick_primary_email_returns_the_only_email():
    text = "Reach out to jobs@company.com for more info."
    assert pick_primary_email(text) == "jobs@company.com"


def test_pick_primary_email_prefers_application_context_over_unrelated():
    """The realistic disambiguation case: a posting mentions a general
    support/info email AND a specific application email -- the one near
    'apply'/'resume'/'careers' context should win."""
    text = """
    Company Overview
    For general inquiries, contact info@company.com.

    To apply for this position, please send your resume and cover
    letter to careers@company.com.

    Copyright company.com 2026.
    """
    assert pick_primary_email(text) == "careers@company.com"


def test_pick_primary_email_works_with_korean_context_hints():
    text = """
    일반 문의: info@company.co.kr

    채용 담당자에게 이력서를 제출해 주세요: recruit@company.co.kr
    """
    assert pick_primary_email(text) == "recruit@company.co.kr"


def test_pick_primary_email_falls_back_to_first_when_no_context_distinguishes():
    """If neither email has application-related context nearby, we
    shouldn't crash or behave unpredictably -- some deterministic email
    should be returned."""
    text = "See a@company.com or b@company.com for details."
    result = pick_primary_email(text)
    assert result in ("a@company.com", "b@company.com")
