import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_fit_studio.cli import build_llm_client, load_resume_attachment, present_result
from job_fit_studio.embeddings import TfidfEmbedding
from job_fit_studio.gate import evaluate_gate
from job_fit_studio.matcher import score_fit
from job_fit_studio.pipeline import PipelineResult
from job_fit_studio.requirements import parse_requirements
from job_fit_studio.cv_corpus import CV_CORPUS


def test_build_llm_client_choice_3_returns_none():
    assert build_llm_client("3") is None


def test_build_llm_client_choice_2_returns_ollama():
    from job_fit_studio.llm_client import OllamaLLMClient
    client = build_llm_client("2")
    assert isinstance(client, OllamaLLMClient)


def test_build_llm_client_choice_1_without_api_key_falls_back_to_none(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert build_llm_client("1") is None


def test_load_resume_attachment_returns_none_for_empty_path():
    assert load_resume_attachment(None) is None
    assert load_resume_attachment("") is None


def test_load_resume_attachment_reads_real_docx_file(tmp_path):
    f = tmp_path / "my_resume.docx"
    f.write_bytes(b"fake docx content")

    filename, content, mime_subtype = load_resume_attachment(str(f))

    assert filename == "my_resume.docx"
    assert content == b"fake docx content"
    assert mime_subtype == "vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_load_resume_attachment_picks_correct_mime_for_pdf(tmp_path):
    f = tmp_path / "resume.pdf"
    f.write_bytes(b"fake pdf content")
    _, _, mime_subtype = load_resume_attachment(str(f))
    assert mime_subtype == "pdf"


def test_load_resume_attachment_unknown_extension_falls_back_gracefully(tmp_path):
    f = tmp_path / "resume.xyz"
    f.write_bytes(b"content")
    _, _, mime_subtype = load_resume_attachment(str(f))
    assert mime_subtype == "octet-stream"


def _build_real_result(posting, min_coverage=0.6):
    embedder = TfidfEmbedding()
    requirements = parse_requirements(posting)
    embedder.fit([c.text for c in CV_CORPUS] + requirements)
    report = score_fit(requirements, CV_CORPUS, embedder, threshold=0.1)
    decision = evaluate_gate(report, min_coverage=min_coverage)
    return PipelineResult(decision=decision, report=report)


def test_present_result_shows_gap_list_when_not_approved(capsys):
    weak_posting = "- 5+ years of Kubernetes cluster administration\n- Expert Go development\n"
    result = _build_real_result(weak_posting)

    present_result(result)

    captured = capsys.readouterr()
    assert "Not proceeding to draft" in captured.out
    assert "Kubernetes" in captured.out


def test_present_result_shows_cover_letter_and_gmail_link_when_available(capsys):
    strong_posting = "- Strong experience with Python and PyTorch\n- Experience with FastAPI and Docker\n"
    result = _build_real_result(strong_posting)
    result.cover_letter = "Dear Hiring Manager, this is my letter."
    result.resume_summary = "A short summary."
    result.improvement_suggestions = "Consider a stronger opening."
    result.recipient_email = "hr@company.com"
    result.gmail_draft = {"id": "abc123"}

    present_result(result)

    captured = capsys.readouterr()
    assert "Dear Hiring Manager, this is my letter." in captured.out
    assert "hr@company.com" in captured.out
    assert "drafts/abc123" in captured.out
    assert "nothing was sent automatically" in captured.out.lower()


def test_present_result_notes_missing_email_gracefully(capsys):
    strong_posting = "- Strong experience with Python and PyTorch\n- Experience with FastAPI and Docker\n"
    result = _build_real_result(strong_posting)
    result.cover_letter = "Letter text"
    result.recipient_email = None

    present_result(result)

    captured = capsys.readouterr()
    assert "No recipient email found" in captured.out
