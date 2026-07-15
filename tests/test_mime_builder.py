import base64
import sys
from email import message_from_bytes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_fit_studio.mime_builder import build_email_message, encode_message_for_gmail_api


def test_build_email_message_sets_to_and_subject():
    msg = build_email_message("hr@company.com", "Application for AI Engineer", "Dear Hiring Manager...")
    assert msg["to"] == "hr@company.com"
    assert msg["subject"] == "Application for AI Engineer"


def test_body_text_survives_round_trip():
    body = "Dear Hiring Manager,\n\nI am writing to apply for this role.\n\nSincerely, Applicant"
    msg = build_email_message("hr@company.com", "Subject", body)

    # Parse it back exactly as a real email client / the Gmail API would
    parsed = message_from_bytes(msg.as_bytes())
    # Walk the parts to find the plain text body
    body_found = None
    for part in parsed.walk():
        if part.get_content_type() == "text/plain":
            body_found = part.get_payload(decode=True).decode("utf-8")
            break

    assert body_found == body


def test_attachment_survives_round_trip_with_correct_filename_and_content():
    fake_docx_bytes = b"PK\x03\x04 fake docx binary content for testing"
    msg = build_email_message(
        "hr@company.com", "Application", "See attached resume.",
        attachments=[("resume.docx", fake_docx_bytes,
                     "vnd.openxmlformats-officedocument.wordprocessingml.document")],
    )

    parsed = message_from_bytes(msg.as_bytes())
    attachment_part = None
    for part in parsed.walk():
        if part.get_filename() == "resume.docx":
            attachment_part = part
            break

    assert attachment_part is not None, "Attachment not found after round-trip"
    assert attachment_part.get_payload(decode=True) == fake_docx_bytes


def test_multiple_attachments_all_survive():
    msg = build_email_message(
        "hr@company.com", "Application", "Documents attached.",
        attachments=[
            ("resume.docx", b"resume content", "vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("cover_letter.pdf", b"cover letter content", "pdf"),
        ],
    )
    parsed = message_from_bytes(msg.as_bytes())
    filenames = {part.get_filename() for part in parsed.walk() if part.get_filename()}
    assert filenames == {"resume.docx", "cover_letter.pdf"}


def test_no_attachments_produces_valid_message():
    msg = build_email_message("hr@company.com", "Subject", "Body only, no attachments.")
    parsed = message_from_bytes(msg.as_bytes())
    # should still parse without error and contain the body
    body_found = None
    for part in parsed.walk():
        if part.get_content_type() == "text/plain":
            body_found = part.get_payload(decode=True).decode("utf-8")
    assert body_found == "Body only, no attachments."


def test_encode_message_for_gmail_api_produces_correct_structure():
    msg = build_email_message("hr@company.com", "Subject", "Body")
    encoded = encode_message_for_gmail_api(msg)

    assert "message" in encoded
    assert "raw" in encoded["message"]
    assert isinstance(encoded["message"]["raw"], str)


def test_encode_message_uses_base64url_not_regular_base64():
    """The specific bug this test guards against: regular base64 uses
    '+' and '/' characters, which are NOT valid in Gmail API's expected
    base64url encoding (which uses '-' and '_' instead). Using the wrong
    encoding produces a confusing API error rather than a clear one, so
    this is worth locking in as an explicit test rather than trusting
    it by inspection."""
    msg = build_email_message("hr@company.com", "Subject", "Body " * 500)  # long enough to likely trigger + or /
    encoded = encode_message_for_gmail_api(msg)
    raw = encoded["message"]["raw"]

    # base64url alphabet should never contain '+' or '/'
    assert "+" not in raw
    assert "/" not in raw

    # and it should be decodable as valid base64url
    decoded = base64.urlsafe_b64decode(raw)
    assert b"Body" in decoded
