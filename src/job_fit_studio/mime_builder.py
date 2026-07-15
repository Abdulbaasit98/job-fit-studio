"""
Builds the actual email message (body + attachments) as a MIME structure,
completely independent of the Gmail API itself. This split matters for
testing: MIME construction is pure, deterministic, and needs zero
credentials -- fully testable here. The Gmail API call that turns this
into a real draft (gmail_draft.py) is the only part that needs your real
Google account, and it's a thin wrapper around this.
"""
import base64
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def build_email_message(to: str, subject: str, body_text: str,
                          attachments: list = None) -> MIMEMultipart:
    """attachments: list of (filename, bytes, mime_subtype) tuples, e.g.
    [("cover_letter.docx", docx_bytes, "vnd.openxmlformats-officedocument.wordprocessingml.document")]
    Returns a standard email.mime.multipart.MIMEMultipart object -- not
    yet Gmail-API-ready (see encode_message_for_gmail_api below for that
    conversion), kept separate so this function is testable with
    Python's own `email` module parser, independent of anything Gmail-specific.
    """
    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    message.attach(MIMEText(body_text, "plain"))

    for filename, file_bytes, mime_subtype in (attachments or []):
        part = MIMEApplication(file_bytes, _subtype=mime_subtype)
        part.add_header("Content-Disposition", "attachment", filename=filename)
        message.attach(part)

    return message


def encode_message_for_gmail_api(message: MIMEMultipart) -> dict:
    """Gmail API's drafts().create() expects the raw MIME message,
    base64url-encoded, wrapped in {"message": {"raw": ...}}. This is the
    exact format the API requires -- documented here explicitly since
    it's a common point of confusion (regular base64 vs base64url use
    different characters for two symbols, and using the wrong one
    produces a cryptic API error rather than a clear one)."""
    raw_bytes = message.as_bytes()
    raw_b64url = base64.urlsafe_b64encode(raw_bytes).decode("utf-8")
    return {"message": {"raw": raw_b64url}}
