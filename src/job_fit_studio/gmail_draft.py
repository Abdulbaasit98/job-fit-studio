"""
The only Gmail API interaction in this entire project: creating a draft.
There is deliberately no send_email(), no drafts().send(), no equivalent
of either, anywhere in this file or this project. See gmail_auth.py's
docstring for the honest nuance about OAuth scope vs. code-level
enforcement -- this file is the code-level half of that guarantee.
"""
from .mime_builder import build_email_message, encode_message_for_gmail_api


def create_application_draft(gmail_service, to: str, subject: str, body_text: str,
                               attachments: list = None) -> dict:
    """Creates a Gmail DRAFT (visible in your Drafts folder, NOT sent).
    gmail_service: an authenticated service object from gmail_auth.get_gmail_service().
    attachments: list of (filename, bytes, mime_subtype) tuples.

    Returns the Gmail API's draft resource dict (includes the draft id,
    useful if you want to open it directly: Gmail's web UI URL pattern
    is https://mail.google.com/mail/u/0/#drafts/{draft_id}).
    """
    message = build_email_message(to, subject, body_text, attachments)
    encoded = encode_message_for_gmail_api(message)

    draft = gmail_service.users().drafts().create(userId="me", body=encoded).execute()
    return draft
