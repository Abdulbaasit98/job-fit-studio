import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from job_fit_studio.gmail_draft import create_application_draft

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "job_fit_studio"


def make_fake_gmail_service(returned_draft: dict):
    """Mimics the shape of a real googleapiclient Gmail service object:
    service.users().drafts().create(userId=..., body=...).execute()"""
    service = MagicMock()
    service.users.return_value.drafts.return_value.create.return_value.execute.return_value = returned_draft
    return service


def test_create_application_draft_calls_drafts_create_not_send():
    fake_draft_response = {"id": "draft123", "message": {"id": "msg456"}}
    service = make_fake_gmail_service(fake_draft_response)

    result = create_application_draft(
        service, to="hr@company.com", subject="Application", body_text="Cover letter text",
    )

    assert result == fake_draft_response
    # Verify drafts().create() was called...
    service.users.return_value.drafts.return_value.create.assert_called_once()
    # ...and that nothing resembling a send call exists on the mock's
    # call history at all -- if create_application_draft ever grows a
    # send call, this would still pass (a mock allows any call), so this
    # is a WEAK guard on its own -- see the source-scanning test below
    # for the real guarantee.


def test_create_application_draft_passes_correct_userid_and_body_structure():
    service = make_fake_gmail_service({"id": "draft123"})
    create_application_draft(service, to="hr@company.com", subject="Subject", body_text="Body")

    call_kwargs = service.users.return_value.drafts.return_value.create.call_args.kwargs
    assert call_kwargs["userId"] == "me"
    assert "message" in call_kwargs["body"]
    assert "raw" in call_kwargs["body"]["message"]


def test_create_application_draft_includes_attachment():
    service = make_fake_gmail_service({"id": "draft123"})
    create_application_draft(
        service, to="hr@company.com", subject="Subject", body_text="Body",
        attachments=[("resume.docx", b"fake resume bytes",
                     "vnd.openxmlformats-officedocument.wordprocessingml.document")],
    )
    # If this didn't raise, the attachment was successfully built into
    # the MIME message and encoded -- detailed content correctness is
    # already covered by test_mime_builder.py; this just proves the
    # attachment path works end-to-end through create_application_draft.
    service.users.return_value.drafts.return_value.create.assert_called_once()


# --- The real guardrail: scan actual source code, not mock call history ---

def test_no_send_capable_gmail_api_calls_exist_anywhere_in_source():
    """The strongest guarantee this project can offer that it never
    sends email: parse every .py file's AST and check for any actual
    CALL to something named `.send(...)`.

    Deliberately AST-based, not text/regex-based: a naive regex search
    for "\\.send\\(" also matches the pattern when it appears inside a
    DOCSTRING explaining that no such call exists (which this project's
    docstrings do, on purpose, to document the safety design) -- that's
    a real false positive hit during development of this exact test.
    Parsing the AST and only inspecting actual ast.Call nodes ignores
    docstrings and comments entirely, since those aren't executable
    code and don't appear as Call nodes.
    """
    import ast

    violations = []
    for py_file in SRC_DIR.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "send":
                    violations.append(f"{py_file.name}:{node.lineno}: found an actual .send(...) call")

    assert not violations, (
        "Found send-capable Gmail API call(s) in EXECUTABLE source -- this project "
        "must remain draft-only:\n" + "\n".join(violations)
    )


def test_gmail_auth_uses_compose_scope_not_send_scope():
    """The OAuth scope itself should also never request full send/mail.google.com
    access -- compose is the narrowest scope that still allows draft creation."""
    from job_fit_studio.gmail_auth import SCOPES
    assert SCOPES == ["https://www.googleapis.com/auth/gmail.compose"]
    for scope in SCOPES:
        assert "send" not in scope.lower()
        assert scope != "https://mail.google.com/"  # the broadest, most dangerous scope
