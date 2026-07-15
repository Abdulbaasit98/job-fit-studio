"""
Gmail OAuth2 authentication. This is the ONE piece of this project that
genuinely cannot be built or tested in a sandbox -- it requires YOUR real
Google account, a real browser consent flow, and credentials that must
never be pasted into a chat, a command-line argument, or committed to
git (same handling discipline as your Kaggle token earlier in this
project's development).

SCOPE CHOICE: 'gmail.compose', not 'gmail.send'. This scope allows
creating/editing DRAFTS but not directly sending mail via a dedicated
send call. Important nuance, stated honestly: gmail.compose technically
ALSO permits calling drafts().send() to send an existing draft -- Google
doesn't split "create drafts" and "send drafts" into fully separate
scopes. The real enforcement that this project only ever creates drafts
is architectural, not permission-based: gmail_draft.py below simply never
implements a send call anywhere in this codebase. If you (or anyone
extending this project later) want a guarantee stronger than "the code
doesn't happen to call send," the only way to get that is to physically
review the drafts in the Gmail UI yourself before ever clicking Send --
which is the intended human-in-the-loop step this whole design assumes.
"""
import os
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/gmail.compose"]

DEFAULT_CREDENTIALS_PATH = "credentials.json"  # downloaded from Google Cloud Console, see README
DEFAULT_TOKEN_PATH = "token.json"                # created automatically after first authorization


def get_gmail_service(credentials_path: str = DEFAULT_CREDENTIALS_PATH,
                        token_path: str = DEFAULT_TOKEN_PATH):
    """Returns an authenticated Gmail API service object. On first run,
    opens a browser for you to log in and approve access (compose-only);
    after that, a token is cached locally and refreshed automatically.

    Real Google Cloud OAuth setup is required before this will work --
    see the README's "Setting up Gmail access" section. This function
    will raise a clear FileNotFoundError if credentials.json isn't
    present, rather than a confusing error deep inside the Google auth
    library.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    if not Path(credentials_path).exists():
        raise FileNotFoundError(
            f"No credentials file at {credentials_path}. You need to download this "
            "from Google Cloud Console first -- see the README's 'Setting up Gmail "
            "access' section. Never share this file or paste its contents anywhere."
        )

    creds = None
    if Path(token_path).exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)  # opens your browser for consent

        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)
