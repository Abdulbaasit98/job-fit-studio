"""
Extracts a recipient email address from pasted job posting text (many
postings include a "send your resume to..." line or a contact email in
the footer). Pure regex, no LLM needed -- email addresses have a
well-defined, reliably-matchable format.
"""
import re

# Standard email pattern -- deliberately not trying to handle every RFC
# 5322 edge case (quoted strings, comments, etc.), since job postings
# use plain, simple email addresses in practice. Overly permissive
# patterns risk false-matching on things like version numbers or IDs
# that happen to contain '@'.
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

# Words near an email that suggest it's the ACTUAL application/contact
# address, not e.g. a copyright footer or an unrelated example. Used to
# rank multiple found emails, not to filter -- we still return all of them.
CONTEXT_HINTS = ["apply", "resume", "cv", "recruiter", "hr", "hiring", "careers",
                  "지원", "이력서", "채용", "담당자"]


def extract_emails(text: str) -> list:
    """Returns every email address found in the text, deduplicated,
    in the order they first appear."""
    found = EMAIL_PATTERN.findall(text)
    seen = set()
    unique = []
    for email in found:
        normalized = email.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(email)
    return unique


def pick_primary_email(text: str) -> str:
    """Returns the single most likely application/contact email, or None
    if none were found. If multiple emails are present, prefers one
    appearing near an application-related keyword (same line or the
    line before/after) over one with no such context.
    """
    emails = extract_emails(text)
    if not emails:
        return None
    if len(emails) == 1:
        return emails[0]

    lines = text.split("\n")
    scored = []
    for email in emails:
        score = 0
        for i, line in enumerate(lines):
            if email.lower() not in line.lower():
                continue
            context_lines = lines[max(0, i - 1):i + 2]
            context_text = " ".join(context_lines).lower()
            if any(hint in context_text for hint in CONTEXT_HINTS):
                score += 1
        scored.append((score, email))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
