"""
Turns a pasted job posting into a list of discrete requirement strings.

Deliberately simple (no LLM call needed for this step): job postings are
already naturally line/bullet-structured, so splitting on lines and
filtering out obvious non-requirement noise (empty lines, very short
fragments, generic boilerplate) gets most of the way there without any
external dependency or cost. This keeps the matcher's core logic testable
without needing an LLM API key -- the LLM only enters later, for the
actual resume/cover-letter WRITING step, not for parsing.
"""
import re

# Lines that are almost never real requirements -- company boilerplate
# that would otherwise pollute the requirement list and get "matched"
# against your CV nonsensically.
NOISE_PATTERNS = [
    r"^(equal opportunity|우대사항|자격요건|담당업무|근무|복지|모집)\s*[:：]?\s*$",
    r"^\s*[-*•]?\s*$",
]


def parse_requirements(posting_text: str, min_words: int = 3) -> list:
    """Splits posting text into candidate requirement lines.

    min_words: filters out fragments too short to be a real requirement
    (e.g. a lone section header like "Requirements" or "우대사항").
    """
    lines = posting_text.strip().split("\n")
    requirements = []

    for line in lines:
        # Strip common bullet markers (-, *, •, digit+period) and whitespace
        cleaned = re.sub(r"^[\s\-\*•\u2022]+", "", line).strip()
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)

        if not cleaned:
            continue
        if len(cleaned.split()) < min_words:
            continue
        if any(re.match(pat, cleaned, re.IGNORECASE) for pat in NOISE_PATTERNS):
            continue

        requirements.append(cleaned)

    return requirements
