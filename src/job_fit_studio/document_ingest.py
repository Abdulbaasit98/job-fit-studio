"""
Ingests arbitrary uploaded documents (resume PDF/docx, portfolio write-ups,
cover letters) into CapabilityChunks -- the "upload your resume/portfolio"
path, complementing github_ingest.py's "pull from GitHub automatically" path.

Unlike cv_corpus.py's hand-authored chunks (one clean chunk per project,
written with a specific evidence field) or the papers RAG project's
fixed-size word-count chunking (right for continuous prose with no
natural breaks), uploaded resumes/portfolios sit in between: they have
SOME structure (paragraphs, sections) but it's not as clean as
hand-authored CapabilityChunks. Paragraph-based chunking is the right
compromise -- split on blank lines, keep each paragraph as one chunk
(a resume bullet point or a portfolio project description is usually
already one coherent paragraph), rather than an arbitrary word count that
could split a bullet point in half.
"""
import re
from pathlib import Path

from .cv_corpus import CapabilityChunk


def extract_text(path: Path) -> str:
    """Extracts plain text from a .txt, .md, .pdf, or .docx file.
    Returns "" for unsupported types rather than raising, so a mixed
    folder of files doesn't halt ingestion over one bad file."""
    suffix = path.suffix.lower()

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        import docx  # python-docx
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)

    return ""


def split_into_paragraphs(text: str, min_words: int = 5) -> list:
    """Splits on blank lines (one or more) into paragraphs, filtering out
    fragments too short to be a meaningful standalone chunk (e.g. a lone
    section header like 'SKILLS' with no content on the same line)."""
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = []
    for p in raw_paragraphs:
        cleaned = " ".join(p.split())  # collapse internal whitespace/newlines
        if cleaned and len(cleaned.split()) >= min_words:
            paragraphs.append(cleaned)
    return paragraphs


def ingest_document(path: Path, kind: str = "uploaded") -> list:
    """Full pipeline for one file: extract text, split into paragraphs,
    wrap each as a CapabilityChunk tagged with the source filename."""
    text = extract_text(path)
    if not text.strip():
        return []

    paragraphs = split_into_paragraphs(text)
    chunks = []
    for i, para in enumerate(paragraphs):
        chunks.append(CapabilityChunk(
            chunk_id=f"{path.name}:{i}",
            kind=kind,
            title=f"{path.name} (paragraph {i+1})",
            text=para,
            evidence=f"Source document: {path.name}",
            tags=[],
        ))
    return chunks


def ingest_directory(directory: str, kind: str = "uploaded") -> list:
    """Ingests every supported file in a directory. Point this at a
    folder containing your resume, portfolio write-ups, past cover
    letters, etc."""
    all_chunks = []
    for path in sorted(Path(directory).iterdir()):
        if path.is_file():
            all_chunks.extend(ingest_document(path, kind=kind))
    return all_chunks
