# job-fit-studio — Complete: Ingestion + RAG Matcher + Capability Gate + Grounded Drafting + Gmail Drafts

An honest job-fit analyzer: build your capability corpus once — from your
resume, portfolio documents, and GitHub repos — then for every new job
posting, get a real, evidence-based fit score, a hard gate that refuses
to draft an application for roles you're not genuinely matched to, and
(when you ARE matched) a cover letter and resume summary — grounded only
in verified evidence — saved directly as a Gmail draft with your resume
attached, ready for you to review and send yourself.

This is **Phase 1**: ingestion (from three sources), requirement parsing,
RAG-based matching, and the capability gate. LLM-drafted resume/cover
letter and the Gmail draft-only integration with attachments are Phase
2/3 (see below).

## Design principle: the gate is the point

Most "AI resume tailoring" tools always produce a confident-sounding
cover letter, regardless of actual fit. This system refuses to draft
below a coverage threshold, and instead tells you **exactly which
requirements you don't have evidence for** — turning a bad-fit posting
into a concrete "go build this" list instead of a dishonest application.

## Three ways to build your corpus — upload once, reuse forever

```python
from job_fit_studio.corpus_builder import build_full_corpus

corpus = build_full_corpus(
    uploaded_docs_dir="my_documents",      # drop your resume, portfolio write-ups, cover letters here
    github_username="Abdulbaasit98",         # pulls every public repo + README automatically
    github_token="ghp_...",                   # optional, only needed to avoid the 60/hour rate limit
)
```

1. **Hand-authored (`cv_corpus.py`)** — your core projects, written once
   with careful `evidence` fields (test counts, metrics, links).
2. **Uploaded documents (`document_ingest.py`)** — drop your actual
   resume file (`.pdf`, `.docx`, `.txt`, `.md`), portfolio write-ups, or
   past cover letters in a folder; each gets extracted and split into
   paragraph-level chunks.
3. **GitHub (`github_ingest.py`)** — pulls every public repo + README via
   the GitHub API automatically, so the corpus stays current as you push
   new work without you touching any file by hand.

You only need to re-run `build_full_corpus()` after you've shipped new
work — not before every single job application.

## A real bug caught while building this, worth knowing about

GitHub's API returns rate-limit errors as a JSON **object** with a
`message` key — not a list of repos. Naive code doing `len(response.json())`
on that response silently "succeeds" with a nonsense small number (the
error object happens to have exactly 2 keys, so it reads as "2 repos
found" instead of failing loudly). Hit this for real during development;
`github_ingest.py`'s `_get()` explicitly checks response *type*, not just
"did the request not throw," and `tests/test_github_ingest.py` locks in
the exact real error body as a permanent regression test.

## How the matcher and gate work

1. **`requirements.py`** — splits a pasted job posting into discrete
   requirement lines (bullet/line-based, no LLM needed for this step).
2. **`matcher.py`** — embeds the full corpus and the requirements, finds
   the best-matching chunk per requirement via cosine similarity, reports
   **coverage** (fraction matched) and mean similarity.
3. **`gate.py`** — coverage >= threshold → proceed to draft; below
   threshold → return the specific unmatched requirements as a gap list.

## Real demo output

```
Coverage: 75%  |  Mean similarity: 0.169
Gate decision: DRAFT

[MATCHED] (0.327) Strong experience with Python and PyTorch for deep learning
           -> Core ML/DL Stack
[GAP    ] (0.000) Experience with Kubernetes cluster orchestration in production
[GAP    ] (0.000) Experience with LangChain or LlamaIndex for LLM application development
```
The two identified gaps independently matched the skill-gap analysis
already done by hand for this same job search — real confirmation the
matching logic finds real signal, not noise.

## Phase 2: grounded drafting — the safeguard is omission, not instruction

`letter_writer.py` writes a cover letter and resume summary, but the LLM
is the *least* important safeguard. The real protection happens before
the LLM ever runs: only chunks that matched a real requirement in the
gate get included in the prompt at all. An LLM told "don't invent things"
in a system prompt can still invent things; an LLM that was simply never
shown a piece of information cannot reference it. Two enforcement layers:

1. **Grounding by omission** — `_build_evidence_block()` includes only
   matched, deduplicated chunks. Verified directly: `tests/test_letter_writer.py`
   checks that chunks NOT matched to any requirement never appear, by
   title, anywhere in the constructed prompt.
2. **Hard refusal** — `write_cover_letter()`/`write_resume_summary()`
   raise immediately if given a `GateDecision` where `proceed_to_draft`
   is `False`, and the LLM is never even called in that case (also
   directly tested).

Real example: a posting requiring Kubernetes (which you don't have)
alongside 4 things you do have still passes the gate at 80% coverage —
but Kubernetes never appears in the evidence block, so the LLM has no
way to claim that experience, even though the application overall
proceeds.

Swappable LLM backend (`llm_client.py`), same pattern as the embeddings
interface: `AnthropicLLMClient` (cloud, default — quality matters here
since this text represents you to a real employer) or `OllamaLLMClient`
(free, local, one config change away). Both are real implementations,
but neither can be exercised end-to-end in the sandbox this was built in
(no API key, no running Ollama server here) — so both are tested via
`FakeLLMClient`, which proves the prompt-construction and grounding
logic are correct independent of which real backend eventually runs it.

## Phase 3: Gmail drafts — draft-only, by construction not by promise

`mime_builder.py` constructs the email (body + resume attachment) as a
real MIME message, tested by actually **parsing it back** with Python's
own `email` module and confirming the body text and attachment bytes
survive round-trip intact — including a specific test for the
base64-vs-base64url distinction (using the wrong one produces a
confusing Gmail API error rather than a clear one).

`gmail_draft.py` is the *only* file in this project that calls the
Gmail API, and it calls exactly one method: `drafts().create()`. There
is no `send()` anywhere in this codebase — enforced by
`tests/test_gmail_draft.py::test_no_send_capable_gmail_api_calls_exist_anywhere_in_source`,
which parses every source file's **AST** and fails the test suite if any
executable `.send(...)` call is ever added, by me or anyone else editing
this later.

**A bug caught building that exact test, worth knowing about:** the
first version used a text/regex search for `.send(`, which produced a
false positive — it matched the *docstrings explaining* that no send
call exists, since those docstrings literally contain the string
`drafts().send()` as part of describing what's absent. Parsing the AST
and only inspecting real `ast.Call` nodes fixed it, since docstrings and
comments aren't part of the executable AST at all. A good general lesson
for any "prove X never happens" test: text search finds text, not
behavior.

### Setting up Gmail access (do this on your own machine, never in a chat)

1. Go to Google Cloud Console -> create a project (or use an existing one)
2. Enable the **Gmail API** for that project (APIs & Services -> Library -> search "Gmail API" -> Enable)
3. Configure the OAuth consent screen (APIs & Services -> OAuth consent screen) -- choose "External," fill in the minimum required fields, add yourself as a test user
4. Create credentials (APIs & Services -> Credentials -> Create Credentials -> OAuth client ID -> Application type: **Desktop app**)
5. Download the resulting JSON, save it as `credentials.json` in this project's root folder
6. `.gitignore` already excludes `credentials.json` and `token.json` -- never commit either

First run of `get_gmail_service()` opens a browser for you to log in and approve **compose-only** access; after that, a `token.json` is cached locally and refreshes automatically.

### Using it end to end

```python
from job_fit_studio.gmail_auth import get_gmail_service
from job_fit_studio.gmail_draft import create_application_draft

service = get_gmail_service()  # opens browser on first run

with open("my_resume.docx", "rb") as f:
    resume_bytes = f.read()

draft = create_application_draft(
    service,
    to="hr@targetcompany.com",
    subject="Application for AI Engineer position",
    body_text=cover_letter_text,  # from write_cover_letter()
    attachments=[("resume.docx", resume_bytes,
                 "vnd.openxmlformats-officedocument.wordprocessingml.document")],
)
print(f"Draft created: https://mail.google.com/mail/u/0/#drafts/{draft['id']}")
```
Open that link, review the draft in Gmail exactly as you would any email
you wrote yourself, edit anything you want, and send it **yourself, when
you're ready** -- this project deliberately stops at the draft.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=src   # Windows: set PYTHONPATH=src
pytest tests/ -v
```
**59 tests** across ingestion (GitHub API, PDF/docx/txt extraction),
corpus merging, matching, gate branching, grounded drafting, and Gmail
MIME/draft construction (including the AST-based send-guardrail test).

## An important calibration note

`threshold=0.1` (matcher) and `min_coverage=0.6` (gate) are TF-IDF-scale
defaults — same lesson as traffic-switch's per-camera calibration, these
are **not universal constants**. Switching to `SentenceTransformerEmbedding`
will very likely need different threshold values. Recalibrate empirically
before trusting gate decisions on that backend.

## Web UI (easiest way to use this)

```bash
streamlit run app.py
```
Opens a browser tab: paste a job posting, optionally add your GitHub
username and upload resume/portfolio files, pick an LLM (or skip to just
check fit), click **Analyze fit**. Same `pipeline.py` core as the CLI —
`app.py` is purely a presentation layer, no matching/gating/drafting
logic duplicated. Verified to actually boot and serve correctly (not
just "no syntax errors") before shipping: started it headless and
confirmed a real HTTP 200 response.

## CLI (terminal-based alternative)

```bash
python -m job_fit_studio.cli
```
Same flow, prompt-by-prompt in the terminal instead of a browser.

## Project status: all three phases complete, plus a web UI

Ingestion (3 sources) -> RAG matching -> capability gate -> grounded
drafting -> Gmail draft with attachment. Every stage tested (59 tests),
including two genuinely real bugs found and fixed during development
(GitHub's rate-limit error misread as "2 repos found"; a safety
guardrail test producing a false positive from its own docstrings) --
both documented above rather than quietly fixed and forgotten.

## Known limitations (state honestly)

- `parse_requirements()` is line/bullet-heuristic based — a posting
  written as flowing paragraphs will parse poorly. An LLM-based parser
  would handle this better but adds cost/complexity Phase 1 avoids.
- The TF-IDF backend (used here — no huggingface.co access in the
  sandbox this was built in) can't detect a genuine skill described with
  completely different vocabulary than your CV uses.
- GitHub ingestion needs a token for repeated testing (60 unauthenticated
  requests/hour is easy to exhaust); a personal-use token is fine and
  free (`Settings > Developer settings > Personal access tokens`).
- `reportlab` is a test-only dependency (used to generate a real PDF
  fixture for testing PDF extraction) — not needed at runtime.

