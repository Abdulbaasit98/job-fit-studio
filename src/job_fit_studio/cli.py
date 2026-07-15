"""
The actual interactive tool: run `python -m job_fit_studio.cli` and it
walks you through the whole flow. Structured as three separable pieces
(gather_inputs, then run_pipeline from pipeline.py, then present_result)
specifically so the parts that CAN be tested without a real terminal are
tested that way -- only the raw input()/print() calls themselves are
untestable, everything they feed into is.
"""
import getpass
import os
import sys

from .embeddings import TfidfEmbedding
from .llm_client import AnthropicLLMClient, OllamaLLMClient
from .pipeline import PipelineResult, run_pipeline


def read_multiline(prompt: str) -> str:
    print(prompt)
    print("(paste the text, then type END on its own line and press Enter)")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def gather_inputs() -> dict:
    """All the raw input() calls, in one place. Returns a plain dict
    rather than immediately constructing pipeline objects, so this
    function's OUTPUT can be handed to run_pipeline() directly in tests
    without needing to mock input() at all -- only THIS function needs
    that, and it's isolated here specifically so nothing else does.
    """
    print("=== job-fit-studio ===\n")

    applicant_name = input("Your name (for the cover letter): ").strip()

    github_username = input("GitHub username (leave blank to skip): ").strip() or None

    resume_dir = input(
        "Folder containing your resume/portfolio documents to ingest "
        "(leave blank to skip): "
    ).strip() or None

    resume_attachment_path = input(
        "Path to a specific resume FILE to attach to the email, e.g. resume.docx "
        "(leave blank to skip attachment): "
    ).strip() or None

    posting_text = read_multiline("\nPaste the job posting:")

    llm_choice = input(
        "\nLLM for drafting -- (1) Anthropic cloud  (2) Ollama local  (3) skip, just check fit: "
    ).strip()

    gmail_choice = ""
    if llm_choice in ("1", "2"):
        gmail_choice = input("Create a Gmail draft if you're a match? (y/n): ").strip().lower()

    return {
        "applicant_name": applicant_name,
        "github_username": github_username,
        "uploaded_docs_dir": resume_dir,
        "resume_attachment_path": resume_attachment_path,
        "posting_text": posting_text,
        "llm_choice": llm_choice,
        "create_gmail_draft": gmail_choice == "y",
    }


def build_llm_client(llm_choice: str):
    if llm_choice == "1":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set in your environment -- set it before "
                  "choosing this option. Falling back to fit-check only.")
            return None
        return AnthropicLLMClient()
    if llm_choice == "2":
        return OllamaLLMClient()
    return None


def load_resume_attachment(path: str):
    if not path:
        return None
    with open(path, "rb") as f:
        file_bytes = f.read()
    filename = os.path.basename(path)
    ext = filename.rsplit(".", 1)[-1].lower()
    mime_subtypes = {
        "docx": "vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf": "pdf",
        "txt": "plain",
    }
    return (filename, file_bytes, mime_subtypes.get(ext, "octet-stream"))


def present_result(result: PipelineResult):
    print("\n" + "=" * 60)
    print(f"Coverage: {result.report.coverage:.0%}  |  "
          f"Mean similarity: {result.report.mean_similarity:.3f}")
    print(result.decision.reason)
    print("=" * 60)

    if not result.decision.proceed_to_draft:
        print("\nNot proceeding to draft. Here's what to build to close the gap:\n")
        for req in result.decision.unmatched_requirements:
            print(f"  - {req}")
        return

    print("\n--- Matched requirements ---")
    for m in result.report.matched_requirements:
        print(f"  [OK] {m.requirement}\n       -> {m.best_chunk.title}")

    if result.recipient_email:
        print(f"\nRecipient email found in posting: {result.recipient_email}")
    else:
        print("\nNo recipient email found in the posting -- you'll need to add one manually.")

    if result.cover_letter:
        print("\n--- Cover letter (this is also the email body) ---")
        print(result.cover_letter)

    if result.resume_summary:
        print("\n--- Tailored resume summary ---")
        print(result.resume_summary)

    if result.improvement_suggestions:
        print("\n--- Suggested improvements to the cover letter ---")
        print(result.improvement_suggestions)

    if result.gmail_draft:
        draft_id = result.gmail_draft.get("id")
        print(f"\nGmail draft created: https://mail.google.com/mail/u/0/#drafts/{draft_id}")
        print("Review it in Gmail before sending -- nothing was sent automatically.")


def main():
    inputs = gather_inputs()

    llm_client = build_llm_client(inputs["llm_choice"])
    resume_attachment = load_resume_attachment(inputs["resume_attachment_path"])

    gmail_service = None
    if inputs["create_gmail_draft"] and llm_client is not None:
        from .gmail_auth import get_gmail_service
        gmail_service = get_gmail_service()  # opens browser for OAuth on first run

    result = run_pipeline(
        posting_text=inputs["posting_text"],
        embedding_model=TfidfEmbedding(),
        applicant_name=inputs["applicant_name"],
        github_username=inputs["github_username"],
        uploaded_docs_dir=inputs["uploaded_docs_dir"],
        llm_client=llm_client,
        gmail_service=gmail_service,
        resume_attachment=resume_attachment,
    )

    present_result(result)


if __name__ == "__main__":
    main()
