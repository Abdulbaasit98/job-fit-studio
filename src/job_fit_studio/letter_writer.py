"""
Writes a cover letter and a tailored resume summary, using an LLM --
but the LLM is the LEAST important safeguard here. The real grounding
enforcement happens BEFORE the LLM ever sees anything: only chunks that
actually matched a real job requirement (per the Phase 1 gate) are
included in the prompt at all. The LLM literally cannot invent an
accomplishment from an unmatched chunk or from thin air, because that
information is never given to it -- this is "grounding by omission,"
which is more reliable than "grounding by instruction" (an LLM told
"don't make things up" can still make things up; an LLM that was never
shown the information physically cannot reference it).

A second, belt-and-suspenders safeguard: write_cover_letter() and
write_resume_summary() both REFUSE to run at all if given a GateDecision
where proceed_to_draft is False. This should never happen if the calling
code respects the gate, but defending against a caller bug here is cheap
and the failure mode (drafting for a role you're not matched to) is
exactly the thing this whole project exists to prevent.
"""
from .gate import GateDecision
from .llm_client import LLMClient

COVER_LETTER_SYSTEM_PROMPT = """You are writing a cover letter on behalf of a job applicant.

STRICT RULE: You may ONLY reference the verified accomplishments listed
below in "VERIFIED EVIDENCE." Do not mention any skill, tool, project, or
experience that is not explicitly listed there, even if it seems related
or likely to be true. Do not embellish metrics or claim a stronger result
than what is stated. If the evidence provided feels insufficient to fully
address a requirement, address it honestly with what IS available rather
than inventing detail to fill the gap.

Write in a professional, direct tone. Avoid generic filler phrases
("I am a highly motivated self-starter..."). Lead with concrete evidence,
not adjectives."""

RESUME_SUMMARY_SYSTEM_PROMPT = """You are writing a short, tailored resume
summary (3-4 sentences) highlighting only the most relevant verified
experience for a specific job posting.

STRICT RULE: identical to a cover letter -- you may ONLY reference the
verified accomplishments listed in "VERIFIED EVIDENCE" below. Never
invent or infer capability beyond what's explicitly listed."""


def _build_evidence_block(decision: GateDecision, report) -> str:
    """Builds the grounding context: one entry per UNIQUE matched chunk
    (deduplicated, since multiple requirements can match the same chunk --
    we don't want the same project description repeated five times)."""
    seen_chunk_ids = set()
    lines = []
    for match in report.matched_requirements:
        chunk = match.best_chunk
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        lines.append(f"- {chunk.title}: {chunk.text}\n  Evidence: {chunk.evidence}")
    return "\n".join(lines)


def _require_gate_approved(decision: GateDecision):
    if not decision.proceed_to_draft:
        raise RuntimeError(
            "Refusing to draft: the capability gate did NOT approve this posting "
            f"(coverage={decision.coverage:.0%}). This function must never be called "
            "when proceed_to_draft is False -- that's exactly the scenario this "
            "project exists to prevent. Use the gap-recommendation path instead."
        )


def write_cover_letter(decision: GateDecision, report, job_posting_summary: str,
                        llm_client: LLMClient, applicant_name: str) -> str:
    _require_gate_approved(decision)

    evidence_block = _build_evidence_block(decision, report)
    user_prompt = (
        f"Applicant name: {applicant_name}\n\n"
        f"Job posting summary:\n{job_posting_summary}\n\n"
        f"VERIFIED EVIDENCE (the ONLY things you may reference):\n{evidence_block}\n\n"
        f"Write a cover letter (roughly 250-350 words) for this applicant, addressed "
        f"generically to 'Hiring Manager' since the company is unspecified here."
    )
    return llm_client.generate(COVER_LETTER_SYSTEM_PROMPT, user_prompt)


CRITIQUE_SYSTEM_PROMPT = """You are a critical, honest reviewer of a job
application cover letter -- NOT the person who wrote it. Point out
genuine weaknesses: generic phrasing, weak opening/closing, claims that
sound vague rather than concrete, awkward flow, or anything a real hiring
manager would find unconvincing. Be specific and actionable (e.g. "the
second paragraph makes a claim without a concrete result -- consider
adding the specific metric from the evidence" rather than "make it
stronger"). Do not simply praise the letter. If it's genuinely solid,
say so, but still look for at least 1-2 real improvement opportunities."""


def write_email_improvements(cover_letter_text: str, llm_client: LLMClient) -> str:
    """Reviews an already-drafted cover letter and returns specific,
    actionable improvement suggestions. Deliberately a SEPARATE call from
    write_cover_letter() rather than asking one call to both write and
    self-critique -- an LLM asked to critique its own just-written output
    in the same turn tends toward self-congratulation rather than genuine
    scrutiny. A fresh call, explicitly instructed to be a critical
    reviewer rather than the author, gives more honest feedback."""
    user_prompt = f"Review this cover letter and suggest specific improvements:\n\n{cover_letter_text}"
    return llm_client.generate(CRITIQUE_SYSTEM_PROMPT, user_prompt)


def write_resume_summary(decision: GateDecision, report, job_posting_summary: str,
                          llm_client: LLMClient) -> str:
    _require_gate_approved(decision)

    evidence_block = _build_evidence_block(decision, report)
    user_prompt = (
        f"Job posting summary:\n{job_posting_summary}\n\n"
        f"VERIFIED EVIDENCE (the ONLY things you may reference):\n{evidence_block}\n\n"
        f"Write a 3-4 sentence resume summary tailored to this posting."
    )
    return llm_client.generate(RESUME_SUMMARY_SYSTEM_PROMPT, user_prompt)
