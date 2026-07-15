"""
Streamlit frontend for job-fit-studio. Run with:
    streamlit run app.py

This is a thin presentation layer over the same pipeline.py used by the
CLI (cli.py) -- no matching/gating/drafting logic lives here, it all
still comes from the already-tested core modules. This file's only job
is turning that into something you interact with by clicking, not typing
input() prompts.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from job_fit_studio.embeddings import TfidfEmbedding
from job_fit_studio.llm_client import AnthropicLLMClient, OllamaLLMClient
from job_fit_studio.pipeline import run_pipeline

st.set_page_config(page_title="job-fit-studio", page_icon="\U0001F3AF", layout="wide")


def get_secret(key: str, default=None):
    """Safe access to st.secrets -- accessing st.secrets at all raises an
    error in some Streamlit versions when no secrets.toml exists (the
    normal case for local development, where secrets aren't configured).
    This wrapper means local runs work with zero secrets configuration,
    while a deployed instance can set IS_DEPLOYED / APP_PASSCODE via
    Streamlit Cloud's secrets manager."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


IS_DEPLOYED = get_secret("IS_DEPLOYED", False)
APP_PASSCODE = get_secret("APP_PASSCODE", None)

if APP_PASSCODE:
    entered = st.text_input("Passcode", type="password")
    if entered != APP_PASSCODE:
        st.info("Enter the passcode to continue.")
        st.stop()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --bg: #F3F6F4;
        --panel: #FFFFFF;
        --border: #DCE5E0;
        --ink: #16241F;
        --ink-dim: #52625B;
        --accent: #1F6F54;
        --matched-bg: #E7F3EC;
        --matched-text: #1F6F54;
        --gap-bg: #FBEAE7;
        --gap-text: #A23624;
    }

    .stApp { background: var(--bg); }
    body, p, label, .stMarkdown, div { color: var(--ink); font-family: 'Inter', sans-serif; }

    h1 {
        font-family: 'Fraunces', serif !important;
        font-weight: 600 !important;
        color: var(--ink) !important;
        letter-spacing: -0.01em;
    }
    h2, h3 {
        font-family: 'Fraunces', serif !important;
        font-weight: 600 !important;
        color: var(--ink) !important;
    }

    .eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
        color: var(--accent); font-weight: 600; margin-bottom: 2px;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--ink) !important;
    }
    div[data-testid="stMetricLabel"] { color: var(--ink-dim) !important; }

    .req-matched {
        background: var(--matched-bg); border-left: 3px solid var(--accent);
        padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
        color: var(--ink); box-shadow: 0 1px 2px rgba(22,36,31,0.04);
    }
    .req-gap {
        background: var(--gap-bg); border-left: 3px solid var(--gap-text);
        padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;
        color: var(--ink); box-shadow: 0 1px 2px rgba(22,36,31,0.04);
    }
    .chunk-title {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent); font-size: 0.82em; margin-top: 4px;
    }

    .stTextArea textarea, .stTextInput input {
        background: var(--panel) !important; color: var(--ink) !important;
        border-color: var(--border) !important; border-radius: 8px !important;
    }
    .stButton button[kind="primary"] {
        background: var(--accent) !important; border: none !important;
        font-weight: 600 !important; border-radius: 8px !important;
    }
    div[data-testid="stFileUploader"] {
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 8px; padding: 8px;
    }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Candidate evidence report</div>', unsafe_allow_html=True)
st.title("job-fit-studio")
st.caption(
    "Paste a job posting, get a real evidence-based fit score against your actual "
    "projects and GitHub \u2014 and if you're not genuinely matched, it tells you what "
    "to build instead of writing a hollow cover letter."
)

# ---------- Inputs ----------
col_left, col_right = st.columns([1, 1.3])

with col_left:
    st.subheader("Your profile")
    applicant_name = st.text_input("Your name", placeholder="Abduganiev Abdulvosit")
    github_username = st.text_input("GitHub username", placeholder="Abdulbaasit98")
    max_repos = st.slider("Max GitHub repos to pull (most recently active)", 0, 30, 10)

    uploaded_files = st.file_uploader(
        "Upload resume / portfolio documents (optional)",
        type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
    )

    resume_attachment_file = st.file_uploader(
        "Resume file to attach if a Gmail draft gets created (optional)",
        type=["pdf", "docx"],
    )

    st.subheader("Drafting")
    llm_options = ["Skip \u2014 just check fit", "Anthropic (cloud)"]
    if not IS_DEPLOYED:
        llm_options.append("Ollama (local)")
    llm_choice = st.radio("LLM for drafting the cover letter", llm_options, index=0)
    if IS_DEPLOYED:
        st.caption(
            "Ollama isn't available on the hosted version \u2014 it runs a model on "
            "your own machine. Use Anthropic (cloud), or run this app locally for "
            "the free Ollama option."
        )

    api_key_input = None
    if llm_choice == "Anthropic (cloud)" and not os.environ.get("ANTHROPIC_API_KEY"):
        api_key_input = st.text_input(
            "ANTHROPIC_API_KEY (used only for this session, never saved to disk)",
            type="password",
        )

    create_gmail_draft = False
    if llm_choice != "Skip \u2014 just check fit":
        if IS_DEPLOYED:
            st.caption(
                "Gmail drafting isn't available on the hosted version \u2014 it needs "
                "a browser login on the same machine running the app. Run this app "
                "locally to create Gmail drafts; the cover letter is still shown "
                "and downloadable here either way."
            )
        else:
            create_gmail_draft = st.checkbox(
                "Create a Gmail draft if approved (opens a browser for one-time login)"
            )

with col_right:
    st.subheader("Job posting")
    posting_text = st.text_area("Paste the full job posting here", height=380,
                                  placeholder="- Requirement one\n- Requirement two\n...")

analyze_clicked = st.button("Analyze fit", type="primary", use_container_width=True)

# ---------- Run ----------
if analyze_clicked:
    if not applicant_name.strip():
        st.error("Enter your name first.")
        st.stop()
    if not posting_text.strip():
        st.error("Paste a job posting first.")
        st.stop()

    uploaded_docs_dir = None
    if uploaded_files:
        uploaded_docs_dir = tempfile.mkdtemp()
        for f in uploaded_files:
            with open(os.path.join(uploaded_docs_dir, f.name), "wb") as out:
                out.write(f.getbuffer())

    resume_attachment = None
    if resume_attachment_file:
        filename = resume_attachment_file.name
        ext = filename.rsplit(".", 1)[-1].lower()
        mime_subtypes = {"docx": "vnd.openxmlformats-officedocument.wordprocessingml.document",
                          "pdf": "pdf"}
        resume_attachment = (filename, resume_attachment_file.getbuffer().tobytes(),
                              mime_subtypes.get(ext, "octet-stream"))

    llm_client = None
    if llm_choice == "Anthropic (cloud)":
        if api_key_input:
            os.environ["ANTHROPIC_API_KEY"] = api_key_input
        if os.environ.get("ANTHROPIC_API_KEY"):
            llm_client = AnthropicLLMClient()
        else:
            st.warning("No API key available \u2014 continuing with fit-check only, no draft.")
    elif llm_choice == "Ollama (local)":
        llm_client = OllamaLLMClient()

    gmail_service = None
    if create_gmail_draft and llm_client is not None:
        try:
            from job_fit_studio.gmail_auth import get_gmail_service
            with st.spinner("Opening browser for Gmail authorization..."):
                gmail_service = get_gmail_service()
        except FileNotFoundError as e:
            st.warning(f"Gmail draft skipped: {e}")

    with st.spinner("Building corpus and scoring fit..."):
        try:
            result = run_pipeline(
                posting_text=posting_text,
                embedding_model=TfidfEmbedding(),
                applicant_name=applicant_name,
                github_username=github_username or None,
                max_github_repos=max_repos or None,
                uploaded_docs_dir=uploaded_docs_dir,
                llm_client=llm_client,
                gmail_service=gmail_service,
                resume_attachment=resume_attachment,
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.divider()

    # ---------- Results ----------
    m1, m2, m3 = st.columns(3)
    m1.metric("Coverage", f"{result.report.coverage:.0%}")
    m2.metric("Mean similarity", f"{result.report.mean_similarity:.3f}")
    m3.metric("Decision", "DRAFT" if result.decision.proceed_to_draft else "GAPS")

    st.progress(min(result.report.coverage, 1.0))

    if not result.decision.proceed_to_draft:
        st.warning(result.decision.reason)
        st.subheader("Build these before applying")
        for req in result.decision.unmatched_requirements:
            st.markdown(f'<div class="req-gap">{req}</div>', unsafe_allow_html=True)
    else:
        st.success(result.decision.reason)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Matched requirements")
            for match in result.report.matched_requirements:
                st.markdown(
                    f'<div class="req-matched">{match.requirement}'
                    f'<div class="chunk-title">\u2192 {match.best_chunk.title}</div></div>',
                    unsafe_allow_html=True,
                )
        with col_b:
            unmatched = result.report.unmatched_requirements
            if unmatched:
                st.subheader("Not matched (didn't block the gate)")
                for m in unmatched:
                    st.markdown(f'<div class="req-gap">{m.requirement}</div>', unsafe_allow_html=True)

        if result.recipient_email:
            st.info(f"Recipient email found in posting: **{result.recipient_email}**")
        else:
            st.warning("No recipient email found in the posting \u2014 add one manually if you draft an email.")

        if result.cover_letter:
            st.subheader("Cover letter (also the email body)")
            st.text_area("Cover letter", result.cover_letter, height=250, label_visibility="collapsed")
            st.download_button("Download cover letter (.txt)", result.cover_letter,
                                file_name="cover_letter.txt")

        if result.resume_summary:
            with st.expander("Tailored resume summary"):
                st.write(result.resume_summary)

        if result.improvement_suggestions:
            with st.expander("Suggested improvements to the letter"):
                st.write(result.improvement_suggestions)

        if result.gmail_draft:
            draft_id = result.gmail_draft.get("id")
            st.success(
                f"Gmail draft created. [Open it in Gmail]"
                f"(https://mail.google.com/mail/u/0/#drafts/{draft_id}) \u2014 "
                f"review before sending, nothing was sent automatically."
            )
