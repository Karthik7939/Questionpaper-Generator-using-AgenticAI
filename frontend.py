"""
frontend.py

Streamlit frontend for the Agentic AI Question Paper Generator.

Run with:
    streamlit run frontend.py

Requires the FastAPI backend to be running at http://localhost:8000
    python -m app.main
"""

import os
import time
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Question Paper Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean white / light theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #111827;
    }

    /* Main background */
    .stApp {
        background: #f5f7fa;
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }

    /* Cards */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 20px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    }

    /* Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Section labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4f46e5;
        margin-bottom: 0.5rem;
    }

    /* Status pills */
    .pill-success {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 50px;
        background: #d1fae5;
        border: 1px solid #6ee7b7;
        color: #065f46;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pill-error {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 50px;
        background: #fee2e2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Metric boxes */
    .metric-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4f46e5;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 2px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #c7d2fe;
        border-radius: 12px;
        background: #eef2ff;
        padding: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca, #6d28d9);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35);
    }

    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #059669, #0d9488);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        width: 100%;
    }

    /* Ensure text in main area is dark */
    .stMarkdown, .stMarkdown p, label, .stTextInput label,
    .stNumberInput label, .stSelectbox label,
    .stSlider label, .stFileUploader label {
        color: #1f2937 !important;
    }

    /* Input fields */
    .stTextInput input, .stNumberInput input {
        background: #ffffff;
        border: 1px solid #d1d5db;
        color: #111827;
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: #e5e7eb;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Backend URL
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_backend_health() -> tuple[bool, dict]:
    """Check if the FastAPI backend is running and return health metadata."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {}
    except requests.exceptions.ConnectionError:
        return False, {}


def get_max_upload_mb() -> int:
    """Read upload limit from backend health endpoint."""
    ok, health = check_backend_health()
    if ok:
        return int(health.get("max_upload_size_mb", 50))
    return 50


def preview_rag_chunks(uploaded_file) -> dict | None:
    """Call /rag/preview to ingest file and return chunk debug info."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/rag/preview",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            timeout=300,
        )
        if resp.status_code == 200:
            return resp.json()
        return {
            "success": False,
            "message": f"Preview failed (HTTP {resp.status_code})",
            "errors": [resp.text],
        }
    except requests.exceptions.ConnectionError:
        return {"success": False, "message": "Cannot connect to backend.", "errors": []}
    except Exception as exc:
        return {"success": False, "message": str(exc), "errors": [str(exc)]}


def render_rag_debug_panel(debug: dict, title: str = "RAG Debug") -> None:
    """Render RAG chunk and retrieval debug information."""
    if not debug:
        st.info("No debug data available.")
        return

    st.markdown(f"**{title}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", debug.get("total_chunks", "—"))
    col2.metric("Total Characters", f"{debug.get('total_characters', 0):,}")
    syllabus = debug.get("syllabus_retrieval", {})
    content = debug.get("content_retrieval", {})
    col3.metric("Retrieved (syllabus / content)", f"{syllabus.get('chunks_returned', 0)} / {content.get('chunks_returned', 0)}")

    if syllabus.get("timings_ms") or content.get("timings_ms"):
        st.caption(
            f"Retrieval timings — syllabus: {syllabus.get('timings_ms', {})} | "
            f"content: {content.get('timings_ms', {})}"
        )

    with st.expander("All chunks preview", expanded=False):
        for chunk in debug.get("all_chunks_preview", []):
            st.markdown(
                f"**Chunk {chunk.get('chunk_id')}** · `{chunk.get('source')}` · "
                f"{chunk.get('length', 0)} chars"
            )
            if chunk.get("first_line"):
                st.caption(chunk["first_line"])
            st.text(chunk.get("content_preview", ""))
            st.divider()

    with st.expander("Syllabus retrieval chunks", expanded=False):
        st.caption(f"Query: `{syllabus.get('query', '')}`")
        for chunk in syllabus.get("chunks_preview", []):
            st.markdown(f"**Chunk {chunk.get('chunk_id')}** — {chunk.get('first_line', '')[:80]}")
            st.text(chunk.get("content_preview", ""))
            st.divider()

    with st.expander("Content retrieval chunks", expanded=False):
        st.caption(f"Query: `{content.get('query', '')}`")
        for chunk in content.get("chunks_preview", []):
            st.markdown(f"**Chunk {chunk.get('chunk_id')}** — {chunk.get('first_line', '')[:80]}")
            st.text(chunk.get("content_preview", ""))
            st.divider()

    with st.expander("Raw debug JSON", expanded=False):
        st.json(debug)


def render_pipeline_debug(debug: dict) -> None:
    """Render agent pipeline debug summary after generation."""
    if not debug:
        return

    pipeline = debug.get("pipeline", {})
    st.markdown("**Pipeline summary**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Syllabus units", pipeline.get("syllabus_units", 0))
    c2.metric("Generated Qs", pipeline.get("generated_questions", 0))
    c3.metric("Validated Qs", pipeline.get("validated_questions", 0))
    c4.metric("Answer keys", pipeline.get("answer_key_entries", 0))

    st.caption(
        f"Status: `{pipeline.get('status')}` · "
        f"Last agent: `{pipeline.get('current_agent')}` · "
        f"File: `{debug.get('uploaded_file', '')}`"
    )

    topics = pipeline.get("syllabus_topics_preview", [])
    if topics:
        with st.expander("Extracted syllabus units", expanded=False):
            for unit in topics:
                st.markdown(f"**Unit {unit.get('unit_number')}: {unit.get('unit_name')}**")
                for topic in unit.get("topics", []):
                    st.markdown(f"- {topic}")
                if unit.get("topic_count", 0) > len(unit.get("topics", [])):
                    st.caption(f"+ {unit['topic_count'] - len(unit['topics'])} more topic(s)")
                st.divider()


def fetch_papers() -> list[str]:
    """Fetch the list of generated PDFs from the backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/papers", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("files", [])
    except Exception:
        pass
    return []


def download_pdf_bytes(filename: str) -> bytes | None:
    """Download a PDF file from the backend as bytes."""
    try:
        resp = requests.get(f"{BACKEND_URL}/download/{filename}", timeout=30)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "generation_result" not in st.session_state:
    st.session_state.generation_result = None
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "rag_preview_result" not in st.session_state:
    st.session_state.rag_preview_result = None
if "show_debug" not in st.session_state:
    st.session_state.show_debug = True
if "_run_rag_preview" not in st.session_state:
    st.session_state._run_rag_preview = False

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">📝 AI Question Paper Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Multi-Agent AI • LangGraph • Groq LLM • Bloom\'s Taxonomy</div>',
    unsafe_allow_html=True,
)

# Backend status indicator
backend_ok, health_info = check_backend_health()
max_upload_mb = int(health_info.get("max_upload_size_mb", 50)) if backend_ok else 50
if backend_ok:
    st.markdown(
        f'<span class="pill-success">● Backend Online</span> '
        f'<span style="color:#6b7280;font-size:0.85rem">· max upload {max_upload_mb} MB · '
        f'{health_info.get("model", "")}</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<span class="pill-error">● Backend Offline — start with: python -m app.main</span>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section-label">📋 Paper Configuration</div>', unsafe_allow_html=True)

    st.markdown("**Institution Details**")
    institution_name = st.text_input("Institution Name", value="University")
    course_name      = st.text_input("Course Name", value="Internet of Things")
    course_code      = st.text_input("Course Code", value="IOT501")
    semester         = st.text_input("Semester", value="V")
    exam_type        = st.selectbox(
        "Exam Type",
        ["End Semester Examination", "Internal Assessment", "Mid Semester Examination", "Unit Test"],
    )
    duration  = st.selectbox("Duration", ["3 Hours", "2 Hours", "1.5 Hours", "1 Hour"])
    exam_date = st.text_input("Exam Date (optional)", placeholder="e.g. June 2026")

    st.divider()
    st.markdown("**Question Distribution**")
    total_marks = st.number_input("Total Marks", min_value=10, max_value=200, value=100, step=5)

    col1, col2 = st.columns(2)
    with col1:
        two_mark   = st.number_input("2 Mark Qs", min_value=0, max_value=20, value=5)
        ten_mark   = st.number_input("10 Mark Qs", min_value=0, max_value=10, value=3)
    with col2:
        five_mark    = st.number_input("5 Mark Qs", min_value=0, max_value=20, value=4)
        fifteen_mark = st.number_input("15 Mark Qs", min_value=0, max_value=10, value=2)

    # Computed marks check
    computed = (two_mark * 2) + (five_mark * 5) + (ten_mark * 10) + (fifteen_mark * 15)
    if computed != total_marks:
        st.warning(f"⚠️ Question marks sum to **{computed}**, not {total_marks}.")
    else:
        st.success(f"✔ Marks add up correctly: {computed}")

    st.divider()
    st.markdown("**Difficulty Distribution**")
    easy_pct   = st.slider("Easy %",   0, 100, 30, step=5)
    medium_pct = st.slider("Medium %", 0, 100, 50, step=5)
    hard_pct   = st.slider("Hard %",   0, 100, 20, step=5)

    pct_sum = easy_pct + medium_pct + hard_pct
    if pct_sum != 100:
        st.warning(f"⚠️ Difficulty % sums to **{pct_sum}**, not 100.")
    else:
        st.success("✔ Difficulty % = 100")

    st.divider()
    st.markdown('<div class="section-label">🔧 Debug Tools</div>', unsafe_allow_html=True)
    st.session_state.show_debug = st.toggle("Show debug panel", value=st.session_state.show_debug)

    if st.session_state.show_debug:
        st.caption(
            "Preview RAG chunking without running the full agent pipeline. "
            "Useful to verify your syllabus is parsed correctly."
        )
        if st.button("🔍 Preview RAG Chunks", use_container_width=True, disabled=not backend_ok):
            st.session_state._run_rag_preview = True
            st.rerun()

# ---------------------------------------------------------------------------
# Main area — two columns
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1.2, 1], gap="large")

# ---- Left column: Upload + Generate ----
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📤 Upload Syllabus</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your syllabus PDF or TXT here",
        type=["pdf", "txt", "docx"],
        help=f"Supported formats: PDF, TXT, DOCX (max {max_upload_mb} MB)",
        label_visibility="collapsed",
    )

    file_too_large = False
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        file_size_kb = file_size_mb * 1024
        size_label = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_size_kb:.1f} KB"
        file_too_large = file_size_mb > max_upload_mb
        st.markdown(
            f"📄 **{uploaded_file.name}** &nbsp;&nbsp; `{size_label}`",
            unsafe_allow_html=True,
        )
        if file_too_large:
            st.error(
                f"File is {file_size_mb:.1f} MB — exceeds the {max_upload_mb} MB limit. "
                f"Set `MAX_UPLOAD_SIZE_MB` in `.env` and restart the backend to allow larger files."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.show_debug and uploaded_file and not file_too_large:
        if st.button("🔍 Preview RAG Chunks (debug)", use_container_width=True, disabled=not backend_ok):
            st.session_state._run_rag_preview = True
            st.rerun()

    # --- Generate button ---
    can_generate = (
        backend_ok
        and uploaded_file is not None
        and not file_too_large
        and computed == total_marks
        and pct_sum == 100
    )

    # RAG preview (triggered from sidebar button)
    if st.session_state.get("_run_rag_preview"):
        st.session_state._run_rag_preview = False
        if uploaded_file is None:
            st.session_state.rag_preview_result = {
                "success": False,
                "message": "Upload a syllabus file first.",
                "errors": [],
            }
        elif file_too_large:
            st.session_state.rag_preview_result = {
                "success": False,
                "message": f"File exceeds {max_upload_mb} MB upload limit.",
                "errors": [],
            }
        else:
            with st.spinner("🔍 Running RAG preview (chunking + retrieval)…"):
                st.session_state.rag_preview_result = preview_rag_chunks(uploaded_file)

    if st.button(
        "⚡ Generate Question Paper",
        disabled=not can_generate,
        use_container_width=True,
    ):
        st.session_state.is_generating = True
        st.session_state.generation_result = None

        with st.spinner("🤖 Running multi-agent pipeline… this takes 30–90 seconds"):
            start = time.time()
            try:
                response = requests.post(
                    f"{BACKEND_URL}/generate",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={
                        "total_marks":          str(total_marks),
                        "two_mark_questions":   str(two_mark),
                        "five_mark_questions":  str(five_mark),
                        "ten_mark_questions":   str(ten_mark),
                        "fifteen_mark_questions": str(fifteen_mark),
                        "easy_percentage":      str(easy_pct),
                        "medium_percentage":    str(medium_pct),
                        "hard_percentage":      str(hard_pct),
                        "institution_name":     institution_name,
                        "course_name":          course_name,
                        "course_code":          course_code,
                        "semester":             semester,
                        "exam_type":            exam_type,
                        "duration":             duration,
                        "exam_date":            exam_date or "",
                    },
                    timeout=None,  # No timeout limit
                )
                elapsed = time.time() - start
                if response.status_code == 200:
                    result = response.json()
                    result["_elapsed"] = elapsed
                    st.session_state.generation_result = result
                else:
                    try:
                        err_body = response.json()
                        err_detail = err_body.get("detail", err_body)
                        if isinstance(err_detail, dict):
                            err_msg = err_detail.get("message", str(err_detail))
                            err_list = err_detail.get("errors", [str(err_detail)])
                            debug = err_detail.get("debug", {})
                        elif isinstance(err_body, dict) and "detail" in err_body:
                            err_msg = f"Backend returned HTTP {response.status_code}"
                            err_list = [str(err_body["detail"])]
                            debug = err_body.get("debug", {})
                        else:
                            err_msg = f"Backend returned HTTP {response.status_code}"
                            err_list = [response.text]
                            debug = {}
                    except Exception:
                        err_msg = f"Backend returned HTTP {response.status_code}"
                        err_list = [response.text]
                        debug = {}
                    st.session_state.generation_result = {
                        "success": False,
                        "message": err_msg,
                        "errors": err_list,
                        "debug": debug,
                        "_elapsed": elapsed,
                    }
            except requests.exceptions.Timeout:
                st.session_state.generation_result = {
                    "success": False,
                    "message": "Request timed out after 5 minutes.",
                    "errors": ["The backend took too long to respond."],
                    "_elapsed": time.time() - start,
                }
            except requests.exceptions.ConnectionError:
                st.session_state.generation_result = {
                    "success": False,
                    "message": "Cannot connect to backend.",
                    "errors": ["Make sure the FastAPI server is running: python -m app.main"],
                    "_elapsed": 0,
                }

        st.session_state.is_generating = False
        st.rerun()

    if not backend_ok:
        st.info("💡 Start the backend first: `python -m app.main`")
    elif uploaded_file is None:
        st.info("💡 Upload a syllabus PDF to get started.")
    elif computed != total_marks:
        st.info("💡 Fix the marks distribution in the sidebar.")
    elif pct_sum != 100:
        st.info("💡 Difficulty percentages must sum to 100.")
    elif file_too_large:
        st.info(f"💡 Reduce file size below {max_upload_mb} MB or increase the backend limit.")

    # RAG preview results (below upload card)
    if st.session_state.show_debug and st.session_state.rag_preview_result:
        preview = st.session_state.rag_preview_result
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if preview.get("success"):
            st.success(preview.get("message", "RAG preview complete"))
            st.caption(f"File: {preview.get('file_name')} · {preview.get('file_size_mb')} MB")
            render_rag_debug_panel(preview.get("debug", {}), title="RAG Preview")
        else:
            st.error(preview.get("message", "RAG preview failed"))
            for err in preview.get("errors", []):
                st.error(err)
        st.markdown("</div>", unsafe_allow_html=True)


# ---- Right column: Results ----
with right_col:
    result = st.session_state.generation_result

    if result is None:
        # Idle state
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📊 Pipeline Steps</div>', unsafe_allow_html=True)
        steps = [
            ("🔍", "Syllabus Agent", "Extracts units & topics"),
            ("📝", "Question Generator", "Creates exam questions"),
            ("🎯", "Bloom Taxonomy Agent", "Classifies difficulty levels"),
            ("✅", "Validation Agent", "Checks quality & coverage"),
            ("🗝️", "Answer Key Agent", "Generates model answers"),
            ("📄", "PDF Generator", "Produces final PDFs"),
        ]
        for icon, name, desc in steps:
            st.markdown(
                f"**{icon} {name}** — <span style='color:rgba(255,255,255,0.5);font-size:0.85rem'>{desc}</span>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    elif result.get("success"):
        # Success state
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<span class="pill-success">✔ Generation Complete</span>', unsafe_allow_html=True)
        st.markdown(f"**{result.get('message', '')}**")
        st.markdown(f"⏱️ Completed in `{result.get('_elapsed', 0):.1f}s`")
        st.markdown("</div>", unsafe_allow_html=True)

        # Download cards
        paper_path = result.get("final_pdf_path")
        key_path   = result.get("answer_key_pdf_path")

        if paper_path:
            paper_filename = Path(paper_path).name
            paper_bytes = download_pdf_bytes(paper_filename)
            if paper_bytes:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">📄 Question Paper</div>', unsafe_allow_html=True)
                st.markdown(f"`{paper_filename}`")
                st.download_button(
                    label="⬇️ Download Question Paper PDF",
                    data=paper_bytes,
                    file_name=paper_filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        if key_path:
            key_filename = Path(key_path).name
            key_bytes = download_pdf_bytes(key_filename)
            if key_bytes:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-label">🗝️ Answer Key</div>', unsafe_allow_html=True)
                st.markdown(f"`{key_filename}`")
                st.download_button(
                    label="⬇️ Download Answer Key PDF",
                    data=key_bytes,
                    file_name=key_filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        # Warnings/errors if any
        errors = result.get("errors", [])
        if errors:
            with st.expander(f"⚠️ {len(errors)} Warning(s)", expanded=False):
                for err in errors:
                    st.markdown(f"- {err}")

        if st.session_state.show_debug:
            with st.expander("🔧 Generation debug info", expanded=False):
                st.metric("RAG chunks used", result.get("rag_chunk_count", 0))
                debug = result.get("debug", {})
                if debug.get("rag"):
                    render_rag_debug_panel(debug["rag"], title="RAG during generation")
                render_pipeline_debug(debug)
                st.markdown("**Full debug JSON**")
                st.json(debug)

    else:
        # Failure state
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<span class="pill-error">✖ Generation Failed</span>', unsafe_allow_html=True)
        st.markdown(f"**{result.get('message', 'Unknown error')}**")
        errors = result.get("errors", [])
        if errors:
            st.markdown("**Error details:**")
            for err in errors:
                st.error(err)
        if st.session_state.show_debug and result.get("debug"):
            with st.expander("🔧 Debug info from failed run", expanded=True):
                debug = result.get("debug", {})
                if debug.get("rag"):
                    render_rag_debug_panel(debug["rag"], title="RAG debug")
                render_pipeline_debug(debug)
                st.json(debug)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Previously Generated Papers section
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown('<div class="section-label">📁 Previously Generated Papers</div>', unsafe_allow_html=True)

papers = fetch_papers()
if papers:
    cols = st.columns(min(len(papers), 4))
    for i, filename in enumerate(papers[:8]):  # Show max 8
        with cols[i % 4]:
            is_answer_key = filename.startswith("answer_key_")
            icon = "🗝️" if is_answer_key else "📄"
            label = "Answer Key" if is_answer_key else "Question Paper"
            ts = filename.replace("question_paper_", "").replace("answer_key_", "").replace(".pdf", "")
            st.markdown(
                f'<div class="metric-box">'
                f'<div style="font-size:1.8rem">{icon}</div>'
                f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.6);margin-top:4px">{label}</div>'
                f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.35);margin-top:2px">{ts}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            pdf_bytes = download_pdf_bytes(filename)
            if pdf_bytes:
                st.download_button(
                    "⬇️",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"dl_{filename}",
                )
else:
    st.markdown(
        '<div style="color:rgba(255,255,255,0.35);font-size:0.9rem">No papers generated yet. Upload a syllabus and generate your first paper!</div>',
        unsafe_allow_html=True,
    )
