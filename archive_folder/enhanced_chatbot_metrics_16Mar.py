"""
enhanced_chatbot_metrics.py - Windows-compatible Streamlit chatbot
WITH USER AUTHENTICATION & PER-USER CHAT HISTORY
Enhanced with hybrid Vector DB + Knowledge Graph retrieval + Metrics tracking
Redesigned UI: Carleton red/black two-pane layout with top-nav page routing
Run: streamlit run enhanced_chatbot_metrics.py
"""
import base64
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import os
import warnings

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import json
import time
import sys
import mimetypes
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# Import custom modules
from config import Config
from model_factory import ModelFactory
from knowledge_graph import CourseKnowledgeGraph
from hybrid_retriever import HybridRetriever
from metrics_tracker import MetricsTracker

# Import authentication modules
from auth_db import DatabaseManager
from auth_ui import render_login_page, render_signup_page, render_user_profile_sidebar

APP_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────
# Initialize Database Manager (BEFORE page config)
# ─────────────────────────────────────────────────────────────
if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager("unichat_users.db")

db_manager = st.session_state.db_manager

# ─────────────────────────────────────────────────────────────
# HTML Rendering Helpers
# ─────────────────────────────────────────────────────────────
def render_html_file(filename: str, height: int = 950):
    html_path = APP_DIR / filename
    if not html_path.exists():
        st.error(f"Missing HTML file: {html_path}")
        return

    html = html_path.read_text(encoding="utf-8", errors="ignore")

    # Embed Carleton logo
    logo_path = (APP_DIR / "assets" / "carleton_logo.png")
    if logo_path.exists():
        b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64}"
        html = html.replace('src="assets/carleton_logo.png"', f'src="{data_uri}"')
        html = html.replace('src="carleton-header-logo.png"', f'src="{data_uri}"')
        html = html.replace("src='assets/carleton_logo.png'", f"src='{data_uri}'")
        html = html.replace("src='carleton-header-logo.png'", f"src='{data_uri}'")

    # Embed Team images
    def embed_img(html_text: str, filename: str, placeholder: str) -> str:
        img_path = (APP_DIR / "assets" / filename)
        if not img_path.exists():
            return html_text
        ext = filename.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
        data_uri = f"data:{mime};base64,{b64}"
        return html_text.replace(placeholder, data_uri)

    html = embed_img(html, "ethan.png", "assets/ethan.png")
    html = embed_img(html, "mat.png", "assets/mat.png")
    html = embed_img(html, "sachin.png", "assets/sachin.png")
    html = embed_img(html, "san.png", "assets/san.png")

    components.html(html, height=height, scrolling=True)

def render_about_page():
    render_html_file("about.html", height=1100)

def render_howto_page():
    render_html_file("how-to-use.html", height=1100)

def render_team_page():
    render_html_file("test.html", height=1100)

def render_forum_page():
    render_html_file("forum.html", height=1100)

# ─────────────────────────────────────────────────────────────
# ASSET PATHS
# ─────────────────────────────────────────────────────────────
ASSETS_DIR    = Path(__file__).parent / "assets"
CARLETON_LOGO = ASSETS_DIR / "carleton_logo.png"
UNICHAT_LOGO  = ASSETS_DIR / "unichat_logo.png"

# ─────────────────────────────────────────────────────────────
# IMAGE HELPER - Convert images to base64 data URIs
# ─────────────────────────────────────────────────────────────
def img_to_data_uri(path: Path) -> str:
    """Convert image file to base64 data URI for HTML embedding"""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def _logo_tag(path: Path, css_class: str, alt: str) -> str:
    """Return <img> tag with base64 data URI if file exists, else a styled placeholder."""
    if path.exists():
        src = img_to_data_uri(path)
        return f'<img src="{src}" class="{css_class}" alt="{alt}">'
    return (
        f'<div class="{css_class}" style="'
        f'background:#B3112D;border-radius:8px;padding:10px 18px;'
        f'color:#fff;font-weight:700;font-size:14px;text-align:center;'
        f'width:fit-content;margin:0 auto;">{alt}</div>'
    )

# ─────────────────────────────────────────────────────────────
# NAVIGATION REGISTRY
# ─────────────────────────────────────────────────────────────
NAV_PAGES = [
    {"key": "unichat", "label": "UniChat"},
    {"key": "about",   "label": "About"},
    {"key": "how",     "label": "How to Use"},
    {"key": "team",    "label": "Team"},
    {"key": "forum",   "label": "Forum"},
]

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UniChat – Carleton University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "show_signup" not in st.session_state:
    st.session_state.show_signup = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "unichat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics_tracker" not in st.session_state:
    st.session_state.metrics_tracker = MetricsTracker(log_dir="chatbot_metrics")
if "pending_example" not in st.session_state:
    st.session_state.pending_example = None
if "selected_course" not in st.session_state:
    st.session_state.selected_course = "All Courses"
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = Config.RETRIEVAL_TOP_K
if "chat_loaded" not in st.session_state:
    st.session_state.chat_loaded = False

# ─────────────────────────────────────────────────────────────
# CHECK AUTHENTICATION - Show login/signup if not authenticated
# ─────────────────────────────────────────────────────────────
if not st.session_state.authenticated:
    if st.session_state.show_signup:
        render_signup_page(db_manager)
    else:
        render_login_page(db_manager)
    st.stop()  # Stop execution here if not authenticated

# ═══════════════════════════════════════════════════════════════
# USER IS AUTHENTICATED - LOAD MAIN APP
# ═══════════════════════════════════════════════════════════════

# Load user's chat history from database (only once per session)
if not st.session_state.chat_loaded and st.session_state.user:
    user_id = st.session_state.user['id']
    messages_data = db_manager.get_user_messages(user_id, limit=100)
    
    # Convert to LangChain message objects
    st.session_state.messages = []
    for msg in messages_data:
        if msg['role'] == 'user':
            st.session_state.messages.append(HumanMessage(msg['content']))
        elif msg['role'] == 'assistant':
            st.session_state.messages.append(AIMessage(msg['content']))
    
    st.session_state.chat_loaded = True

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
CSS = """
<style>
/* ═══════════════════════════════════════════════
   BASE / RESET
   ═══════════════════════════════════════════════ */
.stApp {
    background-color: #f4f5f7;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ═══════════════════════════════════════════════
   LEFT PANE  –  Streamlit sidebar restyled
   ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    border-right: 3px solid #B3112D;
    min-width: 240px;
    max-width: 300px;
}
[data-testid="stSidebarCollapseButton"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stText,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: #d0d0d0 !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 14px !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 20px !important;
    margin-bottom: 8px !important;
    padding-bottom: 6px;
    border-bottom: 1px solid #333;
}

/* ─── COMPACT LOGO SECTION ─── */
.sidebar-logo-section {
    max-height: 10vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px 8px 8px;
    margin-bottom: 12px;
}
.sidebar-logo-section img {
    max-width: 85%;
    max-height: 60px;
    object-fit: contain;
    margin-bottom: 6px;
}

.sidebar-divider {
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, #B3112D, transparent);
    margin: 10px 0;
}
[data-testid="stSidebar"] [data-baseweb="select"] div {
    background-color: #2a2a2a !important;
    border-color: #444 !important;
    color: #eee !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] {
    color: #B3112D !important;
}
[data-testid="stSidebar"] .stButton button {
    background-color: #B3112D !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #950f27 !important;
}
[data-testid="stSidebar"] details summary {
    color: #aaa !important;
    font-size: 12px !important;
}

/* ═══════════════════════════════════════════════
   RIGHT PANE
   ═══════════════════════════════════════════════ */
.main .block-container {
    padding-top: 0 !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
    max-width: 100% !important;
}

/* ─── Top Navigation Bar ─── */
.cu-topnav {
    display: flex;
    align-items: center;
    background-color: #1a1a1a;
    padding: 0 20px;
    height: 52px;
    gap: 6px;
    margin: -8px -24px 0 -24px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
.cu-nav-btn {
    display: inline-block;
    color: #c8c8c8;
    background: transparent;
    border: none;
    padding: 7px 18px;
    font-size: 14px;
    font-weight: 500;
    border-radius: 6px;
    white-space: nowrap;
    text-decoration: none;
}
.cu-nav-btn.active {
    background: #B3112D;
    color: #fff !important;
    font-weight: 600;
}
.cu-red-rule {
    height: 3px;
    background: #B3112D;
    margin: 0 -24px;
}

/* ─── Functional nav buttons row ─── */
.nav-buttons-row {
    display: flex;
    gap: 0;
    background: #1a1a1a;
    padding: 6px 12px;
    margin: 0 -24px;
    border-bottom: 3px solid #B3112D;
}
.nav-buttons-row .stButton {
    flex: 1;
}
.nav-buttons-row .stButton button {
    border-radius: 6px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    height: 38px !important;
    background: transparent !important;
    color: #c8c8c8 !important;
    border: none !important;
    transition: background 0.18s, color 0.18s !important;
}
.nav-buttons-row .stButton button:hover {
    background: rgba(179,17,45,0.25) !important;
    color: #fff !important;
}
.nav-buttons-row .stButton button[data-testid="baseButton-primary"] {
    background: #B3112D !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* ─── Page Title ─── */
.cu-page-title {
    font-size: 26px;
    font-weight: 700;
    color: #1a1a1a;
    margin: 20px 0 4px;
    letter-spacing: -0.3px;
}
.cu-page-subtitle {
    font-size: 13px;
    color: #888;
    margin: 0 0 18px;
}

/* ─── UniChat logo row ─── */
.cu-unichat-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 18px 0 16px;
}
.cu-unichat-header img {
    width: 64px;
    height: 64px;
    object-fit: contain;
}
.cu-unichat-header .cu-uh-text h2 {
    margin: 0;
    font-size: 22px;
    color: #1a1a1a;
}
.cu-unichat-header .cu-uh-text p {
    margin: 2px 0 0;
    font-size: 13px;
    color: #777;
}

/* ─── Example question pills ─── */
.cu-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 4px;
}
.cu-pill {
    background: #fff0f2;
    border: 1px solid #f0c0c8;
    color: #B3112D;
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 12.5px;
}

/* ─── Metrics badges ─── */
.cu-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}
.cu-badge {
    background: #f0f4ff;
    border: 1px solid #d4dff5;
    border-radius: 8px;
    padding: 8px 14px;
    min-width: 110px;
    text-align: center;
}
.cu-badge-label {
    font-size: 10.5px;
    color: #6b7a99;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.cu-badge-value {
    font-size: 17px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 2px;
}

/* ═══════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .nav-buttons-row { padding: 4px 8px; }
    .nav-buttons-row .stButton button { font-size: 12px !important; height: 34px !important; }
    .main .block-container { padding-left: 12px !important; padding-right: 12px !important; }
    .cu-red-rule { margin: 0 -12px; }
}
</style>
"""


# ═══════════════════════════════════════════════════════════════
# SYSTEM INITIALISATION (cached – runs once)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize all system components (Windows-compatible)"""
    try:
        if not st.session_state.get("config_printed", False):
            Config.print_summary()
            st.session_state.config_printed = True

        persist_dir = Path(Config.PERSIST_DIR)
        if not persist_dir.exists():
            st.error(f"❌ Vector database not found at: {persist_dir.resolve()}")
            st.info("📝 Please run: python enhanced_ingestion.py")
            st.stop()

        embeddings = ModelFactory.create_embeddings()
        vector_store = FAISS.load_local(
            str(persist_dir), embeddings, allow_dangerous_deserialization=True
        )

        kg_dir = Path(Config.KG_PERSIST_DIR)
        if not kg_dir.exists():
            st.warning("⚠️ Knowledge graph not found. Running in vector-only mode.")
            kg = None
        else:
            kg = CourseKnowledgeGraph(persist_dir=str(kg_dir))
            kg.load()

        if kg:
            retriever = HybridRetriever(vector_store, kg)
        else:
            class VectorOnlyRetriever:
                def __init__(self, vector_store):
                    self.vector_store = vector_store

                def retrieve(self, query, k=8, course_filter=None):
                    docs_with_scores = self.vector_store.similarity_search_with_score(
                        query, k=k
                    )

                    if course_filter and course_filter != "All Courses":
                        docs_with_scores = [
                            (doc, score) for doc, score in docs_with_scores
                            if doc.metadata.get("course_code") == course_filter
                        ]

                    from hybrid_retriever import RetrievalResult
                    results = []
                    for doc, score in docs_with_scores:
                        similarity = 1 / (1 + score) if score > 0 else 1.0
                        results.append(RetrievalResult(
                            content=doc.page_content,
                            source="vector",
                            score=similarity,
                            metadata=doc.metadata,
                            course_code=doc.metadata.get("course_code", "")
                        ))

                    metadata = {
                        "vector_results": len(results),
                        "kg_results": 0,
                        "total_results": len(results),
                        "fusion_strategy": "vector-only",
                        "sources_breakdown": {"vector": len(results), "kg": 0}
                    }

                    return results, metadata

                def format_results_for_llm(self, results, max_length=3000):
                    formatted_parts = []
                    current_length = 0

                    for result in results:
                        course_info = f" [Course: {result.course_code}]" if result.course_code else ""
                        part = f"📄 Vector DB{course_info} (Score: {result.score:.3f})\n{result.content}\n"

                        if current_length + len(part) > max_length and formatted_parts:
                            break

                        formatted_parts.append(part)
                        current_length += len(part)

                    return "\n---\n".join(formatted_parts)

            retriever = VectorOnlyRetriever(vector_store)

        llm = ModelFactory.create_llm()

        course_codes_file = persist_dir / "course_codes.json"
        if course_codes_file.exists():
            with open(course_codes_file, "r", encoding="utf-8") as f:
                course_codes = json.load(f)
        else:
            course_codes = []

        return {
            "vector_store": vector_store,
            "kg": kg,
            "retriever": retriever,
            "llm": llm,
            "course_codes": course_codes,
        }

    except Exception as e:
        st.error(f"❌ Failed to initialize system: {e}")
        st.info("Please run: python enhanced_ingestion.py")
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()


# Initialize system
with st.spinner("🚀 Initializing Enhanced RAG System..."):
    system = initialize_system()

vector_store      = system["vector_store"]
kg                = system["kg"]
retriever         = system["retriever"]
llm               = system["llm"]
available_courses = system["course_codes"]


# ═══════════════════════════════════════════════════════════════
# Enhanced RAG Prompt
# ═══════════════════════════════════════════════════════════════
rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert academic advisor assistant for course outlines.

Your responsibilities:
1. Answer questions accurately using information from the provided context
2. Pay special attention to COURSE CODES and ensure information is correctly attributed
3. For factual queries (prerequisites, assessments, instructors), be precise
4. For conceptual queries (topics, explanations), be comprehensive
5. If information is incomplete or uncertain, say so clearly
6. Use specific details: course codes, percentages, dates, names, etc.
7. Format lists and relationships clearly

Context from databases:
{context}

Note: The context includes information from vector database and knowledge graph (if available).
""",
        ),
        ("human", "{question}"),
    ]
)


# ═══════════════════════════════════════════════════════════════
# Generate Response with Metrics + Save to DB
# ═══════════════════════════════════════════════════════════════
def generate_enhanced_response(
    question: str, k: int = None, course_filter: Optional[str] = None
) -> tuple:
    """Generate response using hybrid retrieval with metrics tracking"""
    start_time = time.time()
    k = k or Config.RETRIEVAL_TOP_K

    tracker = st.session_state.metrics_tracker
    tracking_data = tracker.start_query()

    enhanced_question = question
    if course_filter and course_filter != "All Courses":
        enhanced_question = f"For course {course_filter}: {question}"

    retrieval_start = time.time()
    results, metadata = retriever.retrieve(
        enhanced_question, k=k, course_filter=course_filter
    )
    retrieval_time = time.time() - retrieval_start

    if not results:
        return (
            f"I couldn't find relevant information{' for course ' + course_filter if course_filter else ''} to answer your question.",
            metadata,
            retrieval_time,
            0,
            None,
        )

    context = retriever.format_results_for_llm(results, max_length=3500)
    messages = rag_prompt.format_messages(context=context, question=question)

    generation_start = time.time()
    response = llm.invoke(messages)
    generation_time = time.time() - generation_start

    answer = response.content if hasattr(response, "content") else str(response)

    courses_referenced = list(
        dict.fromkeys([r.course_code for r in results if r.course_code])
    )

    metrics = tracker.end_query(
        tracking_data=tracking_data,
        query=question,
        answer=answer,
        retrieval_metadata=metadata,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        context=context,
        config=Config,
        course_filter=course_filter,
        courses_referenced=courses_referenced,
    )

    source_breakdown = metadata["sources_breakdown"]
    sources_text = f"\n\n📊 **Information Sources:**\n"
    sources_text += f"- Vector DB: {source_breakdown['vector']} chunks\n"

    if kg:
        sources_text += f"- Knowledge Graph: {source_breakdown['kg']} entities\n"

    sources_text += f"- Fusion Strategy: {metadata['fusion_strategy']}"

    if courses_referenced:
        sources_text += (
            f"\n\n📖 **Courses Referenced:** {', '.join(courses_referenced)}"
        )

    full_response = answer + sources_text

    return full_response, metadata, retrieval_time, generation_time, metrics


# ═══════════════════════════════════════════════════════════════
# PAGE RENDERERS
# ═══════════════════════════════════════════════════════════════

def render_unichat():
    """UniChat chatbot page – full chat interface"""

    # UniChat logo header
    if UNICHAT_LOGO.exists():
        unichat_src = img_to_data_uri(UNICHAT_LOGO)
        st.markdown(
            '<div class="cu-unichat-header">'
            f'<img src="{unichat_src}" alt="UniChat">'
            '<div class="cu-uh-text">'
            '<h2>UniChat</h2>'
            '<p>AI Chatbot for Carleton Students</p>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### 🤖 UniChat\n*AI Chatbot for Carleton Students*")

    # Controls row
    ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 1.2])
    with ctrl1:
        course_options = ["All Courses"] + sorted(available_courses)
        cur = st.session_state.selected_course
        idx = course_options.index(cur) if cur in course_options else 0
        st.session_state.selected_course = st.selectbox(
            "Course Filter", course_options, index=idx,
            help="Filter responses to a specific course",
        )
    with ctrl2:
        st.session_state.retrieval_k = st.slider(
            "Results to Retrieve",
            min_value=3, max_value=15,
            value=st.session_state.retrieval_k,
            help="Number of chunks retrieved per query",
        )
    with ctrl3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            # Clear from database
            user_id = st.session_state.user['id']
            db_manager.clear_user_messages(user_id)
            # Clear from session
            st.session_state.messages = []
            st.rerun()

    # Chat history
    for message in st.session_state.messages[-10:]:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # Example question pills
    examples = [
        "What are the prerequisites for this course?",
        "Who teaches this course?",
        "What is the grading scheme?",
        "What are the learning outcomes?",
        "Explain the topics covered",
        "What textbook is recommended?",
    ]
    pills_html = '<div class="cu-pills">'
    for q in examples:
        pills_html += f'<span class="cu-pill">{q}</span>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    ex_cols = st.columns(len(examples))
    for i, q in enumerate(examples):
        with ex_cols[i]:
            if st.button(q, key=f"ex_{i}", use_container_width=True, type="secondary"):
                st.session_state.pending_example = q
                st.rerun()

    # Chat input
    prefill = st.session_state.pending_example or ""
    if st.session_state.pending_example:
        st.session_state.pending_example = None

    user_question = st.chat_input("Ask about course outlines...")

    if not user_question and prefill:
        user_question = prefill

    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))
        
        # Save to database
        user_id = st.session_state.user['id']
        db_manager.save_message(
            user_id, 
            "user", 
            user_question, 
            st.session_state.selected_course if st.session_state.selected_course != "All Courses" else None
        )

        # Generate and display response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            with st.spinner("🔍 Searching databases..."):
                try:
                    cf = (st.session_state.selected_course
                          if st.session_state.selected_course != "All Courses" else None)
                    response, metadata, ret_time, gen_time, metrics = (
                        generate_enhanced_response(
                            user_question,
                            k=st.session_state.retrieval_k,
                            course_filter=cf,
                        )
                    )

                    message_placeholder.markdown(response)
                    st.session_state.messages.append(AIMessage(response))
                    
                    # Save assistant response to database
                    db_manager.save_message(user_id, "assistant", response, cf)

                    # Metrics badges
                    if metrics:
                        st.markdown(
                            f'<div class="cu-badges">'
                            f'<div class="cu-badge"><div class="cu-badge-label">Total Time</div>'
                            f'<div class="cu-badge-value">{metrics.total_time:.2f}s</div></div>'
                            f'<div class="cu-badge"><div class="cu-badge-label">Retrieval</div>'
                            f'<div class="cu-badge-value">{metrics.retrieval_time:.2f}s</div></div>'
                            f'<div class="cu-badge"><div class="cu-badge-label">Generation</div>'
                            f'<div class="cu-badge-value">{metrics.generation_time:.2f}s</div></div>'
                            f'<div class="cu-badge"><div class="cu-badge-label">Chunks</div>'
                            f'<div class="cu-badge-value">{metrics.num_total_results}</div></div>'
                            f'<div class="cu-badge"><div class="cu-badge-label">Memory</div>'
                            f'<div class="cu-badge-value">{metrics.memory_used_mb:.1f} MB</div></div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Check configuration and ensure all components are initialized.")
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())


# PAGE RENDERER REGISTRY
PAGE_RENDERERS = {
    "unichat": render_unichat,
    "about":   render_about_page,
    "how":     render_howto_page,
    "team":    render_team_page,
    "forum":   render_forum_page,
}


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT ASSEMBLY
# ═══════════════════════════════════════════════════════════════

# 1. Inject global CSS
st.markdown(CSS, unsafe_allow_html=True)

# 2. LEFT PANE – Streamlit sidebar
with st.sidebar:
    # COMPACT LOGO SECTION
    logo_html = _logo_tag(CARLETON_LOGO, "sidebar-logo-img", "Carleton University")
    st.markdown(
        f'<div class="sidebar-logo-section">{logo_html}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        "**Carleton University**\n\n"
        "AI-powered academic assistant helping students navigate course outlines, "
        "prerequisites, and more.",
        unsafe_allow_html=False,
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # User Profile Section
    render_user_profile_sidebar(db_manager)

    # Sidebar settings
    st.markdown("### ⚙️ Configuration")
    with st.expander("🤖 Model Settings", expanded=False):
        st.text(f"Embeddings: {Config.EMBEDDING_PROVIDER}")
        st.text(f"Model: {Config.EMBEDDING_MODEL}")
        st.text(f"LLM: {Config.LLM_PROVIDER}")
        st.text(f"Model: {Config.LLM_MODEL}")

    # Fusion strategy
    if kg:
        st.markdown("### 🔀 Fusion Strategy")
        fusion_strategy = st.selectbox(
            "Strategy:",
            ["adaptive", "weighted", "rrf"],
            index=["adaptive", "weighted", "rrf"].index(Config.FUSION_STRATEGY),
            help="How to combine vector and KG results",
        )
        Config.FUSION_STRATEGY = fusion_strategy

        if fusion_strategy == "weighted":
            st.markdown("**Retrieval Weights**")
            vector_weight = st.slider(
                "Vector DB Weight:", 0.0, 1.0, Config.VECTOR_WEIGHT, 0.1
            )
            kg_weight = 1.0 - vector_weight
            st.text(f"KG Weight: {kg_weight:.1f}")
            Config.VECTOR_WEIGHT = vector_weight
            Config.KG_WEIGHT = kg_weight

    # Session metrics
    st.markdown("### 📈 Session Metrics")
    tracker = st.session_state.metrics_tracker
    if tracker.metrics_buffer:
        summary = tracker.get_summary()
        st.metric("Total Queries", summary["total_queries"])
        st.metric("Avg Response Time", f"{summary['timing']['avg_total_time']:.2f}s")

    st.markdown("---")
    if st.button("📊 Save Metrics"):
        tracker.save_summary(silent=False)
        st.success("Metrics saved!")


# 3. RIGHT PANE – Nav bar + page content

# Visual nav bar
nav_html = '<div class="cu-topnav">'
for page in NAV_PAGES:
    active = "active" if st.session_state.active_page == page["key"] else ""
    nav_html += f'<span class="cu-nav-btn {active}">{page["label"]}</span>'
nav_html += '</div>'
st.markdown(nav_html, unsafe_allow_html=True)

# Functional nav buttons
st.markdown('<div class="nav-buttons-row">', unsafe_allow_html=True)
nav_cols = st.columns(len(NAV_PAGES))
for i, page in enumerate(NAV_PAGES):
    with nav_cols[i]:
        is_active = st.session_state.active_page == page["key"]
        if st.button(
            page["label"],
            key=f"nav_{page['key']}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = page["key"]
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# 4. Render current page
current_page = st.session_state.active_page
if current_page in PAGE_RENDERERS:
    PAGE_RENDERERS[current_page]()

# 5. Footer
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption(f"🔧 Fusion: {Config.FUSION_STRATEGY}")
with col_f2:
    st.caption(f"📦 Vector Weight: {Config.VECTOR_WEIGHT:.1f}")
with col_f3:
    if kg:
        st.caption(f"🕸️ KG Weight: {Config.KG_WEIGHT:.1f}")
    else:
        st.caption("🕸️ KG: Not available")
