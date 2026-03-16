"""
enhanced_chatbot_metrics.py - Windows-compatible Streamlit chatbot
Enhanced with hybrid Vector DB + Knowledge Graph retrieval + Metrics tracking
Redesigned UI: Carleton red/black theme, persistent left sidebar, top-nav right pane
Run: streamlit run enhanced_chatbot_metrics.py
"""
import os
import warnings
import base64
import mimetypes
from pathlib import Path

def img_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import json
import time
import sys
from pathlib import Path
import streamlit as st
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

# ─────────────────────────────────────────────
# ASSET PATHS  (place logos in an "assets" folder)
# ─────────────────────────────────────────────
ASSETS_DIR = Path(__file__).parent / "assets"
CARLETON_LOGO = ASSETS_DIR / "carleton_logo.png"
UNICHAT_LOGO  = ASSETS_DIR / "unichat_logo.png"

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="UniChat – Carleton University",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",   # we build our own sidebar via HTML
)

# ─────────────────────────────────────────────
# GLOBAL CSS  – injected once, drives the whole layout
# ─────────────────────────────────────────────
GLOBAL_CSS = """
<style>
/* ── Reset Streamlit chrome ── */
.stApp {
    background: #f5f5f5;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
/* Hide default Streamlit sidebar toggle & hamburger */
.stSidebar, [data-testid="baseSidebarContent"] { display: none !important; }
section[data-testid="main"] { padding: 0 !important; margin: 0 !important; }
div[data-testid="stDefaultLayout"] > div { padding: 0 !important; }

/* ── Two-pane wrapper ── */
#cu-layout {
    display: flex;
    min-height: 100vh;
    width: 100%;
}

/* ── LEFT PANE (20 %) ── */
#cu-left {
    width: 20%;
    min-width: 220px;
    max-width: 320px;
    background: #1a1a1a;                  /* deep black */
    color: #ffffff;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 32px 16px 24px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
    box-shadow: 3px 0 12px rgba(0,0,0,.25);
    z-index: 100;
}
#cu-left img.cu-logo {
    width: 80%;
    max-width: 200px;
    margin-bottom: 24px;
    filter: brightness(0) invert(1);       /* make dark logo white */
}
#cu-left .cu-divider {
    width: 60%;
    height: 1px;
    background: #B3112D;                   /* Carleton red accent */
    margin: 20px 0;
}
#cu-left .cu-desc {
    font-size: 13px;
    line-height: 1.55;
    color: #c0c0c0;
    text-align: center;
    padding: 0 8px;
}
#cu-left .cu-desc strong {
    color: #B3112D;
}

/* ── RIGHT PANE (80 %) ── */
#cu-right {
    width: 80%;
    flex: 1;
    display: flex;
    flex-direction: column;
    background: #ffffff;
    min-height: 100vh;
}

/* ── TOP NAV BAR ── */
#cu-topnav {
    display: flex;
    align-items: center;
    background: #1a1a1a;
    padding: 0 28px;
    height: 56px;
    gap: 4px;
    position: sticky;
    top: 0;
    z-index: 99;
    box-shadow: 0 2px 8px rgba(0,0,0,.2);
}
.cu-nav-btn {
    color: #e0e0e0;
    background: transparent;
    border: none;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: 500;
    border-radius: 6px;
    cursor: pointer;
    transition: background .18s, color .18s;
    white-space: nowrap;
    letter-spacing: 0.3px;
}
.cu-nav-btn:hover {
    background: rgba(179,17,45,.25);
    color: #ffffff;
}
.cu-nav-btn.active {
    background: #B3112D;
    color: #ffffff;
    font-weight: 600;
}

/* ── RED RULE under nav ── */
#cu-red-rule {
    height: 3px;
    background: #B3112D;
}

/* ── CONTENT AREA ── */
#cu-content {
    flex: 1;
    padding: 28px 36px 24px;
    overflow-y: auto;
}

/* ── Page heading ── */
.cu-page-title {
    font-size: 28px;
    font-weight: 700;
    color: #1a1a1a;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.cu-page-subtitle {
    font-size: 13px;
    color: #888;
    margin: 0 0 20px;
}

/* ── UniChat logo in right pane ── */
#cu-unichat-logo-row {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
}
#cu-unichat-logo-row img {
    width: 72px;
    height: 72px;
    object-fit: contain;
}

/* ── Chat area ── */
.cu-chat-wrap {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 10px;
    padding: 20px 22px;
    min-height: 340px;
    max-height: 540px;
    overflow-y: auto;
    margin-bottom: 14px;
}

/* Streamlit chat messages – keep default but tint */
.stChatMessage {
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}

/* ── Example questions pills ── */
.cu-examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
    margin-bottom: 18px;
}
.cu-example-pill {
    background: #fff0f2;
    border: 1px solid #f0c0c8;
    color: #B3112D;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12.5px;
    cursor: pointer;
    transition: background .15s;
    user-select: none;
}
.cu-example-pill:hover {
    background: #ffe0e4;
}

/* ── About / FAQ section styles ── */
.cu-section-card {
    background: #f9f9f9;
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 18px;
}
.cu-section-card h3 {
    color: #B3112D;
    margin-top: 0;
    font-size: 17px;
}
.cu-section-card p, .cu-section-card li {
    color: #444;
    font-size: 14px;
    line-height: 1.6;
}
.cu-section-card ul { padding-left: 20px; }

/* ── FAQ accordion ── */
.cu-faq-q {
    background: #1a1a1a;
    color: #fff;
    padding: 12px 18px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
}
.cu-faq-a {
    background: #fff;
    border: 1px solid #ddd;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 14px 18px;
    color: #555;
    font-size: 13.5px;
    line-height: 1.55;
}

/* ── Metrics badges ── */
.cu-metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 12px;
}
.cu-metric-badge {
    background: #f0f4ff;
    border: 1px solid #d0d8f0;
    border-radius: 8px;
    padding: 10px 16px;
    min-width: 120px;
    text-align: center;
}
.cu-metric-badge .cu-m-label {
    font-size: 11px;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.cu-metric-badge .cu-m-value {
    font-size: 18px;
    font-weight: 700;
    color: #1a1a1a;
    margin-top: 2px;
}

/* ── Responsive ── */
@media (max-width: 900px) {
    #cu-layout { flex-direction: column; }
    #cu-left {
        width: 100%; max-width: 100%;
        flex-direction: row;
        height: auto; position: relative;
        padding: 16px 20px;
        gap: 16px;
    }
    #cu-left img.cu-logo { width: 52px; max-width: 52px; margin-bottom: 0; }
    #cu-left .cu-divider { display: none; }
    #cu-left .cu-desc { text-align: left; font-size: 12px; }
    #cu-right { width: 100%; }
    #cu-topnav { padding: 0 12px; overflow-x: auto; }
    .cu-nav-btn { padding: 7px 12px; font-size: 13px; }
    #cu-content { padding: 20px 18px; }
}
@media (max-width: 580px) {
    .cu-nav-btn { padding: 6px 9px; font-size: 12px; }
    #cu-content { padding: 16px 12px; }
    .cu-page-title { font-size: 22px; }
}
</style>
"""


# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
if "active_page" not in st.session_state:
    st.session_state.active_page = "unichat"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics_tracker" not in st.session_state:
    st.session_state.metrics_tracker = MetricsTracker(log_dir="chatbot_metrics")
if "pending_example" not in st.session_state:
    st.session_state.pending_example = None


# ─────────────────────────────────────────────
# NAVIGATION DEFINITION  – add new pages here
# ─────────────────────────────────────────────
NAV_PAGES = [
    {"key": "unichat",  "label": "UniChat"},
    {"key": "about",    "label": "About Us"},
    {"key": "faq",      "label": "FAQ"},
    # ➜ To add a new page: append {"key": "mypage", "label": "My Page"}
    #   then implement render_mypage() below.
]


# ─────────────────────────────────────────────
# SYSTEM INITIALISATION (cached)
# ─────────────────────────────────────────────
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


# Boot the system (cached – runs once)
with st.spinner("🚀 Initializing Enhanced RAG System…"):
    system = initialize_system()

vector_store      = system["vector_store"]
kg                = system["kg"]
retriever         = system["retriever"]
llm               = system["llm"]
available_courses = system["course_codes"]


# ─────────────────────────────────────────────
# RAG PROMPT
# ─────────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert academic advisor assistant for course outlines.

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
"""),
    ("human", "{question}"),
])


# ─────────────────────────────────────────────
# RESPONSE GENERATION  (unchanged logic)
# ─────────────────────────────────────────────
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
            f"I couldn't find relevant information"
            f"{' for course ' + course_filter if course_filter else ''} "
            f"to answer your question.",
            metadata, retrieval_time, 0, None,
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
    sources_text  = "\n\n📊 **Information Sources:**\n"
    sources_text += f"- Vector DB: {source_breakdown['vector']} chunks\n"
    if kg:
        sources_text += f"- Knowledge Graph: {source_breakdown['kg']} entities\n"
    sources_text += f"- Fusion Strategy: {metadata['fusion_strategy']}"
    if courses_referenced:
        sources_text += f"\n\n📖 **Courses Referenced:** {', '.join(courses_referenced)}"

    return answer + sources_text, metadata, retrieval_time, generation_time, metrics


# ─────────────────────────────────────────────
# HELPER – render an image or a coloured placeholder
# ─────────────────────────────────────────────
def _logo_tag(path: Path, css_class: str, alt: str) -> str:
    if path.exists():
        src = img_to_data_uri(path)
        return f'<img src="{src}" class="{css_class}" alt="{alt}">'
    return (
        f'<div class="{css_class}" style="'
        f'background:#B3112D;border-radius:8px;padding:10px 18px;'
        f'color:#fff;font-weight:700;font-size:14px;text-align:center;'
        f'width:fit-content;margin:0 auto;">{alt}</div>'
    )



# ─────────────────────────────────────────────
# PAGE RENDERERS
# ─────────────────────────────────────────────

# ── Session-persisted course filter & retrieval-k ──
if "selected_course" not in st.session_state:
    st.session_state.selected_course = "All Courses"
if "retrieval_k" not in st.session_state:
    st.session_state.retrieval_k = Config.RETRIEVAL_TOP_K


def render_unichat():
    """UniChat chatbot page"""
    # UniChat logo row
    unichat_logo_html = ""
    if UNICHAT_LOGO.exists():
        unichat_src = img_to_data_uri(UNICHAT_LOGO)
        unichat_logo_html = (
        '<div id="cu-unichat-logo-row">'
        f'<img src="{unichat_src}" alt="UniChat">'
        '<div><h2 style="margin:0;color:#1a1a1a;font-size:24px;">UniChat</h2>'
        '<p style="margin:2px 0 0;color:#888;font-size:13px;">'
        'AI Chatbot for Carleton Students</p></div></div>'
    )
    else:
        unichat_logo_html = (
            '<div id="cu-unichat-logo-row">'
            '<div style="background:#B3112D;border-radius:10px;padding:10px 16px;'
            'color:#fff;font-weight:700;font-size:20px;">🤖 UniChat</div>'
            '<div><p style="margin:4px 0 0;color:#888;font-size:13px;">'
            'AI Chatbot for Carleton Students</p></div></div>'
        )
    st.markdown(unichat_logo_html, unsafe_allow_html=True)

    # ── Inline controls row (course filter + retrieval k) ──
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([3, 2, 1.5])
    with ctrl_col1:
        course_options = ["All Courses"] + sorted(available_courses)
        st.session_state.selected_course = st.selectbox(
            "Course Filter",
            course_options,
            index=course_options.index(st.session_state.selected_course)
            if st.session_state.selected_course in course_options else 0,
            help="Filter responses to a specific course",
            label_visibility="visible",
        )
    with ctrl_col2:
        st.session_state.retrieval_k = st.slider(
            "Results to Retrieve",
            min_value=3, max_value=15,
            value=st.session_state.retrieval_k,
            help="Number of chunks retrieved",
        )
    with ctrl_col3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Chat history ──
    recent_messages = st.session_state.messages[-10:]
    for message in recent_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # ── Example question pills ──
    example_questions = [
        "What are the prerequisites for this course?",
        "Who teaches this course?",
        "What is the grading scheme?",
        "What are the learning outcomes?",
        "Explain the topics covered",
        "What textbook is recommended?",
    ]
    pills_html = '<div class="cu-examples">'
    for i, q in enumerate(example_questions):
        pills_html += (
            f'<div class="cu-example-pill" '
            f'onclick="document.getElementById(\'cu-example-input\').value=\'{q}\'; '
            f'document.getElementById(\'cu-example-input\').dispatchEvent(new Event(\'input\'));">'
            f'{q}</div>'
        )
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    # Streamlit-based example buttons (fallback since onclick on pills won't trigger st rerun)
    ex_cols = st.columns(len(example_questions))
    for i, q in enumerate(example_questions):
        with ex_cols[i]:
            if st.button(q, key=f"ex_{i}", use_container_width=True,
                         help=q, type="secondary"):
                st.session_state.pending_example = q
                st.rerun()

    # ── Chat input ──
    # If an example was clicked, pre-fill
    default_val = ""
    if st.session_state.pending_example:
        default_val = st.session_state.pending_example
        st.session_state.pending_example = None

    user_question = st.chat_input("Ask about course outlines…", key="chat_input")

    # Also accept the pending example as a question
    if not user_question and default_val:
        user_question = default_val

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append(HumanMessage(user_question))

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("🔍 Searching databases…"):
                try:
                    cf = (st.session_state.selected_course
                          if st.session_state.selected_course != "All Courses" else None)
                    response, metadata, retrieval_time, generation_time, metrics = (
                        generate_enhanced_response(
                            user_question,
                            k=st.session_state.retrieval_k,
                            course_filter=cf,
                        )
                    )
                    message_placeholder.markdown(response)
                    st.session_state.messages.append(AIMessage(response))

                    # Inline metrics badges
                    if metrics:
                        badges = (
                            '<div class="cu-metric-row">'
                            f'<div class="cu-metric-badge">'
                            f'<div class="cu-m-label">Total Time</div>'
                            f'<div class="cu-m-value">{metrics.total_time:.2f}s</div></div>'

                            f'<div class="cu-metric-badge">'
                            f'<div class="cu-m-label">Retrieval</div>'
                            f'<div class="cu-m-value">{metrics.retrieval_time:.2f}s</div></div>'

                            f'<div class="cu-metric-badge">'
                            f'<div class="cu-m-label">Generation</div>'
                            f'<div class="cu-m-value">{metrics.generation_time:.2f}s</div></div>'

                            f'<div class="cu-metric-badge">'
                            f'<div class="cu-m-label">Chunks Used</div>'
                            f'<div class="cu-m-value">{metrics.num_total_results}</div></div>'

                            f'<div class="cu-metric-badge">'
                            f'<div class="cu-m-label">Memory</div>'
                            f'<div class="cu-m-value">{metrics.memory_used_mb:.1f} MB</div></div>'
                            '</div>'
                        )
                        st.markdown(badges, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    with st.expander("Error Details"):
                        import traceback
                        st.code(traceback.format_exc())


def render_about():
    """About Us page"""
    st.markdown(
        '<h2 class="cu-page-title">About Us</h2>'
        '<p class="cu-page-subtitle">Learn more about the UniChat project</p>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="cu-section-card">
        <h3>🎓 What is UniChat?</h3>
        <p>UniChat is an AI-powered chatbot built specifically for <strong>Carleton University</strong> students.
        It uses advanced Retrieval-Augmented Generation (RAG) technology to help you quickly find answers
        about course outlines, prerequisites, grading schemes, learning outcomes, and more.</p>
    </div>

    <div class="cu-section-card">
        <h3>🔧 Technology Stack</h3>
        <ul>
            <li><strong>Vector Database (FAISS)</strong> – Stores and searches course content using semantic embeddings.</li>
            <li><strong>Knowledge Graph</strong> – Captures relationships between courses, prerequisites, and topics.</li>
            <li><strong>Hybrid Retrieval</strong> – Combines vector similarity search with knowledge-graph traversal for richer context.</li>
            <li><strong>Large Language Models</strong> – Supports OpenAI, Ollama, Anthropic, and Cohere backends.</li>
            <li><strong>Streamlit</strong> – Provides the interactive web interface.</li>
        </ul>
    </div>

    <div class="cu-section-card">
        <h3>📌 How to Use</h3>
        <p>Simply type a question in the UniChat input box — for example, <em>"What are the prerequisites for COMP 3001?"</em>
        — and UniChat will retrieve relevant information from Carleton's course outlines and generate a clear, accurate answer.
        You can also use the <strong>Course Filter</strong> to narrow results to a specific course.</p>
    </div>

    <div class="cu-section-card">
        <h3>📞 Contact</h3>
        <p>For questions or feedback about UniChat, please reach out to the project team through the
        Carleton University Computer Science department.</p>
    </div>
    """, unsafe_allow_html=True)


def render_faq():
    """FAQ page with accordion-style questions"""
    st.markdown(
        '<h2 class="cu-page-title">Frequently Asked Questions</h2>'
        '<p class="cu-page-subtitle">Answers to common questions about UniChat</p>',
        unsafe_allow_html=True,
    )

    faqs = [
        ("What types of questions can I ask UniChat?",
         "You can ask about prerequisites, grading breakdowns, course descriptions, learning outcomes, "
         "instructors, textbooks, schedules, and any other information contained in Carleton course outlines."),

        ("How accurate are the answers?",
         "UniChat retrieves information directly from indexed course outlines and uses an LLM to synthesise the answer. "
         "Each response includes source attribution and a confidence indicator. For critical decisions, always cross-reference with official Carleton sources."),

        ("Can I filter by a specific course?",
         "Yes! Use the Course Filter dropdown on the UniChat page to restrict answers to a particular course code. "
         "This is especially useful when asking about prerequisites or grading for a specific class."),

        ("What happens if UniChat doesn't know the answer?",
         "If no relevant information is found in the indexed course outlines, UniChat will clearly state that it could not find a match "
         "rather than guessing. You can then try rephrasing your question or removing the course filter."),

        ("How do I add new course outlines?",
         "Course outlines are ingested using the enhanced_ingestion.py script. An administrator needs to place the new PDF or JSONL files "
         "in the configured directory and re-run the ingestion pipeline to rebuild the vector store and knowledge graph."),

        ("Is my data private?",
         "UniChat runs locally on your institution's servers. Queries are not sent to external services beyond the configured LLM provider. "
         "Check with your IT department for the full data-handling policy."),

        ("Which LLM providers are supported?",
         "UniChat currently supports OpenAI (GPT-4o, GPT-4o-mini), Ollama (Llama, Mistral, Phi), Anthropic (Claude), and Cohere. "
         "The active provider is configured in the .env file."),
    ]

    for i, (q, a) in enumerate(faqs):
        st.markdown(
            f'<div class="cu-faq-q" onclick="'
            f'this.nextElementSibling.style.display='
            f'this.nextElementSibling.style.display===\'none\'?\'block\':\'none\';">'
            f'{q} <span>▼</span></div>'
            f'<div class="cu-faq-a" style="display:none;">{a}</div>',
            unsafe_allow_html=True,
        )


# ── EXTENSIBILITY HOOK: register new page renderers here ──
PAGE_RENDERERS = {
    "unichat": render_unichat,
    "about":   render_about,
    "faq":     render_faq,
    # "mypage": render_mypage,   ← add new pages in the same pattern
}


# ─────────────────────────────────────────────
# MAIN LAYOUT ASSEMBLY
# ─────────────────────────────────────────────

# Inject global CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Build the HTML skeleton (left pane + top nav) ──
left_logo = _logo_tag(CARLETON_LOGO, "cu-logo", "Carleton University")

nav_buttons = ""
for page in NAV_PAGES:
    active_cls = "active" if st.session_state.active_page == page["key"] else ""
    pkey   = page["key"]
    plabel = page["label"]
    nav_buttons += (
        f'<span class="cu-nav-btn {active_cls}" id="navbtn-{pkey}">{plabel}</span>'
    )

layout_html = f"""
<div id="cu-layout">
  <!-- LEFT PANE -->
  <div id="cu-left">
    {left_logo}
    <div class="cu-divider"></div>
    <div class="cu-desc">
      <strong>Carleton University</strong><br>
      AI-powered academic assistant helping students navigate course outlines, prerequisites, and more.
    </div>
  </div>

  <!-- RIGHT PANE -->
  <div id="cu-right">
    <!-- TOP NAV (visual shell – real buttons rendered below) -->
    <div id="cu-topnav" style="visibility:hidden; height:0; padding:0; overflow:hidden;"></div>
    <div id="cu-red-rule"></div>
  </div>
</div>
"""
st.markdown(layout_html, unsafe_allow_html=True)

# ── Render the REAL nav bar using Streamlit buttons (for rerun support) ──
# We use a custom HTML wrapper to style them as our nav bar.
nav_html_open = (
    '<div id="cu-topnav" style="'
    'display:flex;align-items:center;background:#1a1a1a;'
    'padding:0 28px;height:56px;gap:4px;'
    'position:sticky;top:0;z-index:99;'
    'box-shadow:0 2px 8px rgba(0,0,0,.2);">'
)
st.markdown(nav_html_open, unsafe_allow_html=True)

nav_cols = st.columns(len(NAV_PAGES))
for i, page in enumerate(NAV_PAGES):
    with nav_cols[i]:
        is_active = st.session_state.active_page == page["key"]
        # Style active vs inactive via custom CSS class on a button
        btn_style = (
            "background:#B3112D;color:#fff;border:none;padding:8px 20px;"
            "border-radius:6px;font-size:14px;font-weight:600;cursor:pointer;"
            if is_active else
            "background:transparent;color:#e0e0e0;border:none;padding:8px 20px;"
            "border-radius:6px;font-size:14px;font-weight:500;cursor:pointer;"
        )
        # Wrap st.button click
        if st.button(
            page["label"],
            key=f"nav_{page['key']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_page = page["key"]
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Red rule
st.markdown('<div id="cu-red-rule" style="height:3px;background:#B3112D;"></div>',
            unsafe_allow_html=True)

# ── Render active page content ──
st.markdown('<div id="cu-content">', unsafe_allow_html=True)

renderer = PAGE_RENDERERS.get(st.session_state.active_page, render_unichat)
renderer()

st.markdown('</div>', unsafe_allow_html=True)

# ── Session metrics footer (subtle, always visible) ──
tracker = st.session_state.metrics_tracker
if tracker.metrics_buffer:
    summary = tracker.get_summary()
    st.markdown(
        f'<div style="margin-top:24px;padding-top:12px;border-top:1px solid #eee;'
        f'display:flex;gap:24px;font-size:12px;color:#999;">'
        f'<span>📊 Queries this session: <strong>{summary["total_queries"]}</strong></span>'
        f'<span>⏱ Avg response: <strong>{summary["timing"]["avg_total_time"]:.2f}s</strong></span>',
        unsafe_allow_html=True,
    )
    col_save, _ = st.columns([1, 4])
    with col_save:
        if st.button("💾 Save Session Metrics", type="secondary"):
            tracker.save_summary(silent=False)
            st.success("Metrics saved!")
    st.markdown('</div>', unsafe_allow_html=True)