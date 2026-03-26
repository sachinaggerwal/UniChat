"""
enhanced_chatbot_metrics.py - Windows-compatible Streamlit chatbot
Enhanced with hybrid Vector DB + Knowledge Graph retrieval + Metrics tracking
Run: streamlit run enhanced_chatbot_metrics.py
"""
import os
import warnings

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import os
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

# Page configuration
st.set_page_config(
    page_title="🎓 Enhanced Course Outline Assistant",
    page_icon="🎓",
    layout="wide",
)

# Initialize metrics tracker
if "metrics_tracker" not in st.session_state:
    st.session_state.metrics_tracker = MetricsTracker(log_dir="chatbot_metrics")

# ----------------- Initialize Resources -----------------


@st.cache_resource(show_spinner=False)
def initialize_system():
    """Initialize all system components (Windows-compatible)"""
    try:
        # Print config (suppress in Streamlit)
        if not st.session_state.get("config_printed", False):
            Config.print_summary()
            st.session_state.config_printed = True

        # Check if vector store exists
        persist_dir = Path(Config.PERSIST_DIR)
        if not persist_dir.exists():
            st.error(f"❌ Vector database not found at: {persist_dir.resolve()}")
            st.info("📝 Please run: python enhanced_ingestion.py")
            st.stop()

        # Initialize embeddings
        embeddings = ModelFactory.create_embeddings()

        # Load vector store
        vector_store = FAISS.load_local(
            str(persist_dir), embeddings, allow_dangerous_deserialization=True
        )

        # Check if KG exists
        kg_dir = Path(Config.KG_PERSIST_DIR)
        if not kg_dir.exists():
            st.warning("⚠️ Knowledge graph not found. Running in vector-only mode.")
            kg = None
        else:
            # Load knowledge graph
            kg = CourseKnowledgeGraph(persist_dir=str(kg_dir))
            kg.load()

        # Initialize hybrid retriever (or vector-only if no KG)
        if kg:
            retriever = HybridRetriever(vector_store, kg)
        else:
            # Fallback to vector-only retriever
            class VectorOnlyRetriever:
                def __init__(self, vector_store):
                    self.vector_store = vector_store
                
                def retrieve(self, query, k=8, course_filter=None):
                    docs_with_scores = self.vector_store.similarity_search_with_score(
                        query, k=k
                    )
                    
                    # Apply course filter
                    if course_filter and course_filter != "All Courses":
                        docs_with_scores = [
                            (doc, score) for doc, score in docs_with_scores
                            if doc.metadata.get("course_code") == course_filter
                        ]
                    
                    # Convert to RetrievalResult format
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

        # Initialize LLM
        llm = ModelFactory.create_llm()

        # Load course codes
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
        
        # Show detailed error in expander
        with st.expander("Error Details"):
            import traceback
            st.code(traceback.format_exc())
        
        st.stop()


# Initialize system
with st.spinner("🚀 Initializing Enhanced RAG System..."):
    system = initialize_system()

vector_store = system["vector_store"]
kg = system["kg"]
retriever = system["retriever"]
llm = system["llm"]
available_courses = system["course_codes"]

# ----------------- Enhanced RAG Prompt -----------------
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


# ----------------- Generate Response with Metrics -----------------
def generate_enhanced_response(
    question: str, k: int = None, course_filter: Optional[str] = None
) -> tuple:
    """Generate response using hybrid retrieval with metrics tracking"""
    start_time = time.time()
    k = k or Config.RETRIEVAL_TOP_K

    # Start metrics tracking
    tracker = st.session_state.metrics_tracker
    tracking_data = tracker.start_query()

    # Enhance question with course context
    enhanced_question = question
    if course_filter and course_filter != "All Courses":
        enhanced_question = f"For course {course_filter}: {question}"

    # Hybrid retrieval
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

    # Format context for LLM
    context = retriever.format_results_for_llm(results, max_length=3500)

    # Generate response
    messages = rag_prompt.format_messages(context=context, question=question)

    generation_start = time.time()
    response = llm.invoke(messages)
    generation_time = time.time() - generation_start

    # Extract answer
    answer = response.content if hasattr(response, "content") else str(response)

    # Get course references
    courses_referenced = list(
        dict.fromkeys([r.course_code for r in results if r.course_code])
    )

    # Track metrics
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

    # Add source information
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


# ----------------- UI Layout -----------------
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🎓 Enhanced Course Outline Assistant")
    st.caption(
        f"Powered by {Config.LLM_PROVIDER.upper()} {Config.LLM_MODEL} + "
        f"{'Hybrid Vector+KG' if kg else 'Vector-only'} Retrieval"
    )

with col2:
    st.subheader("📊 System Status")
    try:
        doc_count = vector_store.index.ntotal
        st.metric("Vector Chunks", f"{doc_count:,}")
        
        if kg:
            kg_stats = kg.get_statistics()
            st.metric("KG Nodes", f"{kg_stats['total_nodes']:,}")
        
        st.metric("Courses", len(available_courses))
    except Exception:
        st.warning("Stats unavailable")

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Model info
    with st.expander("🤖 Model Settings", expanded=False):
        st.text(f"Embeddings: {Config.EMBEDDING_PROVIDER}")
        st.text(f"Model: {Config.EMBEDDING_MODEL}")
        st.text(f"LLM: {Config.LLM_PROVIDER}")
        st.text(f"Model: {Config.LLM_MODEL}")

    # Course filter
    st.markdown("### 🎯 Course Filter")
    course_options = ["All Courses"] + sorted(available_courses)
    selected_course = st.selectbox(
        "Select a course:",
        course_options,
        help="Filter responses to a specific course",
    )

    # Retrieval settings
    st.markdown("### 🔍 Retrieval Settings")
    retrieval_k = st.slider(
        "Results to retrieve:",
        min_value=3,
        max_value=15,
        value=Config.RETRIEVAL_TOP_K,
        help="Number of results from each source",
    )

    # Fusion strategy (only if KG available)
    if kg:
        fusion_strategy = st.selectbox(
            "Fusion Strategy:",
            ["adaptive", "weighted", "rrf"],
            index=["adaptive", "weighted", "rrf"].index(Config.FUSION_STRATEGY),
            help="How to combine vector and KG results",
        )
        Config.FUSION_STRATEGY = fusion_strategy

        # Weights (only for weighted strategy)
        if fusion_strategy == "weighted":
            st.markdown("#### Retrieval Weights")
            vector_weight = st.slider(
                "Vector DB Weight:", 0.0, 1.0, Config.VECTOR_WEIGHT, 0.1
            )
            kg_weight = 1.0 - vector_weight
            st.text(f"KG Weight: {kg_weight:.1f}")
            Config.VECTOR_WEIGHT = vector_weight
            Config.KG_WEIGHT = kg_weight

    # Performance metrics
    st.markdown("### 📈 Session Metrics")
    tracker = st.session_state.metrics_tracker
    if tracker.metrics_buffer:
        summary = tracker.get_summary()
        st.metric("Total Queries", summary["total_queries"])
        st.metric("Avg Response Time", f"{summary['timing']['avg_total_time']:.2f}s")
        if "avg_accuracy_score" in summary.get("quality", {}):
            st.metric("Avg Accuracy", f"{summary['quality']['avg_accuracy_score']:.2f}")

    # Metrics actions
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📊 Save Metrics"):
            tracker.save_summary(silent=False)
            st.success("Metrics saved!")
    with col_b:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

# ----------------- Chat Interface -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
chat_container = st.container()

with chat_container:
    # Show last 10 messages
    recent_messages = st.session_state.messages[-10:]
    for message in recent_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

# Example questions
with st.expander("💡 Example Questions"):
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Factual Queries:**")
        st.markdown(
            """
        - What are the prerequisites for this course?
        - Who teaches this course?
        - What is the grading scheme?
        - When is the final exam?
        - What percentage is the midterm worth?
        """
        )

    with col_b:
        st.markdown("**Conceptual Queries:**")
        st.markdown(
            """
        - What are the learning outcomes?
        - Explain the topics covered
        - What is this course about?
        - Describe the course structure
        - What textbook is recommended?
        """
        )

# Chat input
user_question = st.chat_input("Ask about course outlines...")

if user_question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))

    # Generate and display response
    with st.chat_message("assistant"):
        # Create placeholder for response
        message_placeholder = st.empty()

        # Show searching status
        with st.spinner("🔍 Searching databases..."):
            try:
                # Generate response
                response, metadata, retrieval_time, generation_time, metrics = (
                    generate_enhanced_response(
                        user_question,
                        k=retrieval_k,
                        course_filter=(
                            selected_course
                            if selected_course != "All Courses"
                            else None
                        ),
                    )
                )

                # Display response
                message_placeholder.markdown(response)
                st.session_state.messages.append(AIMessage(response))

                # Show detailed metrics in expander
                with st.expander("📊 Query Metrics"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Time", f"{metrics.total_time:.3f}s")
                        st.metric("Retrieval", f"{metrics.retrieval_time:.3f}s")
                    with col2:
                        st.metric("Generation", f"{metrics.generation_time:.3f}s")
                        st.metric("Relevance", f"{metrics.relevance_score:.2f}")
                    with col3:
                        st.metric("Results", metrics.num_total_results)
                        st.metric("Memory", f"{metrics.memory_used_mb:.1f} MB")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info(
                    "Check configuration and ensure all components are initialized."
                )
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())

# Footer
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
