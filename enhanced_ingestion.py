"""
enhanced_ingestion.py - Windows-compatible ingestion script
Enhanced with better error handling, progress tracking, and Windows path support
Run: python enhanced_ingestion.py
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

# Import from our modules
from config import Config
from model_factory import ModelFactory
from knowledge_graph import CourseKnowledgeGraph


def extract_course_metadata(raw_text: str, course_code: str) -> Dict:
    """Extract structured metadata from course outline text"""
    metadata = {"course_code": course_code}

    patterns = {
        "course_name": r"(?:course\s+name|course\s+title)[:\s]+([^\n]+)",
        "credits": r"(?:credits?|credit\s+hours?)[:\s]+(\d+)",
        "department": r"(?:department|dept)[:\s]+([^\n]+)",
        "instructor": r"(?:instructor|teacher|professor)[:\s]+([^\n]+)",
        "semester": r"(?:semester|term)[:\s]+([^\n]+)",
    }

    text_lower = raw_text.lower()
    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            metadata[key] = match.group(1).strip()[:100]

    return metadata


def validate_course_outlines_file() -> bool:
    """Validate that course outlines file exists and is readable"""
    course_file = Path(Config.COURSE_OUTLINES_FILE)
    
    if not course_file.exists():
        print(f"\n❌ Error: Course outlines file not found: {course_file}")
        print("\n📝 Please run Course_Outline_Extractor.py first to create this file.")
        print("   This will extract text from your PDF/DOCX course outline files.")
        return False
    
    if course_file.stat().st_size == 0:
        print(f"\n❌ Error: Course outlines file is empty: {course_file}")
        return False
    
    return True


def load_course_outlines() -> List[Dict]:
    """Load and validate course outlines from JSONL file"""
    print(f"\n3️⃣ Loading course outlines from {Config.COURSE_OUTLINES_FILE}...")
    
    file_content = []
    course_file = Path(Config.COURSE_OUTLINES_FILE)
    
    try:
        with open(course_file, encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    
                    # Validate required fields
                    if "raw_text" not in obj:
                        print(f"   ⚠️  Skipping line {idx}: missing raw_text")
                        continue
                    
                    if not obj["raw_text"].strip():
                        print(f"   ⚠️  Skipping line {idx}: empty raw_text")
                        continue
                    
                    # Set defaults for optional fields
                    if "title" not in obj:
                        obj["title"] = f"course-outline-{idx}"
                    if "url" not in obj:
                        obj["url"] = obj["title"][:15]
                    
                    file_content.append(obj)
                    
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Skipping line {idx}: JSON decode error - {e}")
                    continue
        
        print(f"   ✅ Loaded {len(file_content)} course outlines")
        
        if not file_content:
            print("\n❌ Error: No valid course outlines found!")
            print("   Please check your course_outlines.jsonl file.")
            return None
        
        return file_content
        
    except Exception as e:
        print(f"\n❌ Error reading course outlines file: {e}")
        return None


def prepare_documents(course_outlines: List[Dict], text_splitter) -> tuple:
    """Prepare documents for vector store with progress tracking"""
    print("\n4️⃣ Chunking and preparing documents...")
    
    all_docs = []
    all_metadatas = []
    
    with tqdm(total=len(course_outlines), desc="Processing courses") as pbar:
        for course in course_outlines:
            raw = course["raw_text"]
            title = course.get("title", "")
            course_code = course.get("url", "").strip()
            
            # Extract metadata
            extracted_meta = extract_course_metadata(raw, course_code)
            
            # Split into chunks
            chunks = text_splitter.split_text(raw)
            
            for i, chunk in enumerate(chunks):
                all_docs.append(chunk)
                
                chunk_metadata = {
                    "source": course_code,
                    "title": title,
                    "course_code": course_code,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    **extracted_meta,
                }
                all_metadatas.append(chunk_metadata)
            
            pbar.set_postfix({"chunks": len(chunks), "course": course_code})
            pbar.update(1)
    
    print(f"\n   ✅ Total chunks prepared: {len(all_docs):,}")
    
    return all_docs, all_metadatas


def create_vector_store(all_docs: List[str], all_metadatas: List[Dict], embeddings):
    """Create FAISS vector store with progress indication"""
    print("\n5️⃣ Creating FAISS vector store...")
    print("   ⏳ Generating embeddings (this may take a few minutes)...")
    
    try:
        # Create vector store with progress
        print(f"   📊 Processing {len(all_docs):,} chunks...")
        
        vector_store = FAISS.from_texts(
            texts=all_docs, 
            embedding=embeddings, 
            metadatas=all_metadatas
        )
        
        print("   ✅ Vector store created successfully")
        return vector_store
        
    except Exception as e:
        print(f"\n❌ Error creating vector store: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def save_vector_store(vector_store, all_metadatas: List[Dict]):
    """Save vector store and course codes to disk (Windows-compatible)"""
    print(f"\n6️⃣ Saving vector store to {Config.PERSIST_DIR}...")
    
    try:
        # Ensure directory exists
        persist_dir = Path(Config.PERSIST_DIR)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vector store
        vector_store.save_local(str(persist_dir))
        print("   ✅ Vector store saved")
        
        # Save course codes list
        course_codes = sorted(set(m["course_code"] for m in all_metadatas))
        course_codes_file = persist_dir / "course_codes.json"
        
        with open(course_codes_file, "w", encoding="utf-8") as f:
            json.dump(course_codes, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Course codes saved: {len(course_codes)} courses")
        
        # Display sample courses
        sample_courses = course_codes[:5]
        if len(course_codes) > 5:
            sample_courses.append("...")
        print(f"   📚 Courses: {', '.join(sample_courses)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error saving vector store: {e}")
        return False


def build_knowledge_graph(course_outlines: List[Dict]):
    """Build and save knowledge graph"""
    print("\n7️⃣ Building Knowledge Graph...")
    
    try:
        kg = CourseKnowledgeGraph(persist_dir=Config.KG_PERSIST_DIR)
        
        # Build from documents with progress
        print("   ⏳ Extracting entities and relationships...")
        kg.build_from_documents(course_outlines, show_progress=True)
        
        # Save knowledge graph
        print("\n8️⃣ Saving Knowledge Graph...")
        kg.save()
        
        return kg
        
    except Exception as e:
        print(f"\n❌ Error building knowledge graph: {e}")
        import traceback
        print(traceback.format_exc())
        return None


def print_statistics(all_docs: List[str], course_outlines: List[Dict], kg):
    """Print comprehensive statistics"""
    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    
    # Vector Store Statistics
    print(f"\n📊 Vector Store Statistics:")
    print(f"   • Total chunks: {len(all_docs):,}")
    print(f"   • Total courses: {len(course_outlines)}")
    print(f"   • Avg chunks per course: {len(all_docs) / len(course_outlines):.1f}")
    print(f"   • Embedding: {Config.EMBEDDING_PROVIDER}/{Config.EMBEDDING_MODEL}")
    
    # Knowledge Graph Statistics
    if kg:
        kg_stats = kg.get_statistics()
        print(f"\n🕸️  Knowledge Graph Statistics:")
        print(f"   • Total nodes: {kg_stats['total_nodes']:,}")
        print(f"   • Total edges: {kg_stats['total_edges']:,}")
        print(f"   • Courses: {kg_stats['courses']}")
        
        print(f"\n   Entity Types:")
        for entity_type, count in sorted(kg_stats["entity_types"].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"     - {entity_type}: {count}")
        
        print(f"\n   Relationship Types:")
        for rel_type, count in sorted(kg_stats["relation_types"].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"     - {rel_type}: {count}")
    
    # File locations
    print(f"\n📁 Files saved:")
    print(f"   • Vector DB: {Path(Config.PERSIST_DIR).resolve()}")
    print(f"   • Knowledge Graph: {Path(Config.KG_PERSIST_DIR).resolve()}")
    
    # Next steps
    print("\n" + "=" * 70)
    print("✅ Ready to run the chatbot!")
    print("=" * 70)
    print("\nNext steps:")
    print("   1. Run the chatbot:")
    print("      streamlit run enhanced_chatbot_metrics.py")
    print("\n   2. Or test different models:")
    print("      python model_tester.py")
    print("\n" + "=" * 70 + "\n")


def main():
    """Main ingestion workflow"""
    print("\n" + "=" * 70)
    print("ENHANCED COURSE OUTLINE INGESTION")
    print("=" * 70)
    
    # Print configuration
    Config.print_summary()
    
    # Validate configuration
    print("\n1️⃣ Validating configuration...")
    if not Config.validate():
        print("\n❌ Configuration validation failed. Please check your .env file.")
        print("   Run 'python setup_windows.py' to diagnose issues.")
        return False
    print("   ✅ Configuration valid")
    
    # Check if course outlines file exists
    if not validate_course_outlines_file():
        return False
    
    # Initialize components
    print("\n2️⃣ Initializing components...")
    try:
        embeddings = ModelFactory.create_embeddings()
        print(f"   ✅ Embeddings: {Config.EMBEDDING_PROVIDER}/{Config.EMBEDDING_MODEL}")
        
        text_splitter = ModelFactory.get_text_splitter()
        print(f"   ✅ Text splitter: chunk_size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP}")
    except Exception as e:
        print(f"\n❌ Error initializing components: {e}")
        print("   Make sure all required packages are installed:")
        print("   pip install -r requirements_windows.txt")
        return False
    
    # Load course outlines
    course_outlines = load_course_outlines()
    if not course_outlines:
        return False
    
    # Prepare documents
    all_docs, all_metadatas = prepare_documents(course_outlines, text_splitter)
    if not all_docs:
        print("\n❌ No documents prepared. Check your course outlines.")
        return False
    
    # Create vector store
    vector_store = create_vector_store(all_docs, all_metadatas, embeddings)
    if not vector_store:
        return False
    
    # Save vector store
    if not save_vector_store(vector_store, all_metadatas):
        return False
    
    # Build knowledge graph
    kg = build_knowledge_graph(course_outlines)
    
    # Print statistics
    print_statistics(all_docs, course_outlines, kg)
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
