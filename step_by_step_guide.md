# Complete Step-by-Step Guide - Windows RAG System

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Document Preparation](#document-preparation)
4. [Database Building](#database-building)
5. [Running the Chatbot](#running-the-chatbot)
6. [Testing Different Models](#testing-different-models)
7. [Troubleshooting](#troubleshooting)

---

## 1️⃣ Prerequisites

### Required Software

**Python 3.8 or higher**
```bash
# Check your Python version
python --version

# Should show: Python 3.8.x or higher
```

**Download Python:** [python.org/downloads](https://www.python.org/downloads/)
- ⚠️ **Important:** Check "Add Python to PATH" during installation

**Git (Optional)** - For cloning repositories
- Download: [git-scm.com](https://git-scm.com/)

### System Requirements

- **OS:** Windows 10/11 (64-bit)
- **RAM:** 8GB minimum, 16GB recommended
- **Disk Space:** 10GB+ free
- **GPU:** Optional (NVIDIA GPU with CUDA for faster processing)

---

## 2️⃣ Initial Setup

### Step 2.1: Get the Project Files

**Option A: Extract from ZIP**
```bash
# Extract the project ZIP file to a folder, e.g.:
C:\Projects\course-rag-system\
```

**Option B: Clone from Git**
```bash
cd C:\Projects
git clone your-repository-url course-rag-system
cd course-rag-system
```

### Step 2.2: Open Command Prompt

1. Press `Win + R`
2. Type `cmd` and press Enter
3. Navigate to your project folder:
```bash
cd C:\Projects\course-rag-system
```

### Step 2.3: Run Automated Setup

```bash
python setup_windows.py
```

**What this does:**
- ✅ Checks Python version
- ✅ Checks disk space
- ✅ Installs required packages
- ✅ Creates necessary folders
- ✅ Creates `.env` configuration file
- ✅ Tests if everything works

**Expected Output:**
```
==================================================================
Windows RAG System Setup
==================================================================

Checking Platform
==================================================================
✓ Windows 10

Checking Python Version
==================================================================
✓ Python 3.11.5

...

Setup Complete!
```

**If prompted to install packages, type `y` and press Enter.**

---

### Step 2.4: Configure API Keys (If Using OpenAI)

**Option A: Using OpenAI (Recommended for best results)**

1. Open `.env` file in a text editor (Notepad, VS Code, etc.)
2. Add your OpenAI API key:
```properties
OPENAI_API_KEY=sk-your-actual-api-key-here

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

**Option B: Using Local Models (Free, No API Key)**

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Open a **NEW** Command Prompt and run:
```bash
ollama serve
```
3. In **ANOTHER** Command Prompt, pull models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
4. Edit `.env`:
```properties
# Leave OPENAI_API_KEY blank or remove it

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

**Option C: Mixed (Local Embeddings + OpenAI LLM)**
```properties
OPENAI_API_KEY=sk-your-key-here

# Local embeddings (free)
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# OpenAI LLM (paid but fast)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

---

## 3️⃣ Document Preparation

### Step 3.1: Organize Your Course Outlines

Create this folder structure:
```
C:\Projects\course-rag-system\
  └── Course Materials\
      ├── BIT2001B\
      │   ├── Course_Outline.pdf
      │   └── syllabus.docx
      ├── CS101\
      │   └── outline.pdf
      └── MATH201\
          └── course_info.pdf
```

**Supported file types:**
- ✅ PDF files (`.pdf`)
- ✅ Word documents (`.docx`)
- ✅ Old Word documents (`.doc`)

### Step 3.2: Update Extractor Path

1. Open `Course_Outline_Extractor.py`
2. Find this line (near the bottom):
```python
root_folder = r"/Users/sauravbhattacharyya/Downloads/Course Materials"
```
3. Change it to your path:
```python
root_folder = r"C:\Projects\course-rag-system\Course Materials"
```

### Step 3.3: Extract Course Outlines

```bash
python Course_Outline_Extractor.py
```

**Expected Output:**
```
================================================================
Course Outline Extractor
================================================================

Processing course: BIT2001B
  Extracting from PDF: Course_Outline.pdf
  ✓ Extracted 15234 characters

Processing course: CS101
  Extracting from PDF: outline.pdf
  ✓ Extracted 8567 characters

✓ Saved 2 course outlines to course_outlines.jsonl

Total courses processed: 2
```

**Verify:** Check that `course_outlines.jsonl` file was created in your project folder.

---

## 4️⃣ Database Building

### Step 4.1: Build Vector Database and Knowledge Graph

```bash
python enhanced_ingestion.py
```

**What this does:**
- 📦 Loads your course outlines
- ✂️ Splits text into chunks
- 🔢 Creates embeddings (vector representations)
- 💾 Builds FAISS vector database
- 🕸️ Extracts entities and relationships
- 💾 Builds knowledge graph

**Expected Output:**
```
======================================================================
ENHANCED COURSE OUTLINE INGESTION
======================================================================

RAG System Configuration
======================================================================
Embedding......................... openai/text-embedding-3-small
LLM............................... openai/gpt-4o-mini
...

1️⃣ Validating configuration...
   ✅ Configuration valid

2️⃣ Initializing components...
   ✅ Embeddings: openai/text-embedding-3-small
   ✅ Text splitter: chunk_size=500, overlap=150

3️⃣ Loading course outlines from course_outlines.jsonl...
   ✅ Loaded 2 course outlines

4️⃣ Chunking and preparing documents...
Processing courses: 100%|████████████| 2/2 [00:00<00:00]

   ✅ Total chunks prepared: 45

5️⃣ Creating FAISS vector store...
   ⏳ Generating embeddings (this may take a few minutes)...
   📊 Processing 45 chunks...
   ✅ Vector store created successfully

6️⃣ Saving vector store to faiss_course_db...
   ✅ Vector store saved
   ✅ Course codes saved: 2 courses

7️⃣ Building Knowledge Graph...
   ⏳ Extracting entities and relationships...
   Processing: 2/2...
✅ Knowledge Graph built:
   - Nodes: 87
   - Edges: 123
   - Courses: 2

8️⃣ Saving Knowledge Graph...
💾 Knowledge Graph saved to C:\...\knowledge_graph_db

======================================================================
INGESTION COMPLETE
======================================================================

📊 Vector Store Statistics:
   • Total chunks: 45
   • Total courses: 2
   • Embedding: openai/text-embedding-3-small

🕸️  Knowledge Graph Statistics:
   • Total nodes: 87
   • Total edges: 123

✅ Ready to run the chatbot!

Next steps:
   1. Run the chatbot:
      streamlit run enhanced_chatbot_metrics.py
```

**Time Estimate:**
- OpenAI: 2-5 minutes (depends on internet speed)
- Local models: 5-15 minutes (depends on CPU/GPU)

---

## 5️⃣ Running the Chatbot

### Step 5.1: Start the Chatbot

```bash
streamlit run enhanced_chatbot_metrics.py
```

**Expected Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.x:8501
```

### Step 5.2: Use the Chatbot

1. **Browser opens automatically** to `http://localhost:8501`
2. **Try example questions:**
   - "What are the prerequisites for BIT2001B?"
   - "Who teaches this course?"
   - "What is the grading scheme?"
   - "Explain the course topics"

### Step 5.3: Chatbot Features

**Sidebar Configuration:**
- 🎯 **Course Filter** - Filter by specific course
- 🔍 **Retrieval Settings** - Adjust number of results
- 🔀 **Fusion Strategy** - Adaptive, weighted, or RRF
- 📊 **Session Metrics** - View performance stats

**Chat Interface:**
- 💬 Ask questions in natural language
- 📊 View query metrics (timing, relevance)
- 📖 See which courses were referenced
- 🔍 View information sources

### Step 5.4: Stop the Chatbot

- Press `Ctrl + C` in the Command Prompt
- Or close the Command Prompt window

---

## 6️⃣ Testing Different Models

### Step 6.1: View Available Models

```bash
python -c "from config import Config; Config.print_available_models()"
```

**Output shows:**
```
Available Models for Testing
============================================================

📦 Embedding Models:
------------------------------------------------------------
  minilm                    💻 Local
    Provider: sentence-transformers
    Model: all-MiniLM-L6-v2
    Dimension: 384

  bge-small                 💻 Local
    Provider: sentence-transformers
    Model: BAAI/bge-small-en-v1.5
    Dimension: 384
    
  openai-small              🔑 API Key
    Provider: openai
    Model: text-embedding-3-small
    Dimension: 1536

🤖 LLM Models:
------------------------------------------------------------
  llama3.2                  💻 Local
    Provider: ollama
    Model: llama3.2
    
  gpt-4o-mini               🔑 API Key
    Provider: openai
    Model: gpt-4o-mini
```

### Step 6.2: Test Individual Models

**Test Embedding Model:**
```python
# Create test file: test_embedding.py
from model_tester import ModelTester

tester = ModelTester()

# Test different embeddings
tester.test_embedding_model("minilm")
tester.test_embedding_model("bge-small")
tester.test_embedding_model("openai-small")
```

Run:
```bash
python test_embedding.py
```

**Test LLM Model:**
```python
# Create test file: test_llm.py
from model_tester import ModelTester

tester = ModelTester()

# Test different LLMs
tester.test_llm_model("llama3.2")
tester.test_llm_model("gpt-4o-mini")
```

Run:
```bash
python test_llm.py
```

### Step 6.3: Test Model Combinations

**Quick Test Script:**
```python
# Create file: test_combinations.py
from model_tester import ModelTester

tester = ModelTester()

# Define test queries
test_queries = [
    "What are the prerequisites?",
    "Who teaches this course?",
    "Explain the topics covered"
]

# Test combination
result = tester.test_model_combination(
    embedding_key="bge-small",
    llm_key="gpt-4o-mini",
    persist_dir="faiss_course_db"
)

print(f"\nAverage query time: {result['metrics']['avg_query_time']:.2f}s")
```

Run:
```bash
python test_combinations.py
```

### Step 6.4: Comprehensive Testing

**Test all combinations:**
```python
# Create file: test_all.py
from model_tester import ModelTester

tester = ModelTester()

# Models to test
embeddings = ["minilm", "bge-small", "openai-small"]
llms = ["llama3.2", "gpt-4o-mini"]

# Run comprehensive tests
results = tester.run_comprehensive_test(
    embedding_keys=embeddings,
    llm_keys=llms,
    test_combinations=True,  # Test all combinations
    persist_dir="faiss_course_db"
)

print("\nResults saved to test_results/")
```

Run:
```bash
python test_all.py
```

**This creates:**
- `test_results/model_test_results_TIMESTAMP.json`
- Contains detailed metrics for each combination
- Shows performance rankings

### Step 6.5: Change Model in Config

**Method 1: Edit .env file**
```properties
# Change these lines
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

Then rebuild database:
```bash
python enhanced_ingestion.py
```

**Method 2: Programmatically**
```python
from config import Config

# Set models
Config.set_model_config(
    embedding_key="bge-small",
    llm_key="llama3.2"
)

# No need to rebuild - will use new models
```

---

## 7️⃣ Troubleshooting

### Problem: "Python is not recognized"

**Solution:**
1. Reinstall Python from [python.org](https://www.python.org)
2. ✅ Check "Add Python to PATH"
3. Restart Command Prompt

### Problem: "ModuleNotFoundError"

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements_windows.txt
```

### Problem: "Ollama server not responding"

**Solution:**
1. Install Ollama: [ollama.ai](https://ollama.ai)
2. Open Command Prompt as Administrator
3. Run: `ollama serve`
4. Keep this window open
5. In another window, run your script

### Problem: "CUDA not available" (with NVIDIA GPU)

**Solution:**
1. Install CUDA Toolkit: [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)
2. Install GPU PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
3. Restart Command Prompt

### Problem: "Out of memory"

**Solution for Embeddings:**
```properties
# In .env file
EMBEDDING_BATCH_SIZE=16  # Reduce from 32
CHUNK_SIZE=400           # Reduce from 500
```

**Solution for LLM:**
- Use smaller models: `phi3` instead of `llama3.2`
- Or use OpenAI API (no local memory needed)

### Problem: Slow performance

**Solutions:**

**For Embeddings:**
- Use GPU (install CUDA)
- Use smaller models: `minilm` instead of `bge-base`
- Reduce batch size in `.env`

**For LLM:**
- Use API-based models (OpenAI)
- Use smaller local models (`phi3`)
- Reduce `LLM_MAX_TOKENS` in `.env`

### Problem: "course_outlines.jsonl not found"

**Solution:**
1. Check `Course_Outline_Extractor.py` has correct path
2. Run: `python Course_Outline_Extractor.py`
3. Verify file created: `dir course_outlines.jsonl`

### Problem: Chatbot shows errors

**Solution:**
```bash
# Rebuild database
python enhanced_ingestion.py

# Then restart chatbot
streamlit run enhanced_chatbot_metrics.py
```

---

## 📊 Quick Reference Commands

```bash
# Setup
python setup_windows.py

# Extract documents
python Course_Outline_Extractor.py

# Build database
python enhanced_ingestion.py

# Run chatbot
streamlit run enhanced_chatbot_metrics.py

# Test models
python model_tester.py

# View available models
python -c "from config import Config; Config.print_available_models()"

# Check configuration
python -c "from config import Config; Config.validate()"
```

---

## 🎯 Recommended Model Combinations

### Best Performance (Fast)
```properties
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```
**Cost:** ~$0.01 per 100 queries

### Best Quality
```properties
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```
**Cost:** ~$0.05 per 100 queries

### Best Free (Local)
```properties
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```
**Cost:** Free (uses your CPU/GPU)

### Balanced (Local Embeddings + API LLM)
```properties
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```
**Cost:** ~$0.005 per 100 queries

---

## 🎓 Next Steps

1. **Start with OpenAI** for best results (if you have API key)
2. **Try local models** to compare (free)
3. **Test different combinations** using `model_tester.py`
4. **Compare results** in `test_results/` folder
5. **Choose your optimal setup** based on speed/cost/quality

---

## 📞 Need Help?

1. Check troubleshooting section above
2. Run diagnostics: `python setup_windows.py`
3. Check logs in `chatbot_metrics/` folder
4. Review test results in `test_results/` folder

---

**🎉 You're all set! Start asking questions about your courses!**
