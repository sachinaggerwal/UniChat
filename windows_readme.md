# RAG System - Windows Setup Guide

Complete guide for setting up and testing the RAG system on Windows.

## 🎯 Quick Start

### 1. Prerequisites

- **Windows 10/11** (64-bit)
- **Python 3.8+** (Download from [python.org](https://www.python.org/downloads/))
- **10+ GB free disk space** (for models and data)
- **OpenAI API Key** (optional, for best performance)
- **Ollama** (optional, for local LLMs) - Download from [ollama.ai](https://ollama.ai)

### 2. Installation

```bash
# Clone/download the project
cd your-project-folder

# Run automated setup
python setup_windows.py

# This will:
# - Check system requirements
# - Install all dependencies
# - Create necessary directories
# - Generate .env template
# - Test the installation
```

### 3. Configure Models

Edit the `.env` file to choose your models:

#### Option 1: OpenAI (Recommended - Fast & Accurate)
```properties
OPENAI_API_KEY=sk-your-key-here
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

#### Option 2: Local Models (Free - No API Key Needed)
```properties
# Requires Ollama installed and running
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

#### Option 3: HuggingFace (Free - CPU/GPU)
```properties
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
```

### 4. Process Your Documents

```bash
# Extract text from course PDFs/DOCX files
python Course_Outline_Extractor.py

# Build vector database and knowledge graph
python enhanced_ingestion.py
```

### 5. Run the Chatbot

```bash
streamlit run enhanced_chatbot_metrics.py
```

Open browser at: `http://localhost:8501`

---

## 🧪 Testing Different Models

### Automated Testing

The `model_tester.py` script allows you to systematically test different model combinations:

```bash
python model_tester.py
```

### Manual Testing

Use the config system to easily switch models:

```python
from config import Config

# Set models using predefined keys
Config.set_model_config(
    embedding_key="bge-small",  # or "minilm", "openai-small"
    llm_key="llama3.2"          # or "gpt-4o-mini"
)

# Then run your code
```

### Available Models

#### Embedding Models

| Model Key | Provider | Model Name | Dimension | Speed | Quality |
|-----------|----------|------------|-----------|-------|---------|
| `minilm` | HuggingFace | all-MiniLM-L6-v2 | 384 | ⚡ Fast | 🔹 Good |
| `bge-small` | HuggingFace | BAAI/bge-small-en-v1.5 | 384 | ⚡ Fast | ⭐ Better |
| `bge-base` | HuggingFace | BAAI/bge-base-en-v1.5 | 768 | 🐌 Medium | ⭐⭐ Best |
| `e5-small` | HuggingFace | intfloat/e5-small-v2 | 384 | ⚡ Fast | ⭐ Better |
| `nomic-embed` | Ollama | nomic-embed-text | 768 | 🐌 Medium | ⭐ Better |
| `openai-small` | OpenAI | text-embedding-3-small | 1536 | ⚡ Fast | ⭐⭐⭐ Excellent |

#### LLM Models

| Model Key | Provider | Model Name | Speed | Quality | Cost |
|-----------|----------|------------|-------|---------|------|
| `llama3.2` | Ollama | llama3.2 | 🐌 Slow | ⭐ Good | Free |
| `mistral` | Ollama | mistral | 🐌 Slow | ⭐ Good | Free |
| `phi3` | Ollama | phi3 | ⚡ Fast | 🔹 Moderate | Free |
| `gpt-4o-mini` | OpenAI | gpt-4o-mini | ⚡⚡ Very Fast | ⭐⭐⭐ Excellent | Low |
| `gpt-4o` | OpenAI | gpt-4o | ⚡ Fast | ⭐⭐⭐⭐ Best | Medium |

---

## 📊 Testing Framework

### Run Comprehensive Tests

```python
from model_tester import ModelTester

tester = ModelTester()

# Test multiple embedding models
embedding_models = ["minilm", "bge-small", "openai-small"]
llm_models = ["llama3.2", "gpt-4o-mini"]

results = tester.run_comprehensive_test(
    embedding_keys=embedding_models,
    llm_keys=llm_models,
    test_combinations=True,  # Test all combinations
    persist_dir="faiss_course_db"
)
```

### Test Results

Results are saved to `test_results/model_test_results_TIMESTAMP.json` with:

- **Embedding metrics**: Creation time, embedding time, dimension
- **LLM metrics**: Generation time, tokens/second, response quality
- **Combination metrics**: End-to-end query time, retrieval + generation breakdown
- **Comparison tables**: Ranked by performance

---

## 🎛️ Configuration Options

### Environment Variables

```properties
# Retrieval Settings
RETRIEVAL_TOP_K=6              # Number of results to retrieve
CHUNK_SIZE=500                 # Text chunk size
CHUNK_OVERLAP=150              # Overlap between chunks

# Knowledge Graph
USE_NETWORKX_KG=true           # Enable knowledge graph
KG_MAX_HOPS=1                  # Graph traversal depth
KG_TOP_ENTITIES=3              # Max entities per query

# Hybrid Retrieval
VECTOR_WEIGHT=0.7              # Vector database weight
KG_WEIGHT=0.3                  # Knowledge graph weight
FUSION_STRATEGY=adaptive       # "adaptive", "weighted", or "rrf"

# Performance
ENABLE_CACHING=true
EMBEDDING_BATCH_SIZE=32
NUM_WORKERS=4
```

### Programmatic Configuration

```python
from config import Config

# Override settings
Config.RETRIEVAL_TOP_K = 10
Config.VECTOR_WEIGHT = 0.8
Config.KG_WEIGHT = 0.2
Config.FUSION_STRATEGY = "weighted"
```

---

## 🚀 Performance Tips

### For Best Speed (Windows)

1. **Use GPU if available**:
   ```bash
   # Check if CUDA is available
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install GPU version of PyTorch
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Use smaller, faster models**:
   - Embedding: `minilm` or `openai-small`
   - LLM: `gpt-4o-mini` or `phi3`

3. **Reduce chunk size**:
   ```properties
   CHUNK_SIZE=400
   CHUNK_OVERLAP=100
   ```

### For Best Quality

1. **Use larger, more accurate models**:
   - Embedding: `bge-base` or `openai-large`
   - LLM: `gpt-4o` or `llama3.1`

2. **Increase retrieval results**:
   ```properties
   RETRIEVAL_TOP_K=10
   ```

3. **Use adaptive fusion**:
   ```properties
   FUSION_STRATEGY=adaptive
   ```

---

## 🐛 Troubleshooting

### Issue: "Ollama server not responding"

**Solution**:
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Open Command Prompt and run:
   ```bash
   ollama serve
   ```
3. In another terminal, pull models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

### Issue: "CUDA not available" but you have NVIDIA GPU

**Solution**:
1. Install NVIDIA CUDA Toolkit from [developer.nvidia.com](https://developer.nvidia.com/cuda-downloads)
2. Install GPU version of PyTorch:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: "Out of memory" errors

**Solution**:
1. Reduce batch size:
   ```properties
   EMBEDDING_BATCH_SIZE=16
   ```
2. Use smaller models (minilm instead of bge-base)
3. Close other applications

### Issue: Slow embedding generation

**Solution**:
1. Enable GPU (see above)
2. Use smaller embedding models
3. Reduce number of workers:
   ```properties
   NUM_WORKERS=2
   ```

---

## 📁 Project Structure

```
your-project/
├── config.py                      # Configuration management
├── model_factory.py               # Model creation
├── model_tester.py                # Testing framework
├── Course_Outline_Extractor.py    # Document processing
├── enhanced_ingestion.py          # Vector DB + KG building
├── enhanced_chatbot_metrics.py    # Streamlit chatbot
├── knowledge_graph.py             # Knowledge graph logic
├── hybrid_retriever.py            # Hybrid retrieval
├── .env                           # Configuration (create from setup)
├── requirements_windows.txt       # Dependencies
├── setup_windows.py               # Setup script
│
├── faiss_course_db/               # Vector database
├── knowledge_graph_db/            # Knowledge graph data
├── chatbot_metrics/               # Usage metrics
└── test_results/                  # Test results
```

---

## 🎓 Example Testing Workflow

```python
# 1. Test individual models
from model_tester import ModelTester

tester = ModelTester()

# Test embeddings
tester.test_embedding_model("minilm")
tester.test_embedding_model("bge-small")
tester.test_embedding_model("openai-small")

# Test LLMs
tester.test_llm_model("llama3.2")
tester.test_llm_model("gpt-4o-mini")

# 2. Test specific combination
result = tester.test_model_combination(
    embedding_key="bge-small",
    llm_key="gpt-4o-mini",
    persist_dir="faiss_course_db"
)

# 3. Compare all combinations
results = tester.run_comprehensive_test(
    embedding_keys=["minilm", "bge-small", "openai-small"],
    llm_keys=["llama3.2", "gpt-4o-mini"],
    test_combinations=True
)
```

---

## 📈 Metrics and Analysis

Test results include:

- **Speed metrics**: Creation time, embedding time, generation time
- **Quality metrics**: Response length, relevance scores
- **Resource metrics**: Memory usage, GPU utilization
- **Comparison rankings**: Models sorted by performance

Results are saved as JSON for further analysis with pandas, matplotlib, etc.

---

## 🔗 Useful Links

- [OpenAI API](https://platform.openai.com/api-keys)
- [Ollama](https://ollama.ai)
- [HuggingFace Models](https://huggingface.co/models)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

---

## 💡 Tips for Windows Users

1. **Use Windows Terminal** or PowerShell instead of CMD for better display
2. **Run as Administrator** if you encounter permission issues
3. **Add Python to PATH** during installation
4. **Use virtual environment** to avoid conflicts:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
5. **Check antivirus settings** - some may block model downloads

---

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Run `python setup_windows.py` to diagnose problems
3. Review test results in `test_results/` directory
4. Check logs in `chatbot_metrics/` directory
