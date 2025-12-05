"""
config.py - Windows-compatible configuration loader for the RAG system
Loads environment variables from .env file and provides configuration management
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class ModelConfig:
    """Available model configurations for easy testing"""
    
    EMBEDDING_MODELS = {
        # OpenAI Models
        "openai-small": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimension": 1536,
            "requires_api_key": True
        },
        "openai-large": {
            "provider": "openai",
            "model": "text-embedding-3-large",
            "dimension": 3072,
            "requires_api_key": True
        },
        
        # HuggingFace Models (Local)
        "minilm": {
            "provider": "sentence-transformers",
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "requires_api_key": False
        },
        "bge-small": {
            "provider": "sentence-transformers",
            "model": "BAAI/bge-small-en-v1.5",
            "dimension": 384,
            "requires_api_key": False
        },
        "bge-base": {
            "provider": "sentence-transformers",
            "model": "BAAI/bge-base-en-v1.5",
            "dimension": 768,
            "requires_api_key": False
        },
        "e5-small": {
            "provider": "sentence-transformers",
            "model": "intfloat/e5-small-v2",
            "dimension": 384,
            "requires_api_key": False
        },
        "e5-base": {
            "provider": "sentence-transformers",
            "model": "intfloat/e5-base-v2",
            "dimension": 768,
            "requires_api_key": False
        },
        
        # Ollama Models (Local)
        "nomic-embed": {
            "provider": "ollama",
            "model": "nomic-embed-text",
            "dimension": 768,
            "requires_api_key": False
        },
        "mxbai-embed": {
            "provider": "ollama",
            "model": "mxbai-embed-large",
            "dimension": 1024,
            "requires_api_key": False
        },
    }
    
    LLM_MODELS = {
        # OpenAI Models
        "gpt-4o-mini": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "requires_api_key": True
        },
        "gpt-4o": {
            "provider": "openai",
            "model": "gpt-4o",
            "requires_api_key": True
        },
        "gpt-3.5-turbo": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "requires_api_key": True
        },
        
        # Ollama Models (Local)
        "llama3.2": {
            "provider": "ollama",
            "model": "llama3.2",
            "requires_api_key": False
        },
        "llama3.1": {
            "provider": "ollama",
            "model": "llama3.1",
            "requires_api_key": False
        },
        "mistral": {
            "provider": "ollama",
            "model": "mistral",
            "requires_api_key": False
        },
        "phi3": {
            "provider": "ollama",
            "model": "phi3",
            "requires_api_key": False
        },
    }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, List[str]]:
        """List all available models"""
        return {
            "embeddings": list(cls.EMBEDDING_MODELS.keys()),
            "llms": list(cls.LLM_MODELS.keys())
        }
    
    @classmethod
    def get_embedding_config(cls, model_key: str) -> Dict[str, Any]:
        """Get embedding model configuration"""
        return cls.EMBEDDING_MODELS.get(model_key, {})
    
    @classmethod
    def get_llm_config(cls, model_key: str) -> Dict[str, Any]:
        """Get LLM model configuration"""
        return cls.LLM_MODELS.get(model_key, {})


class Config:
    """
    Configuration manager that reads from .env file.
    Windows-compatible with pathlib for cross-platform support.
    """

    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

    # Embedding Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # LLM Configuration
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1500"))
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Knowledge Graph Configuration
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")
    USE_NETWORKX_KG: bool = os.getenv("USE_NETWORKX_KG", "true").lower() == "true"
    KG_MAX_HOPS: int = int(os.getenv("KG_MAX_HOPS", "2"))
    KG_TOP_ENTITIES: int = int(os.getenv("KG_TOP_ENTITIES", "5"))

    # Hybrid Retrieval Settings
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.6"))
    KG_WEIGHT: float = float(os.getenv("KG_WEIGHT", "0.4"))
    FUSION_STRATEGY: str = os.getenv("FUSION_STRATEGY", "adaptive")

    # Storage Paths (Windows-compatible using pathlib)
    PERSIST_DIR: str = os.getenv("PERSIST_DIR", "faiss_course_db")
    KG_PERSIST_DIR: str = os.getenv("KG_PERSIST_DIR", "knowledge_graph_db")
    COURSE_OUTLINES_FILE: str = os.getenv(
        "COURSE_OUTLINES_FILE", "course_outlines.jsonl"
    )

    # Performance Settings
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", "4"))

    @classmethod
    def set_model_config(cls, embedding_key: str = None, llm_key: str = None):
        """
        Set model configuration using predefined model keys
        
        Args:
            embedding_key: Key from ModelConfig.EMBEDDING_MODELS
            llm_key: Key from ModelConfig.LLM_MODELS
        """
        if embedding_key:
            config = ModelConfig.get_embedding_config(embedding_key)
            if config:
                cls.EMBEDDING_PROVIDER = config["provider"]
                cls.EMBEDDING_MODEL = config["model"]
                print(f"✓ Embedding set to: {embedding_key} ({config['model']})")
            else:
                print(f"⚠️  Unknown embedding key: {embedding_key}")
        
        if llm_key:
            config = ModelConfig.get_llm_config(llm_key)
            if config:
                cls.LLM_PROVIDER = config["provider"]
                cls.LLM_MODEL = config["model"]
                print(f"✓ LLM set to: {llm_key} ({config['model']})")
            else:
                print(f"⚠️  Unknown LLM key: {llm_key}")

    @classmethod
    def get_windows_safe_path(cls, path_str: str) -> Path:
        """
        Convert path string to Windows-safe Path object
        
        Args:
            path_str: Path as string
            
        Returns:
            pathlib.Path object
        """
        return Path(path_str).resolve()

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist (Windows-compatible)"""
        directories = [
            cls.PERSIST_DIR,
            cls.KG_PERSIST_DIR,
            "chatbot_metrics",
            "test_results"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        errors = []

        # Check API keys based on providers
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for OpenAI embeddings")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for OpenAI LLM")
        if cls.LLM_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required for Anthropic LLM")
        if cls.EMBEDDING_PROVIDER == "cohere" and not cls.COHERE_API_KEY:
            errors.append("COHERE_API_KEY is required for Cohere embeddings")

        # Validate weights sum to 1
        weight_sum = cls.VECTOR_WEIGHT + cls.KG_WEIGHT
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(
                f"VECTOR_WEIGHT + KG_WEIGHT must equal 1.0 (current: {weight_sum})"
            )

        # Validate Ollama connection if using Ollama
        if cls.EMBEDDING_PROVIDER == "ollama" or cls.LLM_PROVIDER == "ollama":
            try:
                import requests
                response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=2)
                if response.status_code != 200:
                    errors.append(
                        f"Ollama server not responding at {cls.OLLAMA_BASE_URL}"
                    )
            except Exception as e:
                errors.append(f"Cannot connect to Ollama: {e}")

        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  ✗ {error}")
            return False

        return True

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "Embedding": f"{cls.EMBEDDING_PROVIDER}/{cls.EMBEDDING_MODEL}",
            "LLM": f"{cls.LLM_PROVIDER}/{cls.LLM_MODEL}",
            "Vector Weight": cls.VECTOR_WEIGHT,
            "KG Weight": cls.KG_WEIGHT,
            "Fusion Strategy": cls.FUSION_STRATEGY,
            "Chunk Size": cls.CHUNK_SIZE,
            "Top K": cls.RETRIEVAL_TOP_K,
            "Platform": "Windows" if os.name == "nt" else "Unix-like",
        }

    @classmethod
    def print_summary(cls):
        """Print configuration summary"""
        print("\n" + "=" * 60)
        print("RAG System Configuration")
        print("=" * 60)
        for key, value in cls.get_summary().items():
            print(f"{key:.<30} {value}")
        print("=" * 60 + "\n")

    @classmethod
    def print_available_models(cls):
        """Print all available models for testing"""
        print("\n" + "=" * 60)
        print("Available Models for Testing")
        print("=" * 60)
        
        print("\n📦 Embedding Models:")
        print("-" * 60)
        for key, config in ModelConfig.EMBEDDING_MODELS.items():
            api_status = "🔑 API Key" if config["requires_api_key"] else "💻 Local"
            print(f"  {key:.<25} {api_status}")
            print(f"    Provider: {config['provider']}")
            print(f"    Model: {config['model']}")
            print(f"    Dimension: {config['dimension']}")
            print()
        
        print("🤖 LLM Models:")
        print("-" * 60)
        for key, config in ModelConfig.LLM_MODELS.items():
            api_status = "🔑 API Key" if config["requires_api_key"] else "💻 Local"
            print(f"  {key:.<25} {api_status}")
            print(f"    Provider: {config['provider']}")
            print(f"    Model: {config['model']}")
            print()
        
        print("=" * 60 + "\n")


# Ensure directories exist on import
Config.ensure_directories()

# Validate configuration on import
if not Config.validate():
    print("\n⚠️  Please fix configuration errors in .env file\n")
