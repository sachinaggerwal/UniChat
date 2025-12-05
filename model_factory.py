"""
model_factory.py - Enhanced factory for creating embeddings and LLMs
Windows-compatible with improved error handling and model support
Supports: OpenAI, Sentence Transformers, Ollama, Anthropic, Cohere
"""

import os
import warnings
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config

# Suppress warnings for cleaner output on Windows
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ModelFactory:
    """Factory for creating embedding and LLM instances with multi-provider support"""

    @staticmethod
    def create_embeddings():
        """
        Create embeddings based on configuration
        Windows-compatible with proper error handling
        """
        provider = Config.EMBEDDING_PROVIDER.lower()

        try:
            if provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                
                if not Config.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY is required but not set")
                
                return OpenAIEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    openai_api_key=Config.OPENAI_API_KEY,
                    show_progress_bar=False  # Cleaner Windows console output
                )

            elif provider in ["sentence-transformers", "huggingface"]:
                from langchain_huggingface import HuggingFaceEmbeddings
                
                # Check for GPU on Windows
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                        print(f"✓ Using CUDA GPU for embeddings")
                    else:
                        device = "cpu"
                        print(f"ℹ Using CPU for embeddings (CUDA not available)")
                except ImportError:
                    device = "cpu"
                    print(f"ℹ Using CPU for embeddings (PyTorch not installed)")

                # Set cache directory for Windows
                cache_folder = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "huggingface",
                    "hub"
                )
                os.makedirs(cache_folder, exist_ok=True)

                return HuggingFaceEmbeddings(
                    model_name=Config.EMBEDDING_MODEL,
                    model_kwargs={
                        "device": device,
                        "trust_remote_code": True
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                        "batch_size": Config.EMBEDDING_BATCH_SIZE,
                        # "show_progress_bar": False  # Disable progress bar
                    },
                    cache_folder=cache_folder
                )

            elif provider == "cohere":
                from langchain_cohere import CohereEmbeddings
                
                if not Config.COHERE_API_KEY:
                    raise ValueError("COHERE_API_KEY is required but not set")
                
                return CohereEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    cohere_api_key=Config.COHERE_API_KEY
                )

            elif provider == "ollama":
                from langchain_community.embeddings import OllamaEmbeddings
                
                # Test Ollama connection
                try:
                    import requests
                    response = requests.get(
                        f"{Config.OLLAMA_BASE_URL}/api/tags",
                        timeout=5
                    )
                    if response.status_code != 200:
                        raise ConnectionError(
                            f"Ollama server not responding at {Config.OLLAMA_BASE_URL}"
                        )
                except requests.exceptions.RequestException as e:
                    raise ConnectionError(
                        f"Cannot connect to Ollama at {Config.OLLAMA_BASE_URL}. "
                        f"Make sure Ollama is running. Error: {e}"
                    )
                
                return OllamaEmbeddings(
                    model=Config.EMBEDDING_MODEL,
                    base_url=Config.OLLAMA_BASE_URL
                )

            else:
                raise ValueError(
                    f"Unsupported embedding provider: {provider}. "
                    f"Supported: openai, sentence-transformers, cohere, ollama"
                )
                
        except ImportError as e:
            raise ImportError(
                f"Required package not installed for {provider}. "
                f"Error: {e}\n"
                f"Install with: pip install -r requirements.txt"
            )
        except Exception as e:
            raise Exception(f"Failed to create embeddings with {provider}: {e}")

    @staticmethod
    def create_llm():
        """
        Create LLM based on configuration
        Windows-compatible with proper error handling
        """
        provider = Config.LLM_PROVIDER.lower()

        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                
                if not Config.OPENAI_API_KEY:
                    raise ValueError("OPENAI_API_KEY is required but not set")
                
                return ChatOpenAI(
                    model=Config.LLM_MODEL,
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.LLM_MAX_TOKENS,
                    openai_api_key=Config.OPENAI_API_KEY,
                    request_timeout=60  # Timeout for Windows
                )

            elif provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                
                if not Config.ANTHROPIC_API_KEY:
                    raise ValueError("ANTHROPIC_API_KEY is required but not set")
                
                return ChatAnthropic(
                    model=Config.LLM_MODEL,
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.LLM_MAX_TOKENS,
                    anthropic_api_key=Config.ANTHROPIC_API_KEY,
                    timeout=60
                )

            elif provider == "ollama":
                from langchain_community.llms import Ollama
                
                # Test Ollama connection
                try:
                    import requests
                    response = requests.get(
                        f"{Config.OLLAMA_BASE_URL}/api/tags",
                        timeout=5
                    )
                    if response.status_code != 200:
                        raise ConnectionError(
                            f"Ollama server not responding at {Config.OLLAMA_BASE_URL}"
                        )
                except requests.exceptions.RequestException as e:
                    raise ConnectionError(
                        f"Cannot connect to Ollama at {Config.OLLAMA_BASE_URL}. "
                        f"Make sure Ollama is running. Error: {e}"
                    )

                # Wrapper for consistent interface
                class OllamaWrapper:
                    def __init__(self, base_llm):
                        self.llm = base_llm

                    def invoke(self, messages):
                        # Extract prompt from messages
                        if isinstance(messages, list):
                            prompt = "\n".join([
                                m.content if hasattr(m, "content") else str(m)
                                for m in messages
                            ])
                        else:
                            prompt = str(messages)

                        # Get response
                        try:
                            response = self.llm.invoke(prompt)
                        except Exception as e:
                            raise Exception(f"Ollama generation failed: {e}")

                        # Return with content attribute
                        class Response:
                            def __init__(self, text):
                                self.content = text
                        
                        return Response(response)

                base_llm = Ollama(
                    model=Config.LLM_MODEL,
                    base_url=Config.OLLAMA_BASE_URL,
                    temperature=Config.LLM_TEMPERATURE,
                    num_predict=Config.LLM_MAX_TOKENS,
                    num_ctx=2048,
                    timeout=120  # Longer timeout for Windows
                )

                return OllamaWrapper(base_llm)

            elif provider == "cohere":
                from langchain_cohere import ChatCohere
                
                if not Config.COHERE_API_KEY:
                    raise ValueError("COHERE_API_KEY is required but not set")
                
                return ChatCohere(
                    model=Config.LLM_MODEL,
                    temperature=Config.LLM_TEMPERATURE,
                    cohere_api_key=Config.COHERE_API_KEY
                )

            else:
                raise ValueError(
                    f"Unsupported LLM provider: {provider}. "
                    f"Supported: openai, anthropic, ollama, cohere"
                )
                
        except ImportError as e:
            raise ImportError(
                f"Required package not installed for {provider}. "
                f"Error: {e}\n"
                f"Install with: pip install -r requirements.txt"
            )
        except Exception as e:
            raise Exception(f"Failed to create LLM with {provider}: {e}")

    @staticmethod
    def get_text_splitter():
        """
        Get configured text splitter
        Windows-compatible
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    @staticmethod
    def test_models():
        """
        Test if models can be loaded successfully
        Useful for Windows setup validation
        """
        print("\n" + "="*60)
        print("Testing Model Configuration")
        print("="*60)
        
        # Test embeddings
        print(f"\n1️⃣ Testing Embeddings: {Config.EMBEDDING_PROVIDER}/{Config.EMBEDDING_MODEL}")
        try:
            embeddings = ModelFactory.create_embeddings()
            # Test with a simple query
            test_text = ["This is a test sentence."]
            result = embeddings.embed_documents(test_text)
            print(f"   ✓ Success! Dimension: {len(result[0])}")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
        
        # Test LLM
        print(f"\n2️⃣ Testing LLM: {Config.LLM_PROVIDER}/{Config.LLM_MODEL}")
        try:
            llm = ModelFactory.create_llm()
            # Test with a simple query
            response = llm.invoke("Say 'test successful' and nothing else.")
            response_text = response.content if hasattr(response, "content") else str(response)
            print(f"   ✓ Success! Response length: {len(response_text)} chars")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
        
        print("\n" + "="*60)
        print("✓ All models loaded successfully!")
        print("="*60 + "\n")
        return True


# Windows-specific helper functions
def check_windows_prerequisites():
    """Check if Windows system has necessary prerequisites"""
    import platform
    
    if platform.system() != "Windows":
        return True
    
    print("\n" + "="*60)
    print("Windows System Check")
    print("="*60)
    
    issues = []
    
    # Check Python version
    import sys
    python_version = sys.version_info
    if python_version < (3, 8):
        issues.append(f"Python 3.8+ required. Current: {python_version.major}.{python_version.minor}")
    else:
        print(f"✓ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check CUDA availability (optional)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available (CPU mode will be used)")
    except ImportError:
        print("ℹ PyTorch not installed (CPU mode will be used)")
    
    # Check disk space
    import shutil
    stats = shutil.disk_usage(os.path.expanduser("~"))
    free_gb = stats.free / (1024**3)
    if free_gb < 5:
        issues.append(f"Low disk space: {free_gb:.1f} GB free (recommend 5+ GB)")
    else:
        print(f"✓ Disk space: {free_gb:.1f} GB free")
    
    print("="*60 + "\n")
    
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True


if __name__ == "__main__":
    # Run Windows prerequisite check
    check_windows_prerequisites()
    
    # Test current configuration
    ModelFactory.test_models()
