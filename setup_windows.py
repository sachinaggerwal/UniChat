"""
setup_windows.py - Automated setup script for Windows
Checks system requirements, installs dependencies, and configures the RAG system
Run: python setup_windows.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class WindowsSetup:
    """Setup wizard for Windows systems"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
    
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*70)
        print(text)
        print("="*70)
    
    def check_python_version(self):
        """Check Python version"""
        self.print_header("Checking Python Version")
        
        version = sys.version_info
        if version < (3, 8):
            self.issues.append(
                f"Python 3.8+ required. Current: {version.major}.{version.minor}"
            )
            print(f"✗ Python {version.major}.{version.minor} (Need 3.8+)")
            return False
        else:
            print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
            return True
    
    def check_platform(self):
        """Verify Windows platform"""
        self.print_header("Checking Platform")
        
        if platform.system() != "Windows":
            self.warnings.append("This script is optimized for Windows")
            print(f"⚠️  Detected: {platform.system()}")
        else:
            print(f"✓ Windows {platform.release()}")
    
    def check_pip(self):
        """Check if pip is available"""
        self.print_header("Checking pip")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError:
            self.issues.append("pip not found")
            print("✗ pip not available")
            return False
    
    def check_disk_space(self):
        """Check available disk space"""
        self.print_header("Checking Disk Space")
        
        import shutil
        stats = shutil.disk_usage(Path.home())
        free_gb = stats.free / (1024**3)
        
        if free_gb < 5:
            self.warnings.append(
                f"Low disk space: {free_gb:.1f} GB (recommend 10+ GB for models)"
            )
            print(f"⚠️  Free space: {free_gb:.1f} GB (Low)")
        else:
            print(f"✓ Free space: {free_gb:.1f} GB")
    
    def check_cuda(self):
        """Check CUDA availability"""
        self.print_header("Checking CUDA (Optional)")
        
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"✓ CUDA available: {device_name}")
                print(f"  CUDA version: {torch.version.cuda}")
                return True
            else:
                print("ℹ CUDA not available (will use CPU)")
                self.warnings.append(
                    "No CUDA detected. GPU acceleration not available. "
                    "Install CUDA toolkit for faster processing."
                )
                return False
        except ImportError:
            print("ℹ PyTorch not installed yet")
            return False
    
    def install_requirements(self):
        """Install required packages"""
        self.print_header("Installing Requirements")
        
        req_file = Path("requirements.txt")
        if not req_file.exists():
            print("✗ requirements_windows.txt not found")
            self.issues.append("requirements_windows.txt missing")
            return False
        
        print("Installing packages from requirements_windows.txt...")
        print("This may take several minutes...\n")
        
        try:
            # Upgrade pip first
            print("1/2 Upgrading pip...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True
            )
            
            # Install requirements
            print("\n2/2 Installing packages...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                check=True
            )
            
            print("\n✓ All packages installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.issues.append(f"Package installation failed: {e}")
            print(f"\n✗ Installation failed: {e}")
            return False
    
    def check_ollama(self):
        """Check if Ollama is installed (optional)"""
        self.print_header("Checking Ollama (Optional)")
        
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✓ Ollama server is running")
                models = response.json().get("models", [])
                if models:
                    print(f"  Available models: {', '.join([m['name'] for m in models[:5]])}")
                return True
            else:
                print("⚠️  Ollama server not responding")
                self.warnings.append(
                    "Ollama not running. Install from https://ollama.ai to use local models"
                )
                return False
        except Exception:
            print("ℹ Ollama not installed")
            print("  Install from: https://ollama.ai")
            print("  Then run: ollama pull llama3.2")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        self.print_header("Creating Directories")
        
        directories = [
            "faiss_course_db",
            "knowledge_graph_db",
            "chatbot_metrics",
            "test_results"
        ]
        
        for dir_name in directories:
            dir_path = Path(dir_name)
            dir_path.mkdir(exist_ok=True)
            print(f"✓ {dir_name}/")
    
    def check_env_file(self):
        """Check .env file"""
        self.print_header("Checking Configuration")
        
        env_file = Path(".env")
        if not env_file.exists():
            print("⚠️  .env file not found")
            print("Creating template .env file...")
            
            template = """# OpenAI Configuration (RECOMMENDED for best performance)
OPENAI_API_KEY=your_openai_api_key_here

# Embedding Configuration
EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=1500
OLLAMA_BASE_URL=http://localhost:11434

# Retrieval Configuration
RETRIEVAL_TOP_K=6
CHUNK_SIZE=500
CHUNK_OVERLAP=150

# Knowledge Graph Configuration
USE_NETWORKX_KG=true
KG_MAX_HOPS=1
KG_TOP_ENTITIES=3

# Hybrid Retrieval
VECTOR_WEIGHT=0.7
KG_WEIGHT=0.3
FUSION_STRATEGY=adaptive

# Storage
PERSIST_DIR=faiss_course_db
KG_PERSIST_DIR=knowledge_graph_db
COURSE_OUTLINES_FILE=course_outlines.jsonl

# Performance
ENABLE_CACHING=true
EMBEDDING_BATCH_SIZE=32
NUM_WORKERS=4
"""
            
            env_file.write_text(template)
            print("✓ Created .env template")
            print("  ⚠️  Please edit .env and add your API keys if using OpenAI")
        else:
            print("✓ .env file exists")
            
            # Check if API key is set
            from dotenv import load_dotenv
            load_dotenv()
            
            if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here":
                print("  ✓ OpenAI API key configured")
            else:
                print("  ℹ OpenAI API key not configured (using local models)")
    
    def test_installation(self):
        """Test if everything is working"""
        self.print_header("Testing Installation")
        
        print("Testing model loading...")
        
        try:
            from config import Config
            from model_factory import ModelFactory
            
            # Test embeddings
            print("\n1/2 Testing embeddings...")
            embeddings = ModelFactory.create_embeddings()
            test_result = embeddings.embed_documents(["test"])
            print(f"    ✓ Embeddings working (dimension: {len(test_result[0])})")
            
            # Test LLM
            print("\n2/2 Testing LLM...")
            llm = ModelFactory.create_llm()
            response = llm.invoke("Say 'hello'")
            print(f"    ✓ LLM working")
            
            return True
            
        except Exception as e:
            print(f"    ✗ Test failed: {e}")
            self.issues.append(f"Installation test failed: {e}")
            return False
    
    def print_summary(self):
        """Print setup summary"""
        self.print_header("Setup Summary")
        
        if not self.issues:
            print("✓ Setup completed successfully!")
            
            if self.warnings:
                print("\n⚠️  Warnings:")
                for warning in self.warnings:
                    print(f"  - {warning}")
            
            print("\n" + "="*70)
            print("Next Steps:")
            print("="*70)
            print("\n1. Extract course outlines:")
            print("   python Course_Outline_Extractor.py")
            print("\n2. Build the RAG system:")
            print("   python enhanced_ingestion.py")
            print("\n3. Run the chatbot:")
            print("   streamlit run enhanced_chatbot_metrics.py")
            print("\n4. Test different models:")
            print("   python model_tester.py")
            print("\n" + "="*70 + "\n")
        else:
            print("✗ Setup encountered issues:\n")
            for issue in self.issues:
                print(f"  - {issue}")
            print("\nPlease fix these issues and run setup again.")
    
    def run(self):
        """Run complete setup"""
        print("\n" + "="*70)
        print("Windows RAG System Setup")
        print("="*70)
        
        # Run all checks
        self.check_platform()
        self.check_python_version()
        self.check_pip()
        self.check_disk_space()
        
        if self.issues:
            self.print_summary()
            return False
        
        # Install requirements
        print("\nDo you want to install/upgrade required packages? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            self.install_requirements()
        else:
            print("Skipping package installation")
        
        # Additional checks
        self.check_cuda()
        self.check_ollama()
        self.create_directories()
        self.check_env_file()
        
        # Test if user wants to run tests
        if not self.issues:
            print("\nDo you want to test the installation? (y/n): ", end="")
            response = input().strip().lower()
            
            if response == 'y':
                self.test_installation()
        
        # Print summary
        self.print_summary()
        
        return len(self.issues) == 0


if __name__ == "__main__":
    setup = WindowsSetup()
    success = setup.run()
    sys.exit(0 if success else 1)
