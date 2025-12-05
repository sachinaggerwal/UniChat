"""
model_tester.py - Framework for testing different model combinations on Windows
Automatically tests various embedding and LLM combinations and records metrics
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import traceback

from config import Config, ModelConfig
from model_factory import ModelFactory
from langchain_community.vectorstores import FAISS


class ModelTester:
    """Framework for systematically testing different model combinations"""
    
    def __init__(self, test_queries: List[str] = None, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default test queries if none provided
        self.test_queries = test_queries or [
            "What are the prerequisites for BIT2001B?",
            "Who teaches this course?",
            "What is the grading scheme?",
            "What are the learning outcomes?",
            "Describe the course structure"
        ]
        
        self.results = []
    
    def test_embedding_model(
        self, 
        embedding_key: str, 
        sample_texts: List[str] = None
    ) -> Dict[str, Any]:
        """
        Test a single embedding model
        
        Args:
            embedding_key: Key from ModelConfig.EMBEDDING_MODELS
            sample_texts: Optional sample texts to embed
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print(f"Testing Embedding Model: {embedding_key}")
        print(f"{'='*60}")
        
        result = {
            "model_key": embedding_key,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "metrics": {}
        }
        
        try:
            # Get model config
            model_config = ModelConfig.get_embedding_config(embedding_key)
            if not model_config:
                result["error"] = f"Unknown embedding model key: {embedding_key}"
                return result
            
            # Set configuration
            Config.EMBEDDING_PROVIDER = model_config["provider"]
            Config.EMBEDDING_MODEL = model_config["model"]
            
            print(f"Provider: {model_config['provider']}")
            print(f"Model: {model_config['model']}")
            print(f"Dimension: {model_config['dimension']}")
            
            # Create embeddings
            start_time = time.time()
            embeddings = ModelFactory.create_embeddings()
            creation_time = time.time() - start_time
            
            # Test embedding generation
            sample_texts = sample_texts or [
                "This is a test sentence for embedding.",
                "Machine learning is a subset of artificial intelligence.",
                "Python is a popular programming language."
            ]
            
            embedding_start = time.time()
            embedded_samples = embeddings.embed_documents(sample_texts)
            embedding_time = time.time() - embedding_start
            
            # Calculate metrics
            result["success"] = True
            result["metrics"] = {
                "creation_time": creation_time,
                "embedding_time": embedding_time,
                "avg_time_per_doc": embedding_time / len(sample_texts),
                "dimension": len(embedded_samples[0]) if embedded_samples else 0,
                "num_samples": len(sample_texts)
            }
            
            print(f"✓ Success!")
            print(f"  Creation time: {creation_time:.3f}s")
            print(f"  Embedding time: {embedding_time:.3f}s")
            print(f"  Avg per doc: {result['metrics']['avg_time_per_doc']:.3f}s")
            print(f"  Actual dimension: {result['metrics']['dimension']}")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"✗ Error: {e}")
        
        return result
    
    def test_llm_model(
        self, 
        llm_key: str, 
        test_prompt: str = None
    ) -> Dict[str, Any]:
        """
        Test a single LLM model
        
        Args:
            llm_key: Key from ModelConfig.LLM_MODELS
            test_prompt: Optional test prompt
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print(f"Testing LLM Model: {llm_key}")
        print(f"{'='*60}")
        
        result = {
            "model_key": llm_key,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "metrics": {}
        }
        
        try:
            # Get model config
            model_config = ModelConfig.get_llm_config(llm_key)
            if not model_config:
                result["error"] = f"Unknown LLM model key: {llm_key}"
                return result
            
            # Set configuration
            Config.LLM_PROVIDER = model_config["provider"]
            Config.LLM_MODEL = model_config["model"]
            
            print(f"Provider: {model_config['provider']}")
            print(f"Model: {model_config['model']}")
            
            # Create LLM
            start_time = time.time()
            llm = ModelFactory.create_llm()
            creation_time = time.time() - start_time
            
            # Test generation
            test_prompt = test_prompt or "What is machine learning? Provide a brief answer."
            
            generation_start = time.time()
            response = llm.invoke(test_prompt)
            generation_time = time.time() - generation_start
            
            # Extract response text
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Calculate metrics
            result["success"] = True
            result["metrics"] = {
                "creation_time": creation_time,
                "generation_time": generation_time,
                "response_length": len(response_text),
                "tokens_per_second": len(response_text.split()) / generation_time if generation_time > 0 else 0
            }
            result["sample_response"] = response_text[:200]  # First 200 chars
            
            print(f"✓ Success!")
            print(f"  Creation time: {creation_time:.3f}s")
            print(f"  Generation time: {generation_time:.3f}s")
            print(f"  Response length: {result['metrics']['response_length']} chars")
            print(f"  Tokens/sec: {result['metrics']['tokens_per_second']:.2f}")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"✗ Error: {e}")
        
        return result
    
    def test_model_combination(
        self,
        embedding_key: str,
        llm_key: str,
        persist_dir: str = None
    ) -> Dict[str, Any]:
        """
        Test a complete embedding + LLM combination with actual RAG queries
        
        Args:
            embedding_key: Embedding model key
            llm_key: LLM model key
            persist_dir: Optional custom persist directory
            
        Returns:
            Dictionary with comprehensive test results
        """
        print(f"\n{'='*70}")
        print(f"Testing Combination: {embedding_key} + {llm_key}")
        print(f"{'='*70}")
        
        result = {
            "embedding_key": embedding_key,
            "llm_key": llm_key,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "error": None,
            "metrics": {}
        }
        
        try:
            # Set model configuration
            Config.set_model_config(embedding_key=embedding_key, llm_key=llm_key)
            
            # Create embeddings and LLM
            embeddings = ModelFactory.create_embeddings()
            llm = ModelFactory.create_llm()
            
            # Load or create vector store
            if persist_dir and Path(persist_dir).exists():
                vector_store = FAISS.load_local(
                    persist_dir,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"✓ Loaded vector store from {persist_dir}")
            else:
                print("⚠️  No vector store found. Skipping retrieval tests.")
                result["metrics"]["vector_store_available"] = False
                return result
            
            # Run test queries
            query_results = []
            total_time = 0
            
            for query in self.test_queries:
                print(f"\nQuery: {query}")
                
                query_start = time.time()
                
                # Retrieval
                docs = vector_store.similarity_search(query, k=3)
                retrieval_time = time.time() - query_start
                
                # Generation
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
                
                generation_start = time.time()
                response = llm.invoke(prompt)
                generation_time = time.time() - generation_start
                
                query_time = time.time() - query_start
                total_time += query_time
                
                response_text = response.content if hasattr(response, "content") else str(response)
                
                query_results.append({
                    "query": query,
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": query_time,
                    "response_length": len(response_text),
                    "num_docs": len(docs)
                })
                
                print(f"  ✓ Total: {query_time:.2f}s (Retrieval: {retrieval_time:.2f}s, Generation: {generation_time:.2f}s)")
            
            # Calculate aggregate metrics
            result["success"] = True
            result["metrics"] = {
                "num_queries": len(self.test_queries),
                "total_time": total_time,
                "avg_query_time": total_time / len(self.test_queries),
                "avg_retrieval_time": sum(q["retrieval_time"] for q in query_results) / len(query_results),
                "avg_generation_time": sum(q["generation_time"] for q in query_results) / len(query_results),
                "vector_store_available": True
            }
            result["query_results"] = query_results
            
            print(f"\n✓ Combination Test Complete!")
            print(f"  Avg query time: {result['metrics']['avg_query_time']:.2f}s")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"✗ Error: {e}")
        
        return result
    
    def run_comprehensive_test(
        self,
        embedding_keys: List[str] = None,
        llm_keys: List[str] = None,
        test_combinations: bool = False,
        persist_dir: str = None
    ):
        """
        Run comprehensive tests on multiple models
        
        Args:
            embedding_keys: List of embedding model keys to test
            llm_keys: List of LLM model keys to test
            test_combinations: Whether to test all combinations
            persist_dir: Vector store directory for combination tests
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"model_test_results_{timestamp}.json"
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "embedding_tests": [],
            "llm_tests": [],
            "combination_tests": []
        }
        
        # Test embedding models
        if embedding_keys:
            print("\n" + "="*70)
            print("TESTING EMBEDDING MODELS")
            print("="*70)
            
            for emb_key in embedding_keys:
                result = self.test_embedding_model(emb_key)
                all_results["embedding_tests"].append(result)
                time.sleep(1)  # Brief pause between tests
        
        # Test LLM models
        if llm_keys:
            print("\n" + "="*70)
            print("TESTING LLM MODELS")
            print("="*70)
            
            for llm_key in llm_keys:
                result = self.test_llm_model(llm_key)
                all_results["llm_tests"].append(result)
                time.sleep(1)
        
        # Test combinations
        if test_combinations and embedding_keys and llm_keys:
            print("\n" + "="*70)
            print("TESTING MODEL COMBINATIONS")
            print("="*70)
            
            for emb_key in embedding_keys:
                for llm_key in llm_keys:
                    result = self.test_model_combination(emb_key, llm_key, persist_dir)
                    all_results["combination_tests"].append(result)
                    time.sleep(2)
        
        # Save results
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"✓ Test results saved to: {results_file}")
        print(f"{'='*70}\n")
        
        # Print summary
        self.print_test_summary(all_results)
        
        return all_results
    
    def print_test_summary(self, results: Dict[str, Any]):
        """Print a summary of test results"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        # Embedding tests summary
        if results["embedding_tests"]:
            print("\n📦 Embedding Models:")
            successful = [r for r in results["embedding_tests"] if r["success"]]
            failed = [r for r in results["embedding_tests"] if not r["success"]]
            
            print(f"  ✓ Successful: {len(successful)}")
            print(f"  ✗ Failed: {len(failed)}")
            
            if successful:
                print("\n  Performance Ranking (by avg time per doc):")
                sorted_results = sorted(
                    successful,
                    key=lambda x: x["metrics"].get("avg_time_per_doc", float("inf"))
                )
                for i, r in enumerate(sorted_results[:5], 1):
                    avg_time = r["metrics"].get("avg_time_per_doc", 0)
                    print(f"    {i}. {r['model_key']}: {avg_time:.3f}s per doc")
        
        # LLM tests summary
        if results["llm_tests"]:
            print("\n🤖 LLM Models:")
            successful = [r for r in results["llm_tests"] if r["success"]]
            failed = [r for r in results["llm_tests"] if not r["success"]]
            
            print(f"  ✓ Successful: {len(successful)}")
            print(f"  ✗ Failed: {len(failed)}")
            
            if successful:
                print("\n  Performance Ranking (by generation time):")
                sorted_results = sorted(
                    successful,
                    key=lambda x: x["metrics"].get("generation_time", float("inf"))
                )
                for i, r in enumerate(sorted_results[:5], 1):
                    gen_time = r["metrics"].get("generation_time", 0)
                    print(f"    {i}. {r['model_key']}: {gen_time:.3f}s")
        
        # Combination tests summary
        if results["combination_tests"]:
            print("\n🔀 Model Combinations:")
            successful = [r for r in results["combination_tests"] if r["success"]]
            failed = [r for r in results["combination_tests"] if not r["success"]]
            
            print(f"  ✓ Successful: {len(successful)}")
            print(f"  ✗ Failed: {len(failed)}")
            
            if successful:
                print("\n  Best Combinations (by avg query time):")
                sorted_results = sorted(
                    successful,
                    key=lambda x: x["metrics"].get("avg_query_time", float("inf"))
                )
                for i, r in enumerate(sorted_results[:5], 1):
                    avg_time = r["metrics"].get("avg_query_time", 0)
                    combo = f"{r['embedding_key']} + {r['llm_key']}"
                    print(f"    {i}. {combo}: {avg_time:.2f}s per query")
        
        print("\n" + "="*70 + "\n")


# Example usage
if __name__ == "__main__":
    # Create tester
    tester = ModelTester()
    
    # Print available models
    Config.print_available_models()
    
    # Define models to test (start with local models that don't require API keys)
    embedding_models_to_test = [
        "minilm",      # Fast, local
        "bge-small",   # Better quality, local
        # "openai-small",  # Uncomment if you have API key
    ]
    
    llm_models_to_test = [
        "llama3.2",    # Local
        # "gpt-4o-mini",  # Uncomment if you have API key
    ]
    
    # Run tests
    results = tester.run_comprehensive_test(
        embedding_keys=embedding_models_to_test,
        llm_keys=llm_models_to_test,
        test_combinations=False,  # Set to True to test all combinations
        persist_dir=Config.PERSIST_DIR
    )
