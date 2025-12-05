"""
metrics_tracker.py - Comprehensive metrics tracking for RAG system
Tracks performance, accuracy, cost, and quality metrics
"""

import time
import json
import psutil
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import pandas as pd


@dataclass
class QueryMetrics:
    """Metrics for a single query"""

    timestamp: str
    query: str
    actual_answer: str

    # Timing metrics
    total_time: float
    retrieval_time: float
    generation_time: float

    # Retrieval metrics
    num_vector_results: int
    num_kg_results: int
    num_total_results: int
    fusion_strategy: str

    # Quality metrics
    answer_length: int
    context_length: int

    # Resource metrics
    memory_used_mb: float
    cpu_percent: float

    # Model info
    embedding_provider: str
    embedding_model: str
    llm_provider: str
    llm_model: str

    # Optional fields (with defaults must come last)
    expected_answer: Optional[str] = None
    relevance_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    course_filter: Optional[str] = None
    courses_referenced: List[str] = field(default_factory=list)


class MetricsTracker:
    """Track and log RAG system metrics"""

    def __init__(self, log_dir: str = "metrics_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{self.session_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{self.session_id}.json"

        self.metrics_buffer: List[QueryMetrics] = []

    def start_query(self) -> Dict[str, Any]:
        """Start tracking a query"""
        return {
            "start_time": time.time(),
            "start_memory": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }

    def end_query(
        self,
        tracking_data: Dict[str, Any],
        query: str,
        answer: str,
        retrieval_metadata: Dict[str, Any],
        retrieval_time: float,
        generation_time: float,
        context: str,
        config: Any,
        expected_answer: Optional[str] = None,
        course_filter: Optional[str] = None,
        courses_referenced: Optional[List[str]] = None,
    ) -> QueryMetrics:
        """End query tracking and calculate metrics"""

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Calculate accuracy if expected answer provided
        accuracy_score = None
        if expected_answer:
            accuracy_score = self._calculate_accuracy(answer, expected_answer)

        # Calculate relevance (simplified - could use more sophisticated methods)
        relevance_score = self._calculate_relevance(query, answer, context)

        metrics = QueryMetrics(
            timestamp=datetime.now().isoformat(),
            query=query,
            actual_answer=answer,
            total_time=end_time - tracking_data["start_time"],
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_vector_results=retrieval_metadata.get("vector_results", 0),
            num_kg_results=retrieval_metadata.get("kg_results", 0),
            num_total_results=retrieval_metadata.get("total_results", 0),
            fusion_strategy=retrieval_metadata.get("fusion_strategy", "unknown"),
            answer_length=len(answer),
            context_length=len(context),
            memory_used_mb=end_memory - tracking_data["start_memory"],
            cpu_percent=tracking_data["cpu_percent"],
            embedding_provider=config.EMBEDDING_PROVIDER,
            embedding_model=config.EMBEDDING_MODEL,
            llm_provider=config.LLM_PROVIDER,
            llm_model=config.LLM_MODEL,
            expected_answer=expected_answer,
            relevance_score=relevance_score,
            accuracy_score=accuracy_score,
            course_filter=course_filter,
            courses_referenced=courses_referenced or [],
        )

        self.metrics_buffer.append(metrics)
        self._log_metric(metrics)

        return metrics

    def _calculate_accuracy(self, actual: str, expected: str) -> float:
        """Calculate accuracy score between actual and expected answers"""
        # Normalize texts
        actual_norm = actual.lower().strip()
        expected_norm = expected.lower().strip()

        # Exact match
        if actual_norm == expected_norm:
            return 1.0

        # Check if expected answer is contained in actual
        if expected_norm in actual_norm:
            return 0.9

        # Word overlap score
        actual_words = set(actual_norm.split())
        expected_words = set(expected_norm.split())

        if not expected_words:
            return 0.0

        overlap = len(actual_words & expected_words)
        overlap_score = overlap / len(expected_words)

        # Semantic similarity would be better but requires additional models
        # For now, use word overlap as proxy
        return min(overlap_score, 1.0)

    def _calculate_relevance(self, query: str, answer: str, context: str) -> float:
        """Calculate relevance score of answer to query"""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return 0.5

        # Calculate word overlap
        overlap = len(query_words & answer_words)
        relevance = overlap / len(query_words)

        return min(relevance, 1.0)

    def _log_metric(self, metric: QueryMetrics):
        """Log metric to file"""
        with open(self.metrics_file, "a") as f:
            json.dump(asdict(metric), f)
            f.write("\n")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all tracked metrics"""
        if not self.metrics_buffer:
            return {}

        df = pd.DataFrame([asdict(m) for m in self.metrics_buffer])

        summary = {
            "session_id": self.session_id,
            "total_queries": len(self.metrics_buffer),
            "timestamp": datetime.now().isoformat(),
            # Timing statistics
            "timing": {
                "avg_total_time": float(df["total_time"].mean()),
                "avg_retrieval_time": float(df["retrieval_time"].mean()),
                "avg_generation_time": float(df["generation_time"].mean()),
                "max_total_time": float(df["total_time"].max()),
                "min_total_time": float(df["total_time"].min()),
            },
            # Retrieval statistics
            "retrieval": {
                "avg_vector_results": float(df["num_vector_results"].mean()),
                "avg_kg_results": float(df["num_kg_results"].mean()),
                "avg_total_results": float(df["num_total_results"].mean()),
                "fusion_strategies": df["fusion_strategy"].value_counts().to_dict(),
            },
            # Quality statistics
            "quality": {
                "avg_answer_length": float(df["answer_length"].mean()),
                "avg_context_length": float(df["context_length"].mean()),
                "avg_relevance_score": float(df["relevance_score"].mean()),
            },
            # Resource statistics
            "resources": {
                "avg_memory_used_mb": float(df["memory_used_mb"].mean()),
                "max_memory_used_mb": float(df["memory_used_mb"].max()),
                "avg_cpu_percent": float(df["cpu_percent"].mean()),
            },
            # Model configuration
            "model_config": {
                "embedding_provider": df["embedding_provider"].iloc[0],
                "embedding_model": df["embedding_model"].iloc[0],
                "llm_provider": df["llm_provider"].iloc[0],
                "llm_model": df["llm_model"].iloc[0],
            },
        }

        # Add accuracy if available
        if "accuracy_score" in df.columns and df["accuracy_score"].notna().any():
            summary["quality"]["avg_accuracy_score"] = float(
                df["accuracy_score"].dropna().mean()
            )
            summary["quality"]["accuracy_distribution"] = {
                "excellent (>0.9)": int((df["accuracy_score"] > 0.9).sum()),
                "good (0.7-0.9)": int(
                    (
                        (df["accuracy_score"] >= 0.7) & (df["accuracy_score"] <= 0.9)
                    ).sum()
                ),
                "fair (0.5-0.7)": int(
                    ((df["accuracy_score"] >= 0.5) & (df["accuracy_score"] < 0.7)).sum()
                ),
                "poor (<0.5)": int((df["accuracy_score"] < 0.5).sum()),
            }

        return summary

    def save_summary(self, silent: bool = False):
        """Save summary to file"""
        summary = self.get_summary()
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        if not silent:
            print(f"\n📊 Metrics summary saved to {self.summary_file}")

    def get_dataframe(self) -> pd.DataFrame:
        """Get metrics as pandas DataFrame"""
        return pd.DataFrame([asdict(m) for m in self.metrics_buffer])

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("METRICS SUMMARY")
        print("=" * 70)
        print(f"Session ID: {summary['session_id']}")
        print(f"Total Queries: {summary['total_queries']}")

        print(f"\n⏱️  TIMING METRICS:")
        for key, value in summary["timing"].items():
            print(f"   {key}: {value:.4f}s")

        print(f"\n🔍 RETRIEVAL METRICS:")
        for key, value in summary["retrieval"].items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"   {key}: {value:.2f}")

        print(f"\n✨ QUALITY METRICS:")
        for key, value in summary["quality"].items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"   {key}: {value:.2f}")

        print(f"\n💻 RESOURCE METRICS:")
        for key, value in summary["resources"].items():
            print(f"   {key}: {value:.2f}")

        print(f"\n🤖 MODEL CONFIGURATION:")
        for key, value in summary["model_config"].items():
            print(f"   {key}: {value}")

        print("=" * 70)
