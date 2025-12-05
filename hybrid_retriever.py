"""
hybrid_retriever.py - Hybrid retrieval combining Vector DB and Knowledge Graph
Windows-compatible with improved error handling and performance
Implements multiple fusion strategies for optimal results
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from langchain_core.documents import Document
from config import Config


@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata"""

    content: str
    source: str  # 'vector', 'kg', or 'hybrid'
    score: float
    metadata: Dict[str, Any]
    course_code: str = ""


class HybridRetriever:
    """
    Hybrid retriever combining vector similarity search with knowledge graph traversal
    Windows-compatible with enhanced error handling
    """

    def __init__(self, vector_store, knowledge_graph):
        """
        Initialize hybrid retriever
        
        Args:
            vector_store: FAISS vector store
            knowledge_graph: CourseKnowledgeGraph instance (can be None)
        """
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.fusion_strategy = Config.FUSION_STRATEGY
        
        # Track if KG is available
        self.kg_available = knowledge_graph is not None
        
        if not self.kg_available:
            print("⚠️  Knowledge Graph not available - using vector-only mode")

    def retrieve_from_vector_db(
        self, query: str, k: int, course_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve from vector database with error handling
        
        Args:
            query: Search query
            k: Number of results
            course_filter: Optional course code filter
            
        Returns:
            List of RetrievalResult objects
        """
        try:
            # Perform similarity search
            initial_k = k * 2 if course_filter else k
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=initial_k
            )

            # Apply course filter if specified
            if course_filter and course_filter != "All Courses":
                filtered_docs = [
                    (doc, score)
                    for doc, score in docs_with_scores
                    if doc.metadata.get("course_code") == course_filter
                ]
                docs_with_scores = filtered_docs[:k]
            else:
                docs_with_scores = docs_with_scores[:k]

            # Convert to RetrievalResult
            results = []
            for doc, score in docs_with_scores:
                # Convert distance to similarity score (0-1)
                # FAISS returns L2 distance, lower is better
                if score == 0:
                    similarity_score = 1.0
                else:
                    # Normalize score to 0-1 range
                    similarity_score = 1 / (1 + score)

                results.append(
                    RetrievalResult(
                        content=doc.page_content,
                        source="vector",
                        score=similarity_score,
                        metadata=doc.metadata,
                        course_code=doc.metadata.get("course_code", ""),
                    )
                )

            return results
            
        except Exception as e:
            print(f"⚠️  Vector DB retrieval error: {e}")
            return []

    def retrieve_from_kg(
        self, query: str, k: int, course_filter: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve from knowledge graph with error handling
        
        Args:
            query: Search query
            k: Number of results
            course_filter: Optional course code filter
            
        Returns:
            List of RetrievalResult objects
        """
        # Return empty if KG not available
        if not self.kg_available:
            return []
        
        try:
            # Query knowledge graph
            kg_results = self.kg.query_graph(
                query, course_code=course_filter, max_hops=Config.KG_MAX_HOPS
            )

            results = []
            for kg_result in kg_results[:k]:
                # Format KG information as text
                content_parts = []

                # Central entity
                central = kg_result["central_node"]
                node_type = kg_result["node_type"]
                content_parts.append(f"**{node_type}: {central}**\n")

                # Relationships
                if kg_result["relationships"]:
                    content_parts.append("Related Information:")
                    for rel in kg_result["relationships"][:5]:
                        rel_type = rel["type"].replace("_", " ").title()
                        content_parts.append(
                            f"- {rel['source']} → {rel_type} → {rel['target']}"
                        )

                # Neighbors
                if kg_result["neighbors"]:
                    neighbor_summary = []
                    for neighbor in kg_result["neighbors"][:5]:
                        neighbor_summary.append(
                            f"{neighbor['type']}: {neighbor['node']}"
                        )
                    if neighbor_summary:
                        content_parts.append(
                            "\nConnected Entities: " + ", ".join(neighbor_summary)
                        )

                content = "\n".join(content_parts)

                # Score based on number of relationships (heuristic)
                num_relationships = len(kg_result["relationships"])
                score = min(1.0, max(0.1, (num_relationships + 1) / 10.0))

                results.append(
                    RetrievalResult(
                        content=content,
                        source="kg",
                        score=score,
                        metadata={"kg_node": central, "node_type": node_type},
                        course_code=course_filter or "",
                    )
                )

            return results
            
        except Exception as e:
            print(f"⚠️  KG retrieval error: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        kg_results: List[RetrievalResult],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion (RRF) - combines rankings from multiple sources
        
        Args:
            vector_results: Results from vector search
            kg_results: Results from knowledge graph
            k: RRF constant (default 60)
            
        Returns:
            Fused and ranked results
        """
        # Create unified ranking
        rrf_scores = defaultdict(float)
        result_map = {}

        # Add vector results
        for rank, result in enumerate(vector_results):
            key = f"v_{hash(result.content[:100])}"
            rrf_scores[key] += 1 / (k + rank + 1)
            result_map[key] = result

        # Add KG results
        for rank, result in enumerate(kg_results):
            key = f"kg_{hash(result.content[:100])}"
            rrf_scores[key] += 1 / (k + rank + 1)
            result_map[key] = result

        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top results with updated scores
        fused_results = []
        for key, score in sorted_keys:
            result = result_map[key]
            result.score = score
            result.source = "hybrid"
            fused_results.append(result)

        return fused_results

    def _weighted_fusion(
        self, 
        vector_results: List[RetrievalResult], 
        kg_results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Weighted fusion - combines results with configured weights
        
        Args:
            vector_results: Results from vector search
            kg_results: Results from knowledge graph
            
        Returns:
            Weighted and ranked results
        """
        # Normalize scores with safe division
        if vector_results:
            max_v_score = max(r.score for r in vector_results)
            if max_v_score > 0:
                for r in vector_results:
                    r.score = (r.score / max_v_score) * Config.VECTOR_WEIGHT
            else:
                for r in vector_results:
                    r.score = Config.VECTOR_WEIGHT

        if kg_results:
            max_kg_score = max(r.score for r in kg_results)
            if max_kg_score > 0:
                for r in kg_results:
                    r.score = (r.score / max_kg_score) * Config.KG_WEIGHT
            else:
                for r in kg_results:
                    r.score = Config.KG_WEIGHT

        # Combine and sort
        all_results = vector_results + kg_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Mark as hybrid
        for r in all_results:
            r.source = "hybrid"

        return all_results

    def _adaptive_fusion(
        self,
        vector_results: List[RetrievalResult],
        kg_results: List[RetrievalResult],
        query: str,
    ) -> List[RetrievalResult]:
        """
        Adaptive fusion - dynamically adjusts weights based on query type
        
        Args:
            vector_results: Results from vector search
            kg_results: Results from knowledge graph
            query: Original query
            
        Returns:
            Adaptively weighted and ranked results
        """
        # Detect query intent
        query_lower = query.lower()

        # Factual queries benefit more from KG
        factual_keywords = [
            "prerequisite", "instructor", "assessment", "grading", "schedule",
            "when", "who", "percentage", "worth", "midterm", "final", "exam",
            "grade", "credit", "hours", "weight", "due", "date"
        ]
        is_factual = any(kw in query_lower for kw in factual_keywords)

        # Conceptual queries benefit more from vector search
        conceptual_keywords = [
            "explain", "describe", "what is", "how", "why", "learning outcome",
            "topic", "cover", "about", "overview", "summary", "teach", "learn"
        ]
        is_conceptual = any(kw in query_lower for kw in conceptual_keywords)

        # Adjust weights based on query type
        if is_factual:
            vector_weight = 0.3
            kg_weight = 0.7
            query_type = "factual"
        elif is_conceptual:
            vector_weight = 0.7
            kg_weight = 0.3
            query_type = "conceptual"
        else:
            vector_weight = Config.VECTOR_WEIGHT
            kg_weight = Config.KG_WEIGHT
            query_type = "balanced"

        # Normalize scores with adaptive weights and safe division
        if vector_results:
            max_v_score = max(r.score for r in vector_results)
            if max_v_score > 0:
                for r in vector_results:
                    r.score = (r.score / max_v_score) * vector_weight
            else:
                for r in vector_results:
                    r.score = vector_weight

        if kg_results:
            max_kg_score = max(r.score for r in kg_results)
            if max_kg_score > 0:
                for r in kg_results:
                    r.score = (r.score / max_kg_score) * kg_weight
            else:
                for r in kg_results:
                    r.score = kg_weight

        # Combine and sort
        all_results = vector_results + kg_results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Mark source with query type
        for r in all_results:
            r.source = f"hybrid-adaptive({query_type})"

        return all_results

    def retrieve(
        self, 
        query: str, 
        k: int = None, 
        course_filter: Optional[str] = None
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Main retrieval method - combines vector and KG retrieval
        
        Args:
            query: Search query
            k: Number of results (defaults to Config.RETRIEVAL_TOP_K)
            course_filter: Optional course code to filter results
            
        Returns:
            Tuple of (results, metadata)
        """
        k = k or Config.RETRIEVAL_TOP_K

        # Retrieve from both sources
        vector_results = self.retrieve_from_vector_db(query, k, course_filter)
        
        # Only retrieve from KG if available
        if self.kg_available:
            kg_results = self.retrieve_from_kg(query, k, course_filter)
        else:
            kg_results = []

        # If no KG results, return vector results only
        if not kg_results:
            metadata = {
                "vector_results": len(vector_results),
                "kg_results": 0,
                "total_results": len(vector_results),
                "fusion_strategy": "vector-only",
                "sources_breakdown": {
                    "vector": len(vector_results),
                    "kg": 0,
                    "hybrid": 0
                },
            }
            return vector_results[:k], metadata

        # Fuse results based on strategy
        if self.fusion_strategy == "rrf":
            fused_results = self._reciprocal_rank_fusion(vector_results, kg_results)
        elif self.fusion_strategy == "weighted":
            fused_results = self._weighted_fusion(vector_results, kg_results)
        elif self.fusion_strategy == "adaptive":
            fused_results = self._adaptive_fusion(vector_results, kg_results, query)
        else:
            # Default to weighted
            fused_results = self._weighted_fusion(vector_results, kg_results)

        # Metadata about retrieval
        metadata = {
            "vector_results": len(vector_results),
            "kg_results": len(kg_results),
            "total_results": len(fused_results),
            "fusion_strategy": self.fusion_strategy,
            "sources_breakdown": {
                "vector": len([r for r in fused_results if "vector" in r.source]),
                "kg": len([r for r in fused_results if r.source == "kg"]),
                "hybrid": len([r for r in fused_results if "hybrid" in r.source]),
            },
        }

        return fused_results[:k], metadata

    def format_results_for_llm(
        self, 
        results: List[RetrievalResult], 
        max_length: int = 3000
    ) -> str:
        """
        Format retrieval results for LLM context
        
        Args:
            results: List of RetrievalResult objects
            max_length: Maximum total length of formatted text
            
        Returns:
            Formatted string for LLM context
        """
        formatted_parts = []
        current_length = 0

        for i, result in enumerate(results):
            # Format with source indicator
            source_label = {
                "vector": "📄 Vector DB",
                "kg": "🔗 Knowledge Graph",
            }.get(result.source, "🔀 Hybrid")

            course_info = ""
            if result.course_code:
                course_info = f" [Course: {result.course_code}]"

            header = f"{source_label}{course_info} (Score: {result.score:.3f})"
            
            # Truncate content if needed
            content = result.content[:max_length].strip()

            part = f"{header}\n{content}\n"

            # Check if adding this part would exceed max_length
            if current_length + len(part) > max_length and formatted_parts:
                break

            formatted_parts.append(part)
            current_length += len(part)

        return "\n---\n".join(formatted_parts)
