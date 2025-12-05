"""
knowledge_graph.py - Knowledge Graph Construction and Retrieval
Windows-compatible with NetworkX for local graph storage
Performance optimized with better entity/relationship extraction
"""

import os
import json
import pickle
import re
from typing import List, Dict, Any, Tuple, Set
from pathlib import Path
from collections import defaultdict

import networkx as nx
from config import Config


class CourseKnowledgeGraph:
    """Knowledge Graph for Course Outlines using NetworkX (Windows-compatible)"""

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir or Config.KG_PERSIST_DIR
        
        # Ensure directory exists (Windows-compatible)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self.graph = nx.MultiDiGraph()
        self.entity_index = defaultdict(list)  # For fast entity lookup
        self.course_entities = defaultdict(set)  # Course -> entities mapping

        self.llm = None  # Lazy load if needed

    def _get_llm(self):
        """Lazy load LLM for entity extraction (if needed)"""
        if self.llm is None:
            from model_factory import ModelFactory
            self.llm = ModelFactory.create_llm()
        return self.llm

    def extract_entities_and_relations(
        self, text: str, course_code: str, chunk_id: int
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from text using rule-based patterns
        Returns: (entities, relationships)
        """
        # Rule-based extraction for common course outline patterns
        entities = []
        relationships = []

        # 1. Extract Course Information
        entities.append(
            {
                "name": course_code,
                "type": "Course",
                "properties": {"chunk_id": chunk_id},
            }
        )

        # 2. Extract Prerequisites
        prereq_patterns = [
            r"prerequisite(?:s)?[:\s]+([A-Z]{2,4}\d{3,4}[A-Z]?(?:[,\s]+(?:and|or)[,\s]+[A-Z]{2,4}\d{3,4}[A-Z]?)*)",
            r"requires?[:\s]+([A-Z]{2,4}\d{3,4}[A-Z]?)",
            r"pre-req(?:uisite)?[:\s]+([A-Z]{2,4}\d{3,4}[A-Z]?)",
        ]

        for pattern in prereq_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                prereq_text = match.group(1)
                prereq_courses = re.findall(r"[A-Z]{2,4}\d{3,4}[A-Z]?", prereq_text)

                for prereq in prereq_courses:
                    entities.append(
                        {"name": prereq, "type": "Course", "properties": {}}
                    )
                    relationships.append(
                        {
                            "source": prereq,
                            "target": course_code,
                            "type": "PREREQUISITE_FOR",
                            "properties": {"chunk_id": chunk_id},
                        }
                    )

        # 3. Extract Topics/Concepts
        topic_patterns = [
            r"topics?\s*(?:covered|include|discussed)[:\s]+([^.]+(?:\.[^.]+){0,2})",
            r"(?:week|lecture)\s*\d+[:\s]+([^.\n]+)",
            r"learning\s*outcomes?[:\s]+([^.]+(?:\.[^.]+){0,3})",
            r"(?:will|students)\s+(?:learn|study|cover)[:\s]+([^.]+)",
        ]

        topics_found = set()
        for pattern in topic_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                topic_text = match.group(1).strip()
                # Extract key phrases
                topic_phrases = [
                    p.strip()
                    for p in re.split(r"[,;•\n]", topic_text)
                    if len(p.strip()) > 10 and len(p.strip()) < 100
                ]

                for phrase in topic_phrases[:5]:  # Limit to avoid noise
                    if phrase not in topics_found:
                        topics_found.add(phrase)
                        entity_name = f"Topic: {phrase[:50]}"
                        entities.append(
                            {
                                "name": entity_name,
                                "type": "Topic",
                                "properties": {
                                    "description": phrase,
                                    "chunk_id": chunk_id,
                                },
                            }
                        )
                        relationships.append(
                            {
                                "source": course_code,
                                "target": entity_name,
                                "type": "COVERS",
                                "properties": {"chunk_id": chunk_id},
                            }
                        )

        # 4. Extract Instructors (IMPROVED - multiple patterns)
        instructor_patterns = [
            r"(?:instructor|professor|teacher|taught\s+by|faculty)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
            r"(?:Dr\.|Prof\.|Professor|Mr\.|Ms\.|Mrs\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:will\s+teach|is\s+teaching|teaches)",
            r"(?:contact|email)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[<@]",
        ]

        instructors_found = set()
        for pattern in instructor_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                instructor = match.group(1).strip()
                
                # Validate it's a real name (at least 2 words, proper capitalization)
                words = instructor.split()
                if len(words) >= 2 and instructor not in instructors_found:
                    # Additional validation: check it's not a common phrase
                    common_false_positives = {
                        "Course Code", "Course Name", "Course Title", 
                        "Learning Outcomes", "Final Exam", "Midterm Exam"
                    }
                    if instructor not in common_false_positives:
                        instructors_found.add(instructor)
                        entities.append(
                            {
                                "name": instructor,
                                "type": "Instructor",
                                "properties": {"chunk_id": chunk_id},
                            }
                        )
                        relationships.append(
                            {
                                "source": instructor,
                                "target": course_code,
                                "type": "TEACHES",
                                "properties": {"chunk_id": chunk_id},
                            }
                        )

        # 5. Extract Assessment Components
        assessment_patterns = [
            r"(assignment|exam|quiz|midterm|final|project|presentation)(?:s)?[:\s]+(\d+)%",
            r"(\d+)%[:\s]+(assignment|exam|quiz|midterm|final|project)",
        ]

        for pattern in assessment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if groups[0].isdigit():
                    weight, component = groups[0], groups[1]
                else:
                    component, weight = groups[0], groups[1]

                entity_name = f"Assessment: {component.title()}"
                entities.append(
                    {
                        "name": entity_name,
                        "type": "Assessment",
                        "properties": {"weight": f"{weight}%", "chunk_id": chunk_id},
                    }
                )
                relationships.append(
                    {
                        "source": course_code,
                        "target": entity_name,
                        "type": "HAS_ASSESSMENT",
                        "properties": {"weight": f"{weight}%", "chunk_id": chunk_id},
                    }
                )

        return entities, relationships

    def add_entities_and_relations(
        self, entities: List[Dict], relationships: List[Dict], course_code: str
    ):
        """Add entities and relationships to the graph"""
        # Add entities as nodes
        for entity in entities:
            node_id = entity["name"]
            self.graph.add_node(
                node_id, type=entity["type"], **entity.get("properties", {})
            )

            # Update entity index
            self.entity_index[entity["type"]].append(node_id)
            self.course_entities[course_code].add(node_id)

        # Add relationships as edges
        for rel in relationships:
            self.graph.add_edge(
                rel["source"],
                rel["target"],
                type=rel["type"],
                **rel.get("properties", {}),
            )

    def build_from_documents(self, documents: List[Dict], show_progress: bool = True):
        """Build knowledge graph from course outline documents"""
        print(f"\n🔨 Building Knowledge Graph from {len(documents)} documents...")

        for idx, doc in enumerate(documents):
            if show_progress and (idx % 5 == 0 or idx == len(documents) - 1):
                print(f"   Processing: {idx + 1}/{len(documents)}...")

            raw_text = doc.get("raw_text", "")
            course_code = doc.get("url", f"COURSE_{idx}")
            chunk_id = idx

            # Extract entities and relationships
            entities, relationships = self.extract_entities_and_relations(
                raw_text, course_code, chunk_id
            )

            # Add to graph
            self.add_entities_and_relations(entities, relationships, course_code)

        print(f"✅ Knowledge Graph built:")
        print(f"   - Nodes: {self.graph.number_of_nodes():,}")
        print(f"   - Edges: {self.graph.number_of_edges():,}")
        print(f"   - Courses: {len(self.course_entities)}")

    def query_graph(
        self, query: str, course_code: str = None, max_hops: int = None
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge graph for relevant information (OPTIMIZED)
        Returns list of relevant subgraphs and entities
        """
        max_hops = max_hops or Config.KG_MAX_HOPS
        results = []

        # Extract potential entity names from query (optimized)
        query_upper = query.upper()
        query_tokens = set(token for token in query_upper.split() if len(token) > 2)

        # 1. Direct entity matches (optimized)
        matched_nodes = []

        # Fast course code matching
        if course_code and course_code in self.graph:
            matched_nodes.append(course_code)

        # Fast entity type matching
        query_lower = query.lower()
        if (
            "instructor" in query_lower
            or "professor" in query_lower
            or "teacher" in query_lower
        ):
            matched_nodes.extend(self.entity_index.get("Instructor", [])[:5])

        # Token-based matching (limited to avoid slowdown)
        node_list = list(self.graph.nodes())[
            : min(1000, len(self.graph.nodes()))
        ]
        for node in node_list:
            node_upper = str(node).upper()
            if any(token in node_upper for token in query_tokens):
                matched_nodes.append(node)
                if len(matched_nodes) >= Config.KG_TOP_ENTITIES * 2:
                    break

        # 2. Course-specific filtering
        if course_code:
            course_nodes = self.course_entities.get(course_code, set())
            matched_nodes = [
                n for n in matched_nodes if n in course_nodes or n == course_code
            ]

            # If no matches, use course as starting point
            if not matched_nodes and course_code in self.graph:
                matched_nodes = [course_code]

        # Remove duplicates while preserving order
        seen = set()
        matched_nodes = [x for x in matched_nodes if not (x in seen or seen.add(x))]

        # 3. Extract subgraphs around matched nodes (optimized)
        for node in matched_nodes[: Config.KG_TOP_ENTITIES]:
            try:
                # Get immediate neighbors only for speed
                if max_hops == 1:
                    neighbors = [node] + list(self.graph.neighbors(node))[:20]
                else:
                    # Limited BFS for larger hops
                    neighbors = list(
                        nx.single_source_shortest_path_length(
                            self.graph, node, cutoff=max_hops
                        ).keys()
                    )[
                        :30
                    ]

                # Collect information
                node_info = {
                    "central_node": node,
                    "node_type": self.graph.nodes[node].get("type", "Unknown"),
                    "neighbors": [],
                    "relationships": [],
                }

                # Collect neighbor information (limited)
                for neighbor in neighbors[:15]:
                    if neighbor != node:
                        node_info["neighbors"].append(
                            {
                                "node": neighbor,
                                "type": self.graph.nodes[neighbor].get(
                                    "type", "Unknown"
                                ),
                                "properties": dict(self.graph.nodes[neighbor]),
                            }
                        )

                # Collect edge information (limited)
                subgraph = self.graph.subgraph(neighbors)
                edge_count = 0
                for source, target, key, data in subgraph.edges(keys=True, data=True):
                    node_info["relationships"].append(
                        {
                            "source": source,
                            "target": target,
                            "type": data.get("type", "RELATED_TO"),
                            "properties": {
                                k: v for k, v in data.items() if k != "type"
                            },
                        }
                    )
                    edge_count += 1
                    if edge_count >= 20:
                        break

                results.append(node_info)

            except Exception as e:
                print(f"Warning: Error processing node {node}: {e}")
                continue

        return results

    def get_course_summary(self, course_code: str) -> Dict[str, Any]:
        """Get comprehensive summary of a course from the graph"""
        if course_code not in self.graph:
            return {}

        summary = {
            "course_code": course_code,
            "prerequisites": [],
            "topics": [],
            "assessments": [],
            "instructors": [],
        }

        # Get all neighbors
        for neighbor in self.graph.neighbors(course_code):
            node_type = self.graph.nodes[neighbor].get("type", "Unknown")

            if node_type == "Topic":
                summary["topics"].append(neighbor)
            elif node_type == "Assessment":
                props = dict(self.graph.nodes[neighbor])
                summary["assessments"].append(
                    {"name": neighbor, "weight": props.get("weight", "N/A")}
                )

        # Get prerequisites (reverse edges)
        for predecessor in self.graph.predecessors(course_code):
            for edge_key, edge_data in self.graph[predecessor][course_code].items():
                if edge_data.get("type") == "PREREQUISITE_FOR":
                    summary["prerequisites"].append(predecessor)

        # Get instructors
        for predecessor in self.graph.predecessors(course_code):
            if self.graph.nodes[predecessor].get("type") == "Instructor":
                summary["instructors"].append(predecessor)

        return summary

    def save(self):
        """Save knowledge graph to disk (Windows-compatible)"""
        # Ensure directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        graph_file = Path(self.persist_dir) / "course_kg.pkl"
        index_file = Path(self.persist_dir) / "entity_index.json"

        # Save graph using pickle
        with open(graph_file, "wb") as f:
            pickle.dump(self.graph, f, pickle.HIGHEST_PROTOCOL)

        # Save indexes
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "entity_index": {k: list(v) for k, v in self.entity_index.items()},
                    "course_entities": {
                        k: list(v) for k, v in self.course_entities.items()
                    },
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"💾 Knowledge Graph saved to {Path(self.persist_dir).resolve()}")

    def load(self):
        """Load knowledge graph from disk (Windows-compatible)"""
        graph_file = Path(self.persist_dir) / "course_kg.pkl"
        index_file = Path(self.persist_dir) / "entity_index.json"

        if not graph_file.exists():
            raise FileNotFoundError(f"Knowledge graph not found at {graph_file}")

        # Load graph using pickle
        with open(graph_file, "rb") as f:
            self.graph = pickle.load(f)

        # Load indexes
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.entity_index = defaultdict(
                list, {k: v for k, v in data["entity_index"].items()}
            )
            self.course_entities = defaultdict(
                set, {k: set(v) for k, v in data["course_entities"].items()}
            )

        # Simplified output
        print(
            f"✓ KG loaded: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        entity_types = defaultdict(int)
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", "Unknown")
            entity_types[node_type] += 1

        relation_types = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("type", "Unknown")
            relation_types[rel_type] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "courses": len(self.course_entities),
        }
