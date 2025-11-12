"""
Cross-Modal Knowledge Graph
Links text knowledge ↔ image knowledge with bidirectional mappings.

Revolutionary capabilities:
- Query with text, get both text answers AND relevant diagrams
- Query with image, get text explanations AND similar images
- Validate formulas appear in diagrams (cross-modal verification)
- Discover hidden connections between concepts and visuals
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
from difflib import SequenceMatcher
import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TextNode:
    """Text knowledge node"""
    id: str
    content: str
    node_type: str  # formula, material, code_requirement, concept, etc.
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    linked_images: Set[str] = field(default_factory=set)
    linked_texts: Set[str] = field(default_factory=set)


@dataclass
class ImageNode:
    """Image knowledge node"""
    id: str
    image_path: str
    description: str
    node_type: str  # diagram, photo, blueprint, chart, etc.
    domain: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    linked_texts: Set[str] = field(default_factory=set)
    linked_images: Set[str] = field(default_factory=set)


@dataclass
class CrossModalLink:
    """Link between text and image nodes"""
    source_id: str
    target_id: str
    link_type: str  # illustrates, contains, validates, similar_to, etc.
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisualKnowledgeGraph:
    """
    Cross-modal knowledge graph linking text ↔ images.
    
    Enables revolutionary queries:
    - "Show me diagrams related to foundation design" (text → images)
    - "What does this diagram explain?" (image → text)
    - "Verify this formula appears in blueprints" (cross-validation)
    """
    
    def __init__(self, graph_path: str = "data/knowledge_graph"):
        """
        Initialize cross-modal knowledge graph.
        
        Args:
            graph_path: Path to store graph data
        """
        self.graph_path = Path(graph_path)
        self.graph_path.mkdir(parents=True, exist_ok=True)
        
        # Node storage
        self.text_nodes: Dict[str, TextNode] = {}
        self.image_nodes: Dict[str, ImageNode] = {}
        
        # Link storage
        self.links: Dict[Tuple[str, str], CrossModalLink] = {}
        
        # Indexes for fast lookup
        self.text_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.image_by_type: Dict[str, Set[str]] = defaultdict(set)
        self.text_by_domain: Dict[str, Set[str]] = defaultdict(set)
        self.image_by_domain: Dict[str, Set[str]] = defaultdict(set)
        
        # Load existing graph
        self._load_graph()
        
        logger.info(f"✅ Visual Knowledge Graph initialized")
        logger.info(f"   Text nodes: {len(self.text_nodes)}")
        logger.info(f"   Image nodes: {len(self.image_nodes)}")
        logger.info(f"   Cross-modal links: {len(self.links)}")
    
    def add_text_node(
        self,
        node_id: str,
        content: str,
        node_type: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TextNode:
        """
        Add text knowledge node.
        
        Args:
            node_id: Unique node identifier
            content: Text content
            node_type: Type (formula, material, code, etc.)
            domain: Knowledge domain
            metadata: Optional metadata
        
        Returns:
            Created TextNode
        """
        node = TextNode(
            id=node_id,
            content=content,
            node_type=node_type,
            domain=domain,
            metadata=metadata or {}
        )
        
        self.text_nodes[node_id] = node
        self.text_by_type[node_type].add(node_id)
        self.text_by_domain[domain].add(node_id)
        
        logger.debug(f"Added text node: {node_id} ({node_type})")
        
        return node
    
    def add_image_node(
        self,
        node_id: str,
        image_path: str,
        description: str,
        node_type: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ImageNode:
        """
        Add image knowledge node.
        
        Args:
            node_id: Unique node identifier
            image_path: Path to image
            description: Image description
            node_type: Type (diagram, photo, blueprint, etc.)
            domain: Knowledge domain
            metadata: Optional metadata
        
        Returns:
            Created ImageNode
        """
        node = ImageNode(
            id=node_id,
            image_path=image_path,
            description=description,
            node_type=node_type,
            domain=domain,
            metadata=metadata or {}
        )
        
        self.image_nodes[node_id] = node
        self.image_by_type[node_type].add(node_id)
        self.image_by_domain[domain].add(node_id)
        
        logger.debug(f"Added image node: {node_id} ({node_type})")
        
        return node
    
    def link_text_to_image(
        self,
        text_id: str,
        image_id: str,
        link_type: str = "illustrates",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Create bidirectional link between text and image.
        
        Args:
            text_id: Text node ID
            image_id: Image node ID
            link_type: Type of link (illustrates, contains, validates)
            confidence: Link confidence (0-1)
            metadata: Optional metadata
        """
        if text_id not in self.text_nodes:
            logger.warning(f"Text node {text_id} not found")
            return
        
        if image_id not in self.image_nodes:
            logger.warning(f"Image node {image_id} not found")
            return
        
        # Create link
        link = CrossModalLink(
            source_id=text_id,
            target_id=image_id,
            link_type=link_type,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        self.links[(text_id, image_id)] = link
        
        # Update bidirectional references
        self.text_nodes[text_id].linked_images.add(image_id)
        self.image_nodes[image_id].linked_texts.add(text_id)
        
        logger.debug(f"Linked: {text_id} --[{link_type}]--> {image_id}")
    
    def link_image_to_image(
        self,
        image_id_1: str,
        image_id_2: str,
        link_type: str = "similar_to",
        confidence: float = 1.0
    ):
        """Link two similar images"""
        if image_id_1 not in self.image_nodes or image_id_2 not in self.image_nodes:
            logger.warning("One or both image nodes not found")
            return
        
        self.image_nodes[image_id_1].linked_images.add(image_id_2)
        self.image_nodes[image_id_2].linked_images.add(image_id_1)
        
        logger.debug(f"Linked images: {image_id_1} <--> {image_id_2}")
    
    def query_with_text(
        self,
        text_query: str,
        include_text: bool = True,
        include_images: bool = True,
        domain_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query graph with text, get both text and image results.
        
        Args:
            text_query: Text search query
            include_text: Include text nodes in results
            include_images: Include linked images in results
            domain_filter: Optional domain filter
        
        Returns:
            Dict with text_results and image_results
        """
        logger.info(f"🔍 Querying graph with text: '{text_query}'")
        
        text_results = []
        image_results = []
        
        # Search text nodes
        query_lower = text_query.lower()
        
        for node_id, node in self.text_nodes.items():
            # Apply domain filter
            if domain_filter and node.domain != domain_filter:
                continue
            
            # Simple text matching (in production, use embeddings)
            if query_lower in node.content.lower():
                if include_text:
                    text_results.append({
                        "id": node.id,
                        "content": node.content,
                        "type": node.node_type,
                        "domain": node.domain,
                        "linked_images": list(node.linked_images)
                    })
                
                # Get linked images
                if include_images:
                    for image_id in node.linked_images:
                        if image_id in self.image_nodes:
                            img_node = self.image_nodes[image_id]
                            image_results.append({
                                "id": img_node.id,
                                "path": img_node.image_path,
                                "description": img_node.description,
                                "type": img_node.node_type,
                                "domain": img_node.domain,
                                "linked_from_text": node.id
                            })
        
        # Remove duplicate images
        unique_images = {img["id"]: img for img in image_results}
        image_results = list(unique_images.values())
        
        logger.info(f"📊 Found {len(text_results)} text results, {len(image_results)} image results")
        
        return {
            "query": text_query,
            "text_results": text_results,
            "image_results": image_results,
            "total_results": len(text_results) + len(image_results)
        }
    
    def query_with_image(
        self,
        image_id: str,
        include_text: bool = True,
        include_similar_images: bool = True
    ) -> Dict[str, Any]:
        """
        Query graph with image, get text explanations and similar images.
        
        Args:
            image_id: Image node ID
            include_text: Include linked text nodes
            include_similar_images: Include similar images
        
        Returns:
            Dict with text_results and similar_images
        """
        logger.info(f"🔍 Querying graph with image: {image_id}")
        
        if image_id not in self.image_nodes:
            return {"error": f"Image node {image_id} not found"}
        
        image_node = self.image_nodes[image_id]
        
        text_results = []
        similar_images = []
        
        # Get linked text
        if include_text:
            for text_id in image_node.linked_texts:
                if text_id in self.text_nodes:
                    text_node = self.text_nodes[text_id]
                    text_results.append({
                        "id": text_node.id,
                        "content": text_node.content,
                        "type": text_node.node_type,
                        "domain": text_node.domain
                    })
        
        # Get similar images
        if include_similar_images:
            for similar_id in image_node.linked_images:
                if similar_id in self.image_nodes:
                    similar_node = self.image_nodes[similar_id]
                    similar_images.append({
                        "id": similar_node.id,
                        "path": similar_node.image_path,
                        "description": similar_node.description,
                        "type": similar_node.node_type,
                        "domain": similar_node.domain
                    })
        
        return {
            "query_image": {
                "id": image_node.id,
                "path": image_node.image_path,
                "description": image_node.description
            },
            "text_explanations": text_results,
            "similar_images": similar_images,
            "total_results": len(text_results) + len(similar_images)
        }
    
    async def find_visual_evidence(
        self,
        text: str,
        query: str,
        top_k: int = 3,
        domain: Optional[str] = None,
        vision_engine=None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant diagrams/images for a given answer.
        
        Args:
            text: Synthesized answer text
            query: Original query
            top_k: Maximum number of diagrams to return
            domain: Optional domain filter
            vision_engine: Optional vision engine for additional validation
        """
        if not text:
            return []
        
        search_text = f"{query}\n{text}".strip()
        results: Dict[str, Dict[str, Any]] = {}
        
        for node_id, node in self.text_nodes.items():
            if domain and node.domain != domain:
                continue
            
            similarity = self._text_similarity(search_text, node.content)
            if similarity < 0.2:
                continue
            
            for image_id in node.linked_images:
                if image_id not in self.image_nodes:
                    continue
                
                image_node = self.image_nodes[image_id]
                link = self.links.get((node_id, image_id))
                link_confidence = link.confidence if link else 0.5
                
                image_similarity = self._text_similarity(search_text, image_node.description)
                base_score = 0.6 * similarity + 0.4 * image_similarity
                base_score = min(1.0, base_score * (0.8 + link_confidence * 0.2))
                
                vision_score = 0.0
                analysis_summary = ""
                
                if vision_engine and self._looks_like_image_path(image_node.image_path) and os.path.exists(image_node.image_path):
                    analysis = await self._analyze_with_vision(
                        vision_engine,
                        image_node.image_path,
                        f"Explain how this image relates to: {search_text}"
                    )
                    analysis_text = analysis.get("analysis", "")
                    analysis_summary = analysis_text[:500]
                    vision_score = self._text_similarity(search_text, analysis_text) if analysis_text else 0.0
                    base_score = min(1.0, base_score * 0.7 + vision_score * 0.3)
                
                previous = results.get(image_id)
                combined_score = base_score
                if previous:
                    combined_score = max(previous["relevance"], combined_score)
                
                results[image_id] = {
                    "image_id": image_id,
                    "image_path": image_node.image_path,
                    "description": image_node.description,
                    "domain": image_node.domain,
                    "relevance": combined_score,
                    "link_confidence": link_confidence,
                    "source_text_node": node_id,
                    "analysis": analysis_summary,
                    "vision_score": vision_score
                }
        
        ranked = sorted(results.values(), key=lambda item: item["relevance"], reverse=True)
        return ranked[:top_k]
    
    def validate_formula_in_diagrams(
        self,
        formula_id: str,
        vision_engine=None
    ) -> Dict[str, Any]:
        """
        🔥 Validate that a formula actually appears in linked diagrams.
        
        Uses vision model to verify cross-modal consistency.
        
        Args:
            formula_id: Formula text node ID
            vision_engine: Vision model for validation
        
        Returns:
            Validation results
        """
        if formula_id not in self.text_nodes:
            return {"error": f"Formula node {formula_id} not found"}
        
        formula_node = self.text_nodes[formula_id]
        formula_content = formula_node.content
        
        logger.info(f"🔍 Validating formula in diagrams: {formula_content[:50]}...")
        
        if not formula_node.linked_images:
            return {
                "formula": formula_content,
                "validation": "no_diagrams_linked"
            }
        
        if not vision_engine:
            return {
                "formula": formula_content,
                "linked_diagrams": len(formula_node.linked_images),
                "validation": "vision_engine_required"
            }
        
        # Check each linked diagram
        validations = []
        
        for image_id in formula_node.linked_images:
            if image_id not in self.image_nodes:
                continue
            
            image_node = self.image_nodes[image_id]
            
            # Ask vision model if formula appears in image
            try:
                query = f"Does this formula or calculation appear in this diagram: {formula_content}? Answer yes or no and explain where."
                
                result = vision_engine.analyze_image(
                    image_node.image_path,
                    query
                )
                
                analysis = result.get("analysis", "").lower()
                appears = "yes" in analysis[:100]
                
                validations.append({
                    "diagram": image_node.image_path,
                    "formula_appears": appears,
                    "analysis": result.get("analysis", "")
                })
                
            except Exception as e:
                logger.error(f"Error validating {image_id}: {e}")
        
        # Calculate validation score
        if validations:
            validation_score = sum(v["formula_appears"] for v in validations) / len(validations)
        else:
            validation_score = 0.0
        
        return {
            "formula": formula_content,
            "linked_diagrams": len(formula_node.linked_images),
            "validations": validations,
            "validation_score": validation_score,
            "validated": validation_score > 0.5
        }

    async def _ensure_image_node(self, source: str, domain: str, vision_engine=None) -> Optional[str]:
        """Ensure an image node exists for a given source reference."""
        if not source or not isinstance(source, str):
            return None
        
        if not self._looks_like_image_path(source):
            return None

        image_path = Path(source)
        if not image_path.exists():
            return None

        image_id = f"img_{hashlib.sha1(source.encode()).hexdigest()[:10]}"
        if image_id in self.image_nodes:
            return image_id

        description = image_path.stem.replace('_', ' ').title()
        metadata = {"source": source}

        if vision_engine:
            analysis = await self._analyze_with_vision(
                vision_engine,
                str(image_path),
                "Provide a concise description of this technical diagram."
            )
            analysis_text = analysis.get("analysis")
            if analysis_text:
                metadata["vision_summary"] = analysis_text[:500]
                description = analysis_text.split('\n')[0][:160]

        self.add_image_node(
            node_id=image_id,
            image_path=str(image_path),
            description=description,
            node_type='diagram',
            domain=domain,
            metadata=metadata
        )
        return image_id

    def _looks_like_image_path(self, source: str) -> bool:
        return bool(re.search(r'\.(png|jpe?g|gif|bmp|tif|tiff|svg)$', source.lower()))

    async def _analyze_with_vision(self, vision_engine, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze an image using vision engine, handling sync/async responses."""
        if not vision_engine:
            return {}
        try:
            response = vision_engine.analyze_image(image_path, prompt)
            if asyncio.iscoroutine(response):
                response = await response
            if isinstance(response, dict):
                return response
            return {"analysis": str(response)}
        except Exception as e:
            logger.error(f"Vision analysis failed for {image_path}: {e}")
            return {"error": str(e)}

    def _text_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        a_norm = re.sub(r'\s+', ' ', a.lower()).strip()
        b_norm = re.sub(r'\s+', ' ', b.lower()).strip()
        if not a_norm or not b_norm:
            return 0.0
        return SequenceMatcher(None, a_norm, b_norm).ratio()
    
    def get_subgraph(
        self,
        center_node_id: str,
        depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get subgraph around a node (within N hops).
        
        Args:
            center_node_id: Center node ID
            depth: Number of hops to include
        
        Returns:
            Subgraph with nodes and links
        """
        nodes_to_include = {center_node_id}
        current_layer = {center_node_id}
        
        # BFS traversal
        for _ in range(depth):
            next_layer = set()
            
            for node_id in current_layer:
                # Add connected nodes
                if node_id in self.text_nodes:
                    node = self.text_nodes[node_id]
                    next_layer.update(node.linked_images)
                    next_layer.update(node.linked_texts)
                elif node_id in self.image_nodes:
                    node = self.image_nodes[node_id]
                    next_layer.update(node.linked_texts)
                    next_layer.update(node.linked_images)
            
            nodes_to_include.update(next_layer)
            current_layer = next_layer
        
        # Extract subgraph
        subgraph_text = {
            nid: self.text_nodes[nid]
            for nid in nodes_to_include
            if nid in self.text_nodes
        }
        
        subgraph_images = {
            nid: self.image_nodes[nid]
            for nid in nodes_to_include
            if nid in self.image_nodes
        }
        
        # Extract relevant links
        subgraph_links = {
            link_key: link
            for link_key, link in self.links.items()
            if link.source_id in nodes_to_include and link.target_id in nodes_to_include
        }
        
        return {
            "center": center_node_id,
            "depth": depth,
            "text_nodes": len(subgraph_text),
            "image_nodes": len(subgraph_images),
            "links": len(subgraph_links),
            "nodes": {
                "text": {nid: self._serialize_text_node(node) for nid, node in subgraph_text.items()},
                "images": {nid: self._serialize_image_node(node) for nid, node in subgraph_images.items()}
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        # Calculate connectivity
        connected_text = sum(
            1 for node in self.text_nodes.values()
            if node.linked_images or node.linked_texts
        )
        
        connected_images = sum(
            1 for node in self.image_nodes.values()
            if node.linked_texts or node.linked_images
        )
        
        return {
            "total_nodes": len(self.text_nodes) + len(self.image_nodes),
            "text_nodes": len(self.text_nodes),
            "image_nodes": len(self.image_nodes),
            "cross_modal_links": len(self.links),
            "connected_text_nodes": connected_text,
            "connected_image_nodes": connected_images,
            "connectivity_rate": {
                "text": connected_text / len(self.text_nodes) if self.text_nodes else 0,
                "images": connected_images / len(self.image_nodes) if self.image_nodes else 0
            },
            "domains": {
                "text": dict(self.text_by_domain),
                "images": dict(self.image_by_domain)
            }
        }
    
    def save_graph(self):
        """Save graph to disk"""
        try:
            # Save text nodes
            text_data = {
                nid: self._serialize_text_node(node)
                for nid, node in self.text_nodes.items()
            }
            
            # Save image nodes
            image_data = {
                nid: self._serialize_image_node(node)
                for nid, node in self.image_nodes.items()
            }
            
            # Save links
            links_data = [
                {
                    "source": link.source_id,
                    "target": link.target_id,
                    "type": link.link_type,
                    "confidence": link.confidence,
                    "metadata": link.metadata
                }
                for link in self.links.values()
            ]
            
            # Write to files
            with open(self.graph_path / "text_nodes.json", 'w') as f:
                json.dump(text_data, f, indent=2)
            
            with open(self.graph_path / "image_nodes.json", 'w') as f:
                json.dump(image_data, f, indent=2)
            
            with open(self.graph_path / "links.json", 'w') as f:
                json.dump(links_data, f, indent=2)
            
            logger.info(f"💾 Saved knowledge graph: {len(self.text_nodes)} text, {len(self.image_nodes)} images, {len(self.links)} links")
            
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
    
    def _load_graph(self):
        """Load graph from disk"""
        try:
            # Load text nodes
            text_file = self.graph_path / "text_nodes.json"
            if text_file.exists():
                with open(text_file) as f:
                    text_data = json.load(f)
                
                for nid, data in text_data.items():
                    node = TextNode(
                        id=data["id"],
                        content=data["content"],
                        node_type=data["node_type"],
                        domain=data["domain"],
                        metadata=data.get("metadata", {}),
                        linked_images=set(data.get("linked_images", [])),
                        linked_texts=set(data.get("linked_texts", []))
                    )
                    self.text_nodes[nid] = node
                    self.text_by_type[node.node_type].add(nid)
                    self.text_by_domain[node.domain].add(nid)
            
            # Load image nodes
            image_file = self.graph_path / "image_nodes.json"
            if image_file.exists():
                with open(image_file) as f:
                    image_data = json.load(f)
                
                for nid, data in image_data.items():
                    node = ImageNode(
                        id=data["id"],
                        image_path=data["image_path"],
                        description=data["description"],
                        node_type=data["node_type"],
                        domain=data["domain"],
                        metadata=data.get("metadata", {}),
                        linked_texts=set(data.get("linked_texts", [])),
                        linked_images=set(data.get("linked_images", []))
                    )
                    self.image_nodes[nid] = node
                    self.image_by_type[node.node_type].add(nid)
                    self.image_by_domain[node.domain].add(nid)
            
            # Load links
            links_file = self.graph_path / "links.json"
            if links_file.exists():
                with open(links_file) as f:
                    links_data = json.load(f)
                
                for link_data in links_data:
                    link = CrossModalLink(
                        source_id=link_data["source"],
                        target_id=link_data["target"],
                        link_type=link_data["type"],
                        confidence=link_data["confidence"],
                        metadata=link_data.get("metadata", {})
                    )
                    self.links[(link.source_id, link.target_id)] = link
            
        except Exception as e:
            logger.warning(f"Could not load existing graph: {e}")
    
    def _serialize_text_node(self, node: TextNode) -> Dict[str, Any]:
        """Serialize text node to dict"""
        return {
            "id": node.id,
            "content": node.content,
            "node_type": node.node_type,
            "domain": node.domain,
            "metadata": node.metadata,
            "linked_images": list(node.linked_images),
            "linked_texts": list(node.linked_texts)
        }
    
    def _serialize_image_node(self, node: ImageNode) -> Dict[str, Any]:
        """Serialize image node to dict"""
        return {
            "id": node.id,
            "image_path": node.image_path,
            "description": node.description,
            "node_type": node.node_type,
            "domain": node.domain,
            "metadata": node.metadata,
            "linked_texts": list(node.linked_texts),
            "linked_images": list(node.linked_images)
        }
    
    async def add_new_knowledge(
        self,
        query: str,
        answer: str,
        confidence: float = 0.5,
        sources: Optional[List[str]] = None,
        domain: str = 'general',
        knowledge_type: str = 'research_answer',
        vision_engine=None
    ) -> str:
        """
        Add new knowledge generated from research into the knowledge graph.
        
        Args:
            query: Original user query
            answer: Synthesized answer
            confidence: Confidence in the answer (0-1)
            sources: Optional list of source references (URLs or file paths)
            domain: Knowledge domain
            knowledge_type: Type/category of the knowledge node
            vision_engine: Optional vision engine to analyze linked images automatically
        """
        node_hash = hashlib.sha1(f"{query}|{answer}".encode()).hexdigest()
        node_id = f"{knowledge_type}_{node_hash[:10]}"
        
        metadata = {
            "query": query,
            "confidence": confidence,
            "sources": sources or [],
            "created_at": datetime.now().isoformat()
        }
        
        self.add_text_node(
            node_id=node_id,
            content=answer,
            node_type=knowledge_type,
            domain=domain,
            metadata=metadata
        )
        
        # Attempt to attach visual evidence from sources
        if sources:
            for source in sources:
                image_id = await self._ensure_image_node(source, domain, vision_engine)
                if image_id:
                    self.link_text_to_image(
                        text_id=node_id,
                        image_id=image_id,
                        link_type='evidence',
                        confidence=min(1.0, confidence + 0.1),
                        metadata={"source": source}
                    )
        
        self.save_graph()
        logger.info(f"Added knowledge node: {node_id} (domain={domain}, confidence={confidence:.2f})")
        return node_id


if __name__ == "__main__":
    print("🕸️ Cross-Modal Knowledge Graph")
    print("=" * 60)
    print("\nCapabilities:")
    print("  ✅ Link text knowledge ↔ images")
    print("  ✅ Query with text → get text + images")
    print("  ✅ Query with image → get explanations + similar images")
    print("  ✅ Validate formulas appear in diagrams")
    print("  ✅ Discover hidden connections")
    print("  ✅ Bidirectional navigation")
    print("\nUsage:")
    print("  graph = VisualKnowledgeGraph()")
    print("  graph.add_text_node('formula_1', 'beam_load = ...', 'formula', 'construction')")
    print("  graph.add_image_node('diagram_1', 'beam_diagram.png', 'Beam loads', 'diagram', 'construction')")
    print("  graph.link_text_to_image('formula_1', 'diagram_1', 'illustrates')")
    print("  results = graph.query_with_text('beam load')")
