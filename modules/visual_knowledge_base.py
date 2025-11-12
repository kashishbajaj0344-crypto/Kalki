"""
Visual Knowledge Base with Similarity Search
Revolutionary diagram search using CLIP embeddings.

Capabilities:
- Find similar diagrams by uploading an image
- Search diagrams by text description
- "Show me all foundation details like this" queries
- Cross-modal search (text → images, image → images)
- Semantic understanding of construction diagrams
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json
import hashlib
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP not available - install transformers and torch")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available - install chromadb")


@dataclass
class DiagramSearchResult:
    """Result from diagram search"""
    image_path: str
    similarity_score: float
    description: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None


@dataclass
class DiagramMetadata:
    """Metadata for indexed diagram"""
    image_path: str
    description: str
    source_document: Optional[str]
    page_number: Optional[int]
    diagram_type: Optional[str]
    domain: str
    tags: List[str]
    indexed_at: str


class VisualKnowledgeBase:
    """
    Visual knowledge base with CLIP-powered similarity search.
    
    Enables revolutionary diagram search capabilities:
    - Find similar diagrams (image → images)
    - Text search for diagrams (text → images)
    - Semantic understanding of technical drawings
    """
    
    def __init__(
        self,
        db_path: str = "data/visual_knowledge",
        clip_model: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None
    ):
        """
        Initialize visual knowledge base.
        
        Args:
            db_path: Path to store ChromaDB and metadata
            clip_model: CLIP model identifier
            device: Device for CLIP (cuda/mps/cpu)
        """
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP not available - install transformers and torch")
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available - install chromadb")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        print(f"🎨 Loading CLIP model on {device}...")
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model.eval()  # Set to evaluation mode
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.db_path / "chromadb")
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="construction_diagrams",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity
        )
        
        # Metadata storage
        self.metadata_file = self.db_path / "diagram_metadata.json"
        self.metadata = self._load_metadata()
        
        print(f"✅ Visual Knowledge Base initialized ({len(self.metadata)} diagrams indexed)")
    
    def index_diagram(
        self,
        image_path: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_reindex: bool = False
    ) -> str:
        """
        Index a diagram into the visual knowledge base.
        
        Args:
            image_path: Path to diagram image
            description: Text description of diagram
            metadata: Optional metadata (source, page, type, etc.)
            force_reindex: Force re-indexing if already exists
        
        Returns:
            Diagram ID (hash of image path)
        """
        image_path = str(Path(image_path).resolve())
        
        # Generate diagram ID
        diagram_id = self._generate_diagram_id(image_path)
        
        # Check if already indexed
        if diagram_id in self.metadata and not force_reindex:
            print(f"⏭️ Diagram already indexed: {Path(image_path).name}")
            return diagram_id
        
        # Generate CLIP embeddings
        try:
            image_embedding = self._encode_image(image_path)
            text_embedding = self._encode_text(description)
            
            # Average image and text embeddings for better retrieval
            combined_embedding = (image_embedding + text_embedding) / 2
            combined_embedding = F.normalize(combined_embedding, dim=0)
            
        except Exception as e:
            print(f"❌ Error encoding {image_path}: {e}")
            return None
        
        # Prepare metadata
        diagram_metadata = DiagramMetadata(
            image_path=image_path,
            description=description,
            source_document=metadata.get("source") if metadata else None,
            page_number=metadata.get("page") if metadata else None,
            diagram_type=metadata.get("type") if metadata else None,
            domain=metadata.get("domain", "construction") if metadata else "construction",
            tags=metadata.get("tags", []) if metadata else [],
            indexed_at=datetime.now().isoformat()
        )
        
        # Store in ChromaDB
        self.collection.add(
            ids=[diagram_id],
            embeddings=[combined_embedding.cpu().numpy().tolist()],
            metadatas=[{
                "image_path": image_path,
                "description": description,
                "domain": diagram_metadata.domain,
                "indexed_at": diagram_metadata.indexed_at
            }],
            documents=[description]  # Store description as document
        )
        
        # Store metadata
        self.metadata[diagram_id] = diagram_metadata
        self._save_metadata()
        
        print(f"✅ Indexed: {Path(image_path).name}")
        
        return diagram_id
    
    def search_by_image(
        self,
        image_path: str,
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> List[DiagramSearchResult]:
        """
        Find similar diagrams by image.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            List of similar diagrams
        """
        print(f"🔍 Searching for diagrams similar to: {Path(image_path).name}")
        
        # Encode query image
        query_embedding = self._encode_image(image_path)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.cpu().numpy().tolist()],
            n_results=top_k
        )
        
        # Parse results
        search_results = []
        
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, diagram_id in enumerate(results["ids"][0]):
                similarity = 1.0 - results["distances"][0][i]  # Convert distance to similarity
                
                if similarity < min_similarity:
                    continue
                
                metadata = self.metadata.get(diagram_id)
                if not metadata:
                    continue
                
                search_results.append(DiagramSearchResult(
                    image_path=metadata.image_path,
                    similarity_score=similarity,
                    description=metadata.description,
                    metadata={
                        "source": metadata.source_document,
                        "page": metadata.page_number,
                        "type": metadata.diagram_type,
                        "domain": metadata.domain,
                        "tags": metadata.tags
                    }
                ))
        
        print(f"📊 Found {len(search_results)} similar diagrams")
        
        return search_results
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.5,
        domain_filter: Optional[str] = None
    ) -> List[DiagramSearchResult]:
        """
        Search diagrams by text description.
        
        Args:
            query: Text search query
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            domain_filter: Optional domain filter
        
        Returns:
            List of matching diagrams
        """
        print(f"🔍 Text search: '{query}'")
        
        # Encode query text
        query_embedding = self._encode_text(query)
        
        # Apply domain filter if specified
        where_filter = None
        if domain_filter:
            where_filter = {"domain": domain_filter}
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.cpu().numpy().tolist()],
            n_results=top_k,
            where=where_filter
        )
        
        # Parse results
        search_results = []
        
        if results["ids"] and len(results["ids"][0]) > 0:
            for i, diagram_id in enumerate(results["ids"][0]):
                similarity = 1.0 - results["distances"][0][i]
                
                if similarity < min_similarity:
                    continue
                
                metadata = self.metadata.get(diagram_id)
                if not metadata:
                    continue
                
                search_results.append(DiagramSearchResult(
                    image_path=metadata.image_path,
                    similarity_score=similarity,
                    description=metadata.description,
                    metadata={
                        "source": metadata.source_document,
                        "page": metadata.page_number,
                        "type": metadata.diagram_type,
                        "domain": metadata.domain,
                        "tags": metadata.tags
                    }
                ))
        
        print(f"📊 Found {len(search_results)} matching diagrams")
        
        return search_results
    
    def search_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        top_k: int = 50
    ) -> List[DiagramSearchResult]:
        """
        Search diagrams by tags.
        
        Args:
            tags: List of tags to search
            match_all: If True, diagram must have all tags
            top_k: Maximum results
        
        Returns:
            List of matching diagrams
        """
        print(f"🏷️ Searching by tags: {tags}")
        
        results = []
        
        for diagram_id, metadata in self.metadata.items():
            if match_all:
                # Must have all tags
                if all(tag in metadata.tags for tag in tags):
                    results.append(DiagramSearchResult(
                        image_path=metadata.image_path,
                        similarity_score=1.0,
                        description=metadata.description,
                        metadata={
                            "source": metadata.source_document,
                            "page": metadata.page_number,
                            "type": metadata.diagram_type,
                            "domain": metadata.domain,
                            "tags": metadata.tags
                        }
                    ))
            else:
                # Must have at least one tag
                if any(tag in metadata.tags for tag in tags):
                    # Calculate score based on tag overlap
                    overlap = len(set(tags) & set(metadata.tags))
                    score = overlap / len(tags)
                    
                    results.append(DiagramSearchResult(
                        image_path=metadata.image_path,
                        similarity_score=score,
                        description=metadata.description,
                        metadata={
                            "source": metadata.source_document,
                            "page": metadata.page_number,
                            "type": metadata.diagram_type,
                            "domain": metadata.domain,
                            "tags": metadata.tags
                        }
                    ))
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results[:top_k]
    
    def batch_index_diagrams(
        self,
        diagrams: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[str]:
        """
        Batch index multiple diagrams efficiently.
        
        Args:
            diagrams: List of dicts with keys: image_path, description, metadata
            show_progress: Show progress during indexing
        
        Returns:
            List of diagram IDs
        """
        print(f"📦 Batch indexing {len(diagrams)} diagrams...")
        
        indexed_ids = []
        
        for i, diagram in enumerate(diagrams):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(diagrams)}")
            
            try:
                diagram_id = self.index_diagram(
                    image_path=diagram["image_path"],
                    description=diagram["description"],
                    metadata=diagram.get("metadata")
                )
                
                if diagram_id:
                    indexed_ids.append(diagram_id)
                    
            except Exception as e:
                print(f"⚠️ Error indexing {diagram['image_path']}: {e}")
        
        print(f"✅ Batch indexing complete: {len(indexed_ids)}/{len(diagrams)} successful")
        
        return indexed_ids
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        domain_counts = {}
        type_counts = {}
        
        for metadata in self.metadata.values():
            # Count by domain
            domain = metadata.domain
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            # Count by type
            if metadata.diagram_type:
                dtype = metadata.diagram_type
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        return {
            "total_diagrams": len(self.metadata),
            "domains": domain_counts,
            "diagram_types": type_counts,
            "collection_size": self.collection.count(),
            "device": self.device,
            "model": "CLIP ViT-B/32"
        }
    
    # Private helper methods
    
    def _encode_image(self, image_path: str) -> torch.Tensor:
        """Encode image to CLIP embedding"""
        from PIL import Image
        
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.clip_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features.squeeze(0)
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to CLIP embedding"""
        inputs = self.clip_processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features.squeeze(0)
    
    def _generate_diagram_id(self, image_path: str) -> str:
        """Generate unique ID for diagram"""
        return hashlib.sha256(str(image_path).encode()).hexdigest()[:16]
    
    def _load_metadata(self) -> Dict[str, DiagramMetadata]:
        """Load diagram metadata from disk"""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file) as f:
                data = json.load(f)
            
            metadata = {}
            for diagram_id, meta_dict in data.items():
                metadata[diagram_id] = DiagramMetadata(**meta_dict)
            
            return metadata
            
        except Exception as e:
            print(f"⚠️ Error loading metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save diagram metadata to disk"""
        try:
            data = {}
            for diagram_id, metadata in self.metadata.items():
                data[diagram_id] = {
                    "image_path": metadata.image_path,
                    "description": metadata.description,
                    "source_document": metadata.source_document,
                    "page_number": metadata.page_number,
                    "diagram_type": metadata.diagram_type,
                    "domain": metadata.domain,
                    "tags": metadata.tags,
                    "indexed_at": metadata.indexed_at
                }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Error saving metadata: {e}")


# Convenience functions

def create_visual_knowledge_base(db_path: str = "data/visual_knowledge") -> VisualKnowledgeBase:
    """
    Create or load visual knowledge base.
    
    Args:
        db_path: Path to knowledge base
    
    Returns:
        VisualKnowledgeBase instance
    """
    return VisualKnowledgeBase(db_path=db_path)


def search_diagrams(
    query: str,
    vkb: Optional[VisualKnowledgeBase] = None,
    top_k: int = 10
) -> List[DiagramSearchResult]:
    """
    Convenience function for diagram search.
    
    Args:
        query: Text search query
        vkb: Optional VKB instance (creates new if None)
        top_k: Number of results
    
    Returns:
        List of search results
    """
    if vkb is None:
        vkb = create_visual_knowledge_base()
    
    return vkb.search_by_text(query, top_k=top_k)


if __name__ == "__main__":
    print("🎨 Visual Knowledge Base with CLIP Embeddings")
    print("=" * 60)
    
    if not CLIP_AVAILABLE:
        print("❌ CLIP not available")
        print("Install with: pip install transformers torch")
        exit(1)
    
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB not available")
        print("Install with: pip install chromadb")
        exit(1)
    
    print("\n✅ All dependencies available")
    print("\nCapabilities:")
    print("  🔍 Find similar diagrams (image → images)")
    print("  📝 Text search for diagrams (text → images)")
    print("  🏷️ Tag-based filtering")
    print("  🔄 Cross-modal search (text ↔ images)")
    print("  ⚡ Fast vector similarity with ChromaDB")
    print("  🧠 Semantic understanding with CLIP")
    print("\nUsage:")
    print("  vkb = VisualKnowledgeBase()")
    print("  vkb.index_diagram('path/to/diagram.png', 'Foundation detail')")
    print("  results = vkb.search_by_text('show me foundation footings')")
    print("  similar = vkb.search_by_image('example_diagram.png')")
