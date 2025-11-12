"""
Advanced Memory System
Long-term memory with intelligent retrieval for Kalki.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import hashlib

from modules.llm import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    """Episodic memory - remembers specific events, projects, conversations"""
    memory_id: str
    episode_type: str  # "project", "conversation", "task", "interaction"
    content: Dict[str, Any]
    timestamp: datetime
    domain: str = "general"
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5  # 0-1, how important to remember
    access_count: int = 0
    last_accessed: Optional[datetime] = None


@dataclass
class SemanticMemory:
    """Semantic memory - remembers concepts, patterns, knowledge"""
    memory_id: str
    concept: str
    knowledge: Dict[str, Any]
    domain: str = "general"
    related_concepts: List[str] = field(default_factory=list)
    confidence: float = 0.8
    learned_at: datetime = field(default_factory=datetime.now)
    times_applied: int = 0


@dataclass
class ProceduralMemory:
    """Procedural memory - remembers how to do things"""
    memory_id: str
    procedure_name: str
    steps: List[Dict[str, Any]]
    domain: str = "general"
    success_rate: float = 1.0
    times_executed: int = 0
    last_executed: Optional[datetime] = None


class AdvancedMemorySystem:
    """
    Advanced memory system with:
    - Episodic memory: Specific events, projects, conversations
    - Semantic memory: Concepts, patterns, knowledge
    - Procedural memory: How to do things
    - Intelligent retrieval: Context-aware memory recall
    - Memory consolidation: Merge related memories
    """
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        
        # Memory stores
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        self.procedural_memories: Dict[str, ProceduralMemory] = {}
        
        # Memory persistence
        self.memory_dir = Path("memory/advanced")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persisted memories
        self._load_memories()
    
    async def store_episode(
        self,
        episode_type: str,
        content: Dict[str, Any],
        domain: str = "general",
        importance: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """
        Store episodic memory.
        
        Args:
            episode_type: Type of episode ("project", "conversation", etc.)
            content: Episode content
            domain: Domain context
            importance: Importance score (0-1)
            tags: Tags for retrieval
        
        Returns:
            Memory ID
        """
        memory_id = hashlib.md5(
            f"{episode_type}_{content}_{datetime.now()}".encode()
        ).hexdigest()[:16]
        
        memory = EpisodicMemory(
            memory_id=memory_id,
            episode_type=episode_type,
            content=content,
            timestamp=datetime.now(),
            domain=domain,
            tags=tags or [],
            importance=importance
        )
        
        self.episodic_memories[memory_id] = memory
        await self._persist_memory("episodic", memory)
        
        logger.info(f"💾 Stored episodic memory: {episode_type} ({memory_id})")
        return memory_id
    
    async def store_semantic(
        self,
        concept: str,
        knowledge: Dict[str, Any],
        domain: str = "general",
        related_concepts: List[str] = None,
        confidence: float = 0.8
    ) -> str:
        """
        Store semantic memory.
        
        Args:
            concept: Concept name
            knowledge: Knowledge about the concept
            domain: Domain context
            related_concepts: Related concepts
            confidence: Confidence in knowledge
        
        Returns:
            Memory ID
        """
        memory_id = hashlib.md5(f"semantic_{concept}_{domain}".encode()).hexdigest()[:16]
        
        # Check if concept already exists
        if memory_id in self.semantic_memories:
            # Update existing
            existing = self.semantic_memories[memory_id]
            existing.knowledge.update(knowledge)
            existing.confidence = (existing.confidence + confidence) / 2
            existing.related_concepts.extend(related_concepts or [])
            existing.related_concepts = list(set(existing.related_concepts))
        else:
            memory = SemanticMemory(
                memory_id=memory_id,
                concept=concept,
                knowledge=knowledge,
                domain=domain,
                related_concepts=related_concepts or [],
                confidence=confidence
            )
            self.semantic_memories[memory_id] = memory
        
        await self._persist_memory("semantic", self.semantic_memories[memory_id])
        
        logger.info(f"💾 Stored semantic memory: {concept} ({memory_id})")
        return memory_id
    
    async def store_procedure(
        self,
        procedure_name: str,
        steps: List[Dict[str, Any]],
        domain: str = "general"
    ) -> str:
        """
        Store procedural memory.
        
        Args:
            procedure_name: Name of procedure
            steps: Procedure steps
            domain: Domain context
        
        Returns:
            Memory ID
        """
        memory_id = hashlib.md5(f"procedure_{procedure_name}_{domain}".encode()).hexdigest()[:16]
        
        memory = ProceduralMemory(
            memory_id=memory_id,
            procedure_name=procedure_name,
            steps=steps,
            domain=domain
        )
        
        self.procedural_memories[memory_id] = memory
        await self._persist_memory("procedural", memory)
        
        logger.info(f"💾 Stored procedural memory: {procedure_name} ({memory_id})")
        return memory_id
    
    async def retrieve_relevant_memories(
        self,
        query: str,
        context: Dict[str, Any],
        memory_types: List[str] = None,
        limit: int = 10
    ) -> Dict[str, List[Any]]:
        """
        Retrieve relevant memories for current task.
        
        Args:
            query: Query or task description
            context: Context dictionary
            memory_types: Types to retrieve (["episodic", "semantic", "procedural"])
            limit: Max memories per type
        
        Returns:
            Dict with memory types as keys, lists of memories as values
        """
        if memory_types is None:
            memory_types = ["episodic", "semantic", "procedural"]
        
        domain = context.get("domain", "general")
        results = {mt: [] for mt in memory_types}
        
        # Use LLM to determine relevance
        relevance_prompt = f"""Given this query: {query}

Context: {json.dumps(context, indent=2)}

What concepts, episodes, or procedures are relevant? List keywords."""
        
        relevance_response = await self.llm_engine.generate(
            prompt=relevance_prompt,
            context=context,
            task="memory_retrieval"
        )
        
        keywords = self._extract_keywords(str(relevance_response))
        
        # Retrieve episodic memories
        if "episodic" in memory_types:
            for memory in self.episodic_memories.values():
                if memory.domain == domain or domain == "general":
                    # Check relevance
                    relevance = self._calculate_relevance(memory, query, keywords, context)
                    if relevance > 0.3:  # Threshold
                        results["episodic"].append((memory, relevance))
            
            # Sort by relevance and importance
            results["episodic"].sort(key=lambda x: x[1] * x[0].importance, reverse=True)
            results["episodic"] = [m[0] for m in results["episodic"][:limit]]
        
        # Retrieve semantic memories
        if "semantic" in memory_types:
            for memory in self.semantic_memories.values():
                if memory.domain == domain or domain == "general":
                    relevance = self._calculate_relevance(memory, query, keywords, context)
                    if relevance > 0.3:
                        results["semantic"].append((memory, relevance))
            
            results["semantic"].sort(key=lambda x: x[1] * x[0].confidence, reverse=True)
            results["semantic"] = [m[0] for m in results["semantic"][:limit]]
        
        # Retrieve procedural memories
        if "procedural" in memory_types:
            for memory in self.procedural_memories.values():
                if memory.domain == domain or domain == "general":
                    relevance = self._calculate_relevance(memory, query, keywords, context)
                    if relevance > 0.3:
                        results["procedural"].append((memory, relevance))
            
            results["procedural"].sort(key=lambda x: x[1] * x[0].success_rate, reverse=True)
            results["procedural"] = [m[0] for m in results["procedural"][:limit]]
        
        # Update access counts
        for memory_type, memories in results.items():
            for memory in memories:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
        
        logger.info(f"🔍 Retrieved {sum(len(m) for m in results.values())} relevant memories")
        return results
    
    async def consolidate_memories(self, domain: str = None):
        """
        Merge and consolidate related memories.
        
        Args:
            domain: Domain to consolidate (None for all)
        """
        logger.info(f"🔄 Consolidating memories for {domain or 'all domains'}")
        
        # Consolidate semantic memories
        semantic_to_merge = {}
        for memory in self.semantic_memories.values():
            if domain and memory.domain != domain:
                continue
            
            # Find similar concepts
            for other_id, other_memory in self.semantic_memories.items():
                if memory.memory_id == other_id:
                    continue
                if memory.domain != other_memory.domain:
                    continue
                
                # Check similarity
                similarity = self._concept_similarity(memory.concept, other_memory.concept)
                if similarity > 0.7:
                    # Merge
                    memory.knowledge.update(other_memory.knowledge)
                    memory.related_concepts.extend(other_memory.related_concepts)
                    memory.related_concepts = list(set(memory.related_concepts))
                    semantic_to_merge[other_id] = memory.memory_id
        
        # Remove merged memories
        for merged_id, target_id in semantic_to_merge.items():
            if merged_id in self.semantic_memories:
                del self.semantic_memories[merged_id]
        
        logger.info(f"✅ Consolidated {len(semantic_to_merge)} semantic memories")
    
    # Helper methods
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        words = text.lower().split()
        # Filter common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return list(set(keywords))[:10]  # Top 10 unique keywords
    
    def _calculate_relevance(
        self,
        memory: Any,
        query: str,
        keywords: List[str],
        context: Dict[str, Any]
    ) -> float:
        """Calculate relevance score for memory"""
        query_lower = query.lower()
        
        # Check tags (for episodic)
        if hasattr(memory, 'tags'):
            tag_matches = sum(1 for tag in memory.tags if tag.lower() in query_lower)
            if tag_matches > 0:
                return 0.8
        
        # Check concept (for semantic)
        if hasattr(memory, 'concept'):
            if memory.concept.lower() in query_lower:
                return 0.9
            if any(kw in memory.concept.lower() for kw in keywords):
                return 0.7
        
        # Check procedure name (for procedural)
        if hasattr(memory, 'procedure_name'):
            if memory.procedure_name.lower() in query_lower:
                return 0.9
        
        # Check content
        content_str = json.dumps(memory.content if hasattr(memory, 'content') else memory.knowledge if hasattr(memory, 'knowledge') else {})
        content_lower = content_str.lower()
        
        keyword_matches = sum(1 for kw in keywords if kw in content_lower)
        return min(0.6, keyword_matches * 0.1)
    
    def _concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate similarity between concepts"""
        c1_lower = concept1.lower()
        c2_lower = concept2.lower()
        
        if c1_lower == c2_lower:
            return 1.0
        
        # Check if one contains the other
        if c1_lower in c2_lower or c2_lower in c1_lower:
            return 0.8
        
        # Check word overlap
        words1 = set(c1_lower.split())
        words2 = set(c2_lower.split())
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            return overlap
        
        return 0.0
    
    async def _persist_memory(self, memory_type: str, memory: Any):
        """Persist memory to disk"""
        try:
            memory_file = self.memory_dir / f"{memory_type}_{memory.memory_id}.json"
            memory_dict = {
                "memory_id": memory.memory_id,
                "type": memory_type,
                "data": self._memory_to_dict(memory),
                "timestamp": memory.timestamp.isoformat() if hasattr(memory, 'timestamp') else datetime.now().isoformat()
            }
            with open(memory_file, 'w') as f:
                json.dump(memory_dict, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to persist memory: {e}")
    
    def _memory_to_dict(self, memory: Any) -> Dict[str, Any]:
        """Convert memory object to dict"""
        if isinstance(memory, EpisodicMemory):
            return {
                "episode_type": memory.episode_type,
                "content": memory.content,
                "domain": memory.domain,
                "tags": memory.tags,
                "importance": memory.importance
            }
        elif isinstance(memory, SemanticMemory):
            return {
                "concept": memory.concept,
                "knowledge": memory.knowledge,
                "domain": memory.domain,
                "related_concepts": memory.related_concepts,
                "confidence": memory.confidence
            }
        elif isinstance(memory, ProceduralMemory):
            return {
                "procedure_name": memory.procedure_name,
                "steps": memory.steps,
                "domain": memory.domain,
                "success_rate": memory.success_rate
            }
        return {}
    
    def _load_memories(self):
        """Load persisted memories from disk"""
        try:
            for memory_file in self.memory_dir.glob("*.json"):
                with open(memory_file) as f:
                    data = json.load(f)
                
                memory_type = data.get("type")
                memory_data = data.get("data", {})
                memory_id = data.get("memory_id")
                
                if memory_type == "episodic":
                    memory = EpisodicMemory(
                        memory_id=memory_id,
                        episode_type=memory_data.get("episode_type", ""),
                        content=memory_data.get("content", {}),
                        timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
                        domain=memory_data.get("domain", "general"),
                        tags=memory_data.get("tags", []),
                        importance=memory_data.get("importance", 0.5)
                    )
                    self.episodic_memories[memory_id] = memory
                elif memory_type == "semantic":
                    memory = SemanticMemory(
                        memory_id=memory_id,
                        concept=memory_data.get("concept", ""),
                        knowledge=memory_data.get("knowledge", {}),
                        domain=memory_data.get("domain", "general"),
                        related_concepts=memory_data.get("related_concepts", []),
                        confidence=memory_data.get("confidence", 0.8)
                    )
                    self.semantic_memories[memory_id] = memory
                elif memory_type == "procedural":
                    memory = ProceduralMemory(
                        memory_id=memory_id,
                        procedure_name=memory_data.get("procedure_name", ""),
                        steps=memory_data.get("steps", []),
                        domain=memory_data.get("domain", "general"),
                        success_rate=memory_data.get("success_rate", 1.0)
                    )
                    self.procedural_memories[memory_id] = memory
            
            logger.info(f"✅ Loaded {len(self.episodic_memories)} episodic, {len(self.semantic_memories)} semantic, {len(self.procedural_memories)} procedural memories")
        except Exception as e:
            logger.warning(f"Failed to load memories: {e}")

