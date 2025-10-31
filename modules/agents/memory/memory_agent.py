#!/usr/bin/env python3
"""
MemoryAgent: Episodic and semantic memory storage
Phase 4: Persistent Memory & Session Management
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from cryptography.fernet import Fernet

from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ...config import ROOT


def now_ts() -> str:
    """Get current timestamp as ISO string"""
    return datetime.utcnow().isoformat()


def load_json(filepath: Path, default=None):
    """Load JSON from file with default fallback"""
    try:
        if filepath.exists():
            return json.loads(filepath.read_text())
        return default if default is not None else {}
    except Exception:
        return default if default is not None else {}


def save_json(filepath: Path, data: Dict[str, Any]):
    """Save data as JSON to file"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Failed to save JSON to {filepath}: {e}")


class MemoryAgent(BaseAgent):
    """
    Manages episodic and semantic memory storage
    - Episodic: Event-based, time-ordered memories
    - Semantic: Concept-based, structured knowledge
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="MemoryAgent",
            capabilities=[AgentCapability.MEMORY],
            description="Manages episodic and semantic memory storage with vector embeddings",
            config=config or {}
        )
        self.memory_dir = Path(ROOT) / "memory"
        self.episodic_dir = self.memory_dir / "episodic"
        self.semantic_dir = self.memory_dir / "semantic"
        self.episodic_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_index = self.memory_dir / "episodic_index.json"
        self.semantic_index = self.memory_dir / "semantic_index.json"

        # Optional encryption
        self.encryption_key = self.config.get("encryption_key") if self.config else None
        self.cipher = None
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)

        # Memory pruning settings
        self.max_episodic_memories = self.config.get("max_episodic_memories", 10000)
        self.max_semantic_memories = self.config.get("max_semantic_memories", 5000)
        self.pruning_policy = self.config.get("pruning_policy", "time_based")  # time_based, capacity_based, lru
        
        # Vector embeddings support (optional)
        self.use_embeddings = self.config.get("use_embeddings", False)
        self.embedding_model = self.config.get("embedding_model", "simple")  # simple, sentence-transformers, etc.
        
        # Initialize embeddings if requested
        if self.use_embeddings:
            self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize vector embeddings support"""
        try:
            if self.embedding_model == "sentence-transformers":
                # Optional: sentence-transformers for better embeddings
                try:
                    from sentence_transformers import SentenceTransformer
                    self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                    self.logger.info("Initialized sentence-transformers embeddings")
                except ImportError:
                    self.logger.warning("sentence-transformers not available, falling back to simple embeddings")
                    self.embedding_model = "simple"
                    self._init_simple_embeddings()
            else:
                self._init_simple_embeddings()
        except Exception as e:
            self.logger.exception(f"Failed to initialize embeddings: {e}")
            self.use_embeddings = False
    
    def _init_simple_embeddings(self):
        """Initialize simple TF-IDF style embeddings"""
        self.embedder = None  # Placeholder for simple embeddings
        self.embedding_model = "simple"
        self.logger.info("Initialized simple embeddings")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        if not self.use_embeddings:
            return []
        
        try:
            if self.embedding_model == "sentence-transformers" and self.embedder:
                return self.embedder.encode(text).tolist()
            elif self.embedding_model == "simple":
                # Simple character-based embedding
                return [hash(word) % 1000 / 1000.0 for word in text.split()[:10]]
            else:
                return []
        except Exception as e:
            self.logger.debug(f"Failed to generate embedding: {e}")
            return []

    async def initialize(self) -> bool:
        """Initialize memory storage"""
        try:
            if not self.episodic_index.exists():
                save_json(self.episodic_index, [])
            if not self.semantic_index.exists():
                save_json(self.semantic_index, {})
            self.update_status(AgentStatus.READY)
            self.logger.info("MemoryAgent initialized")
            return True
        except Exception as e:
            self.logger.exception(f"MemoryAgent initialization failed: {e}")
            self.update_status(AgentStatus.ERROR)
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory management tasks"""
        action = task.get("action")
        memory_type = task.get("type", "episodic")

        if action == "store":
            if memory_type == "episodic":
                memory_id = await self.store_episodic(task["event"])
                return {"status": "success", "memory_id": memory_id}
            elif memory_type == "semantic":
                memory_id = await self.store_semantic(task["concept"], task["knowledge"])
                return {"status": "success", "memory_id": memory_id}
        elif action == "recall":
            if memory_type == "episodic":
                memories = await self.recall_episodic(
                    limit=task.get("limit", 10),
                    start_time=task.get("start_time"),
                    end_time=task.get("end_time")
                )
                return {"status": "success", "memories": memories}
            elif memory_type == "semantic":
                memories = await self.recall_semantic(
                    task["concept"],
                    task.get("query_text"),
                    task.get("use_similarity", False)
                )
                return {"status": "success", "memories": memories}
        elif action == "prune":
            pruned_count = await self.prune_memories(memory_type)
            return {"status": "success", "pruned_count": pruned_count}
        else:
            return {"status": "error", "message": f"Unknown action or type: {action}/{memory_type}"}

    async def store_episodic(self, event: Dict[str, Any]) -> str:
        """
        Store episodic memory (event-based)
        """
        try:
            memory_id = f"episodic_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            memory_data = {
                "memory_id": memory_id,
                "timestamp": now_ts(),
                "event": event,
                "type": "episodic"
            }

            memory_file = self.episodic_dir / f"{memory_id}.json"
            await self._save_memory(memory_file, memory_data)

            # Update index
            index = load_json(self.episodic_index, [])
            index.append({
                "memory_id": memory_id,
                "timestamp": memory_data["timestamp"],
                "summary": event.get("summary", "")
            })

            # Prune if necessary
            if len(index) > self.max_episodic_memories:
                await self._prune_episodic_index()

            save_json(self.episodic_index, index)

            self.logger.debug(f"Stored episodic memory {memory_id}")
            return memory_id
        except Exception as e:
            self.logger.exception(f"Failed to store episodic memory: {e}")
            raise

    async def store_semantic(self, concept: str, knowledge: Dict[str, Any]) -> str:
        """
        Store semantic memory (concept-based knowledge)
        """
        try:
            memory_id = f"semantic_{concept}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            memory_data = {
                "memory_id": memory_id,
                "concept": concept,
                "timestamp": now_ts(),
                "knowledge": knowledge,
                "type": "semantic"
            }
            
            # Add vector embedding if enabled
            if self.use_embeddings:
                concept_text = f"{concept} {' '.join(str(v) for v in knowledge.values() if isinstance(v, str))}"
                memory_data["embedding"] = self._generate_embedding(concept_text)
            
            memory_file = self.semantic_dir / f"{memory_id}.json"
            await self._save_memory(memory_file, memory_data)            # Update index
            index = load_json(self.semantic_index, {})
            if concept not in index:
                index[concept] = []
            index[concept].append(memory_id)

            # Prune if necessary
            if len(index[concept]) > self.max_semantic_memories // 100:  # Rough limit per concept
                await self._prune_semantic_concept(concept)

            save_json(self.semantic_index, index)

            self.logger.debug(f"Stored semantic memory {memory_id} for concept '{concept}'")
            return memory_id
        except Exception as e:
            self.logger.exception(f"Failed to store semantic memory: {e}")
            raise

    async def recall_episodic(self, limit: int = 10, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recall episodic memories within time range
        """
        try:
            index = load_json(self.episodic_index, [])
            filtered = index

            if start_time:
                filtered = [m for m in filtered if m["timestamp"] >= start_time]
            if end_time:
                filtered = [m for m in filtered if m["timestamp"] <= end_time]

            # Sort by timestamp descending
            filtered.sort(key=lambda x: x["timestamp"], reverse=True)
            filtered = filtered[:limit]

            # Load full memory data
            memories = []
            for entry in filtered:
                memory_file = self.episodic_dir / f"{entry['memory_id']}.json"
                if memory_file.exists():
                    memories.append(await self._load_memory(memory_file))

            return memories
        except Exception as e:
            self.logger.exception(f"Failed to recall episodic memories: {e}")
            return []

    async def recall_semantic(self, concept: str, query_text: str = None, use_similarity: bool = False) -> List[Dict[str, Any]]:
        """
        Recall semantic memories for a concept, with optional vector similarity search
        """
        try:
            index = load_json(self.semantic_index, {})
            memory_ids = index.get(concept, [])

            memories = []
            for memory_id in memory_ids:
                memory_file = self.semantic_dir / f"{memory_id}.json"
                if memory_file.exists():
                    memories.append(await self._load_memory(memory_file))

            # If using similarity search and we have query text
            if use_similarity and query_text and self.use_embeddings:
                query_embedding = self._generate_embedding(query_text)
                if query_embedding:
                    # Sort by cosine similarity
                    memories_with_similarity = []
                    for memory in memories:
                        embedding = memory.get("embedding", [])
                        if embedding:
                            similarity = self._cosine_similarity(query_embedding, embedding)
                            memory_copy = memory.copy()
                            memory_copy["similarity"] = similarity
                            memories_with_similarity.append(memory_copy)
                    
                    memories_with_similarity.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                    memories = memories_with_similarity

            return memories
        except Exception as e:
            self.logger.exception(f"Failed to recall semantic memories: {e}")
            return []

    async def prune_memories(self, memory_type: str) -> int:
        """Prune memories based on policy"""
        if memory_type == "episodic":
            return await self._prune_episodic_index()
        elif memory_type == "semantic":
            return await self._prune_semantic_index()
        else:
            return 0

    async def _prune_episodic_index(self) -> int:
        """Prune episodic memories based on policy"""
        try:
            index = load_json(self.episodic_index, [])
            if len(index) <= self.max_episodic_memories:
                return 0

            if self.pruning_policy == "time_based":
                # Keep most recent memories
                index.sort(key=lambda x: x["timestamp"], reverse=True)
                pruned = index[self.max_episodic_memories:]
                index = index[:self.max_episodic_memories]
            elif self.pruning_policy == "capacity_based":
                # Remove oldest memories
                index.sort(key=lambda x: x["timestamp"])
                pruned = index[:-self.max_episodic_memories]
                index = index[-self.max_episodic_memories:]
            else:
                return 0

            # Remove pruned memory files
            for entry in pruned:
                memory_file = self.episodic_dir / f"{entry['memory_id']}.json"
                if memory_file.exists():
                    memory_file.unlink()

            save_json(self.episodic_index, index)
            self.logger.info(f"Pruned {len(pruned)} episodic memories")
            return len(pruned)
        except Exception as e:
            self.logger.exception("Failed to prune episodic memories")
            return 0

    async def _prune_semantic_concept(self, concept: str) -> int:
        """Prune semantic memories for a specific concept"""
        try:
            index = load_json(self.semantic_index, {})
            memory_ids = index.get(concept, [])
            if len(memory_ids) <= 10:  # Keep at least 10 per concept
                return 0

            # Keep most recent
            memory_ids.sort(reverse=True)
            pruned_ids = memory_ids[10:]
            index[concept] = memory_ids[:10]

            # Remove pruned files
            for memory_id in pruned_ids:
                memory_file = self.semantic_dir / f"{memory_id}.json"
                if memory_file.exists():
                    memory_file.unlink()

            save_json(self.semantic_index, index)
            return len(pruned_ids)
        except Exception as e:
            self.logger.exception(f"Failed to prune semantic memories for concept {concept}")
            return 0

    async def _prune_semantic_index(self) -> int:
        """Prune all semantic memories"""
        try:
            index = load_json(self.semantic_index, {})
            total_pruned = 0
            for concept in list(index.keys()):
                total_pruned += await self._prune_semantic_concept(concept)
            return total_pruned
        except Exception as e:
            self.logger.exception("Failed to prune semantic index")
            return 0

    async def _save_memory(self, filepath: Path, data: Dict[str, Any]):
        """Save memory data with optional encryption"""
        if self.cipher:
            encrypted_data = self.cipher.encrypt(json.dumps(data).encode())
            filepath.write_bytes(encrypted_data)
        else:
            save_json(filepath, data)

    async def _load_memory(self, filepath: Path) -> Dict[str, Any]:
        """Load memory data with optional decryption"""
        if self.cipher:
            encrypted_data = filepath.read_bytes()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        else:
            return load_json(filepath, {})

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    async def shutdown(self) -> bool:
        """Shutdown the memory agent"""
        self.logger.info("MemoryAgent shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True