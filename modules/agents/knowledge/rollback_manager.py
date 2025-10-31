#!/usr/bin/env python3
"""
RollbackManager - Phase 7: Knowledge Quality, Validation & Lifecycle

Maintains reversible checkpoints — the "time travel" mechanism for Kalki.

Features:
- Deterministic checkpoint IDs with timestamps
- SHA256 hash integrity for state validation
- Context coupling with tags and metadata
- Automatic retention policy with pruning
- Integration hooks for orchestrator events
- Optional compressed JSON storage
- Async I/O operations
"""

import asyncio
import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

from ..base_agent import BaseAgent, AgentCapability


class RollbackManager(BaseAgent):
    """
    Manages reversible checkpoints for system state with integrity guarantees and retention policies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="RollbackManager",
            capabilities=[AgentCapability.LIFECYCLE_MANAGEMENT, AgentCapability.VALIDATION],
            description="Manages reversible checkpoints with integrity guarantees",
            config=config
        )

        # Directory structure
        self.checkpoints_dir = Path.home() / "Desktop" / "Kalki" / "vector_db" / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Index file
        self.checkpoint_index_path = self.checkpoints_dir / "checkpoint_index.json"

        # Configuration
        self.enable_compression = config.get("enable_compression", True) if config else True
        self.max_checkpoints = config.get("max_checkpoints", 100) if config else 100
        self.retention_days = config.get("retention_days", 30) if config else 30
        self.auto_prune = config.get("auto_prune", True) if config else True

        # Integration hooks
        self.pre_checkpoint_hooks: List[Callable] = []
        self.post_checkpoint_hooks: List[Callable] = []
        self.pre_rollback_hooks: List[Callable] = []
        self.post_rollback_hooks: List[Callable] = []

        # Async support check
        if not AIOFILES_AVAILABLE:
            self.logger.warning("aiofiles not available, falling back to sync operations")

    async def initialize(self) -> bool:
        """Initialize rollback manager with async support."""
        try:
            # Initialize index if it doesn't exist
            if not self.checkpoint_index_path.exists():
                await self._save_index_async({
                    "checkpoints": [],
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                })

            # Validate existing index
            index = await self._load_index_async()
            if not isinstance(index, dict) or "checkpoints" not in index:
                self.logger.warning("Invalid index format, reinitializing")
                await self._save_index_async({
                    "checkpoints": [],
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                })

            # Auto-prune old checkpoints
            if self.auto_prune:
                await self._auto_prune_checkpoints()

            self.state = "ready"
            self.logger.info("RollbackManager initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"RollbackManager initialization failed: {e}")
            self.state = "error"
            return False

    async def create_checkpoint(
        self,
        name: str,
        state: Dict[str, Any],
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_agent: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Create a checkpoint with integrity validation and rich metadata.

        Args:
            name: Human-readable checkpoint name
            state: System state to checkpoint
            tags: List of tags for categorization
            metadata: Additional metadata
            source_agent: Agent that triggered the checkpoint
            description: Human-readable description

        Returns:
            Checkpoint ID
        """
        try:
            # Run pre-checkpoint hooks
            for hook in self.pre_checkpoint_hooks:
                try:
                    await hook(name, state)
                except Exception as e:
                    self.logger.warning(f"Pre-checkpoint hook failed: {e}")

            # Generate checkpoint ID
            timestamp = datetime.now()
            checkpoint_id = f"cp_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

            # Calculate state hash for integrity
            state_str = json.dumps(state, sort_keys=True)
            state_hash = hashlib.sha256(state_str.encode()).hexdigest()

            # Create checkpoint data
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "name": name,
                "state": state,
                "state_hash": state_hash,
                "created_at": datetime.now().isoformat(),
                "tags": tags or [],
                "metadata": metadata or {},
                "source_agent": source_agent,
                "description": description,
                "size_bytes": len(state_str.encode())
            }

            # Save checkpoint
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            if self.enable_compression:
                checkpoint_file = checkpoint_file.with_suffix('.json.gz')

            await self._save_checkpoint_async(checkpoint_file, checkpoint_data)

            # Update index
            index = await self._load_index_async()
            index["checkpoints"].append({
                "checkpoint_id": checkpoint_id,
                "name": name,
                "created_at": checkpoint_data["created_at"],
                "tags": checkpoint_data["tags"],
                "source_agent": source_agent,
                "state_hash": state_hash,
                "compressed": self.enable_compression
            })

            # Maintain max checkpoints limit
            if len(index["checkpoints"]) > self.max_checkpoints:
                await self._prune_old_checkpoints(index, self.max_checkpoints)

            await self._save_index_async(index)

            # Run post-checkpoint hooks
            for hook in self.post_checkpoint_hooks:
                try:
                    await hook(checkpoint_id, checkpoint_data)
                except Exception as e:
                    self.logger.warning(f"Post-checkpoint hook failed: {e}")

            self.logger.info(f"Created checkpoint {checkpoint_id}: {name}")
            return checkpoint_id

        except Exception as e:
            self.logger.exception(f"Failed to create checkpoint '{name}': {e}")
            raise

    async def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        validate_integrity: bool = True
    ) -> Dict[str, Any]:
        """
        Rollback to a specific checkpoint with integrity validation.

        Args:
            checkpoint_id: Checkpoint to rollback to
            validate_integrity: Whether to validate state hash

        Returns:
            The checkpoint state

        Raises:
            ValueError: If checkpoint not found or corrupted
        """
        try:
            # Run pre-rollback hooks
            for hook in self.pre_rollback_hooks:
                try:
                    await hook(checkpoint_id)
                except Exception as e:
                    self.logger.warning(f"Pre-rollback hook failed: {e}")

            # Find checkpoint in index
            index = await self._load_index_async()
            checkpoint_info = None
            for cp in index["checkpoints"]:
                if cp["checkpoint_id"] == checkpoint_id:
                    checkpoint_info = cp
                    break

            if not checkpoint_info:
                raise ValueError(f"Checkpoint {checkpoint_id} not found in index")

            # Load checkpoint file
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            if checkpoint_info.get("compressed", False):
                checkpoint_file = checkpoint_file.with_suffix('.json.gz')

            if not checkpoint_file.exists():
                raise ValueError(f"Checkpoint file {checkpoint_file} not found")

            checkpoint_data = await self._load_checkpoint_async(checkpoint_file)

            # Validate checkpoint ID matches
            if checkpoint_data["checkpoint_id"] != checkpoint_id:
                raise ValueError(f"Checkpoint ID mismatch in file {checkpoint_file}")

            # Validate integrity if requested
            if validate_integrity:
                state_str = json.dumps(checkpoint_data["state"], sort_keys=True)
                calculated_hash = hashlib.sha256(state_str.encode()).hexdigest()
                if calculated_hash != checkpoint_data["state_hash"]:
                    raise ValueError(f"State integrity check failed for checkpoint {checkpoint_id}")

            # Run post-rollback hooks
            for hook in self.post_rollback_hooks:
                try:
                    await hook(checkpoint_id, checkpoint_data)
                except Exception as e:
                    self.logger.warning(f"Post-rollback hook failed: {e}")

            self.logger.info(f"Successfully rolled back to checkpoint {checkpoint_id}")
            return checkpoint_data["state"]

        except Exception as e:
            self.logger.exception(f"Failed to rollback to checkpoint {checkpoint_id}: {e}")
            raise

    async def list_checkpoints(
        self,
        tags: Optional[List[str]] = None,
        source_agent: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List checkpoints with optional filtering.

        Args:
            tags: Filter by tags (checkpoint must have all specified tags)
            source_agent: Filter by source agent
            limit: Maximum number of results

        Returns:
            List of checkpoint summaries
        """
        try:
            index = await self._load_index_async()
            checkpoints = index["checkpoints"]

            # Apply filters
            if tags:
                checkpoints = [
                    cp for cp in checkpoints
                    if all(tag in cp.get("tags", []) for tag in tags)
                ]

            if source_agent:
                checkpoints = [
                    cp for cp in checkpoints
                    if cp.get("source_agent") == source_agent
                ]

            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x["created_at"], reverse=True)

            # Apply limit
            if limit:
                checkpoints = checkpoints[:limit]

            return checkpoints

        except Exception as e:
            self.logger.exception("Failed to list checkpoints: {e}")
            return []

    async def get_checkpoint_info(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a checkpoint."""
        try:
            # Find in index
            index = await self._load_index_async()
            for cp in index["checkpoints"]:
                if cp["checkpoint_id"] == checkpoint_id:
                    return cp

            return None

        except Exception as e:
            self.logger.exception(f"Failed to get checkpoint info for {checkpoint_id}: {e}")
            return None

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint and its associated file.

        Args:
            checkpoint_id: Checkpoint to delete

        Returns:
            Success status
        """
        try:
            # Remove from index
            index = await self._load_index_async()
            original_length = len(index["checkpoints"])
            index["checkpoints"] = [
                cp for cp in index["checkpoints"]
                if cp["checkpoint_id"] != checkpoint_id
            ]

            if len(index["checkpoints"]) == original_length:
                self.logger.warning(f"Checkpoint {checkpoint_id} not found in index")
                return False

            await self._save_index_async(index)

            # Delete checkpoint file
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            elif checkpoint_file.with_suffix('.json.gz').exists():
                checkpoint_file.with_suffix('.json.gz').unlink()

            self.logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    async def cleanup_old_checkpoints(self, days_old: Optional[int] = None) -> int:
        """
        Clean up checkpoints older than specified days.

        Args:
            days_old: Days threshold (uses config default if None)

        Returns:
            Number of checkpoints deleted
        """
        try:
            threshold_days = days_old or self.retention_days
            cutoff_date = datetime.now() - timedelta(days=threshold_days)

            index = await self._load_index_async()
            checkpoints_to_delete = []

            for cp in index["checkpoints"]:
                cp_date = datetime.fromisoformat(cp["created_at"])
                if cp_date < cutoff_date:
                    checkpoints_to_delete.append(cp["checkpoint_id"])

            deleted_count = 0
            for checkpoint_id in checkpoints_to_delete:
                if await self.delete_checkpoint(checkpoint_id):
                    deleted_count += 1

            self.logger.info(f"Cleaned up {deleted_count} checkpoints older than {threshold_days} days")
            return deleted_count

        except Exception as e:
            self.logger.exception(f"Failed to cleanup old checkpoints: {e}")
            return 0

    # Integration hooks management

    def add_pre_checkpoint_hook(self, hook: Callable) -> None:
        """Add a hook to run before checkpoint creation."""
        self.pre_checkpoint_hooks.append(hook)

    def add_post_checkpoint_hook(self, hook: Callable) -> None:
        """Add a hook to run after checkpoint creation."""
        self.post_checkpoint_hooks.append(hook)

    def add_pre_rollback_hook(self, hook: Callable) -> None:
        """Add a hook to run before rollback."""
        self.pre_rollback_hooks.append(hook)

    def add_post_rollback_hook(self, hook: Callable) -> None:
        """Add a hook to run after rollback."""
        self.post_rollback_hooks.append(hook)

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback management tasks."""
        try:
            action = task.get("action")

            if action == "create":
                checkpoint_id = await self.create_checkpoint(
                    task["name"],
                    task["state"],
                    task.get("tags"),
                    task.get("metadata"),
                    task.get("source_agent"),
                    task.get("description")
                )
                return {"status": "success", "checkpoint_id": checkpoint_id}

            elif action == "rollback":
                state = await self.rollback_to_checkpoint(
                    task["checkpoint_id"],
                    task.get("validate_integrity", True)
                )
                return {"status": "success", "state": state}

            elif action == "list":
                checkpoints = await self.list_checkpoints(
                    task.get("tags"),
                    task.get("source_agent"),
                    task.get("limit")
                )
                return {"status": "success", "checkpoints": checkpoints}

            elif action == "info":
                info = await self.get_checkpoint_info(task["checkpoint_id"])
                return {"status": "success", "checkpoint_info": info}

            elif action == "delete":
                success = await self.delete_checkpoint(task["checkpoint_id"])
                return {"status": "success" if success else "error"}

            elif action == "cleanup":
                deleted_count = await self.cleanup_old_checkpoints(task.get("days_old"))
                return {"status": "success", "deleted_count": deleted_count}

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            self.logger.exception(f"Task execution failed: {e}")
            return {"status": "error", "message": str(e)}

    # Private helper methods

    async def _load_index_async(self) -> Dict[str, Any]:
        """Load checkpoint index asynchronously."""
        return await self._load_json_async(self.checkpoint_index_path, {"checkpoints": [], "metadata": {}})

    async def _save_index_async(self, index: Dict[str, Any]) -> None:
        """Save checkpoint index asynchronously."""
        await self._save_json_async(self.checkpoint_index_path, index)

    async def _load_checkpoint_async(self, file_path: Path) -> Dict[str, Any]:
        """Load checkpoint file (handles compression)."""
        if file_path.suffix == '.gz':
            # Compressed file
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(file_path, 'rb') as f:
                    compressed_data = await f.read()
                    decompressed_data = gzip.decompress(compressed_data)
                    return json.loads(decompressed_data.decode('utf-8'))
            else:
                # Fallback to sync
                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
        else:
            # Regular JSON file
            return await self._load_json_async(file_path)

    async def _save_checkpoint_async(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save checkpoint file (handles compression)."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix == '.gz':
            # Compressed file
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            compressed_data = gzip.compress(json_str.encode('utf-8'))

            if AIOFILES_AVAILABLE:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(compressed_data)
            else:
                # Fallback to sync
                with open(file_path, 'wb') as f:
                    f.write(compressed_data)
        else:
            # Regular JSON file
            await self._save_json_async(file_path, data)

    async def _load_json_async(self, file_path: Path, default: Any = None) -> Any:
        """Load JSON file asynchronously."""
        if not file_path.exists():
            return default or {}

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        else:
            # Fallback to sync
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    async def _save_json_async(self, file_path: Path, data: Any) -> None:
        """Save JSON file asynchronously."""
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            # Fallback to sync
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    async def _auto_prune_checkpoints(self) -> None:
        """Automatically prune old checkpoints based on retention policy."""
        try:
            await self.cleanup_old_checkpoints()
        except Exception as e:
            self.logger.exception(f"Auto-pruning failed: {e}")

    async def _prune_old_checkpoints(self, index: Dict[str, Any], keep_count: int) -> None:
        """Prune oldest checkpoints to maintain the specified count."""
        try:
            checkpoints = index["checkpoints"]

            # Sort by creation time (oldest first)
            checkpoints.sort(key=lambda x: x["created_at"])

            # Remove excess checkpoints
            excess_checkpoints = checkpoints[:-keep_count]
            for cp in excess_checkpoints:
                checkpoint_id = cp["checkpoint_id"]
                try:
                    await self.delete_checkpoint(checkpoint_id)
                except Exception as e:
                    self.logger.warning(f"Failed to prune checkpoint {checkpoint_id}: {e}")

        except Exception as e:
            self.logger.exception("Failed to prune old checkpoints: {e}")

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the rollback manager
        """
        try:
            self.logger.info("RollbackManager shutting down")
            # Save any pending index changes
            if hasattr(self, '_index') and self._index:
                await self._save_index_async(self._index)
            return True
        except Exception as e:
            self.logger.exception(f"Error during shutdown: {e}")
            return False