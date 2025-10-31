#!/usr/bin/env python3
"""
KnowledgeLifecycleAgent - Phase 7: Knowledge Quality, Validation & Lifecycle

Maintains knowledge object lineage (versioning → archival → obsolescence).

Features:
- Dedicated folders for versions and archive
- Central version_index.json with normalized structure
- SHA256 checksums for integrity
- Enhanced metadata (source_agent, validation_score, dependencies)
- JSON schema validation
- Optional compression for large archives
- Async I/O operations
"""

import asyncio
import hashlib
import json
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

from ..base_agent import BaseAgent


class KnowledgeLifecycleAgent(BaseAgent):
    """
    Manages knowledge versioning, updates, archival, and obsolescence with integrity guarantees.
    """

    # JSON Schema for version validation
    VERSION_SCHEMA = {
        "type": "object",
        "required": ["version_id", "knowledge_id", "version_num", "content", "metadata", "created_at"],
        "properties": {
            "version_id": {"type": "string"},
            "knowledge_id": {"type": "string"},
            "version_num": {"type": "integer", "minimum": 1},
            "content": {"type": "object"},
            "metadata": {
                "type": "object",
                "properties": {
                    "source_agent": {"type": "string"},
                    "validation_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"}
                }
            },
            "created_at": {"type": "string"},
            "status": {"type": "string", "enum": ["active", "archived", "deprecated", "experimental"]},
            "checksum": {"type": "string"},
            "size_bytes": {"type": "integer"}
        }
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="KnowledgeLifecycleAgent", config=config)

        # Directory structure
        self.knowledge_dir = Path.home() / "Desktop" / "Kalki" / "vector_db" / "knowledge_lifecycle"
        self.versions_dir = self.knowledge_dir / "versions"
        self.archive_dir = self.knowledge_dir / "archive"
        self.temp_dir = self.knowledge_dir / "temp"

        # Create directories
        for dir_path in [self.versions_dir, self.archive_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Index file
        self.version_index_path = self.knowledge_dir / "version_index.json"

        # Configuration
        self.enable_compression = config.get("enable_compression", True) if config else True
        self.max_versions_per_knowledge = config.get("max_versions_per_knowledge", 50) if config else 50
        self.auto_archive_days = config.get("auto_archive_days", 90) if config else 90

        # Async support check
        if not AIOFILES_AVAILABLE:
            self.logger.warning("aiofiles not available, falling back to sync operations")

    async def initialize(self) -> bool:
        """Initialize knowledge lifecycle management with async support."""
        try:
            # Initialize index if it doesn't exist
            if not self.version_index_path.exists():
                await self._save_index_async({})

            # Validate existing index
            index = await self._load_index_async()
            if not isinstance(index, dict):
                self.logger.warning("Invalid index format, reinitializing")
                await self._save_index_async({})

            self.state = "ready"
            self.logger.info("KnowledgeLifecycleAgent initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"KnowledgeLifecycleAgent initialization failed: {e}")
            self.state = "error"
            return False

    async def create_version(
        self,
        knowledge_id: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        source_agent: Optional[str] = None,
        validation_score: Optional[float] = None,
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Create a new version of knowledge with integrity checks and enhanced metadata.

        Args:
            knowledge_id: Unique identifier for the knowledge item
            content: The knowledge content
            metadata: Additional metadata
            source_agent: Agent that created this version
            validation_score: Confidence/validation score (0.0-1.0)
            dependencies: List of knowledge IDs this version depends on

        Returns:
            The version ID of the created version
        """
        try:
            # Load current index
            index = await self._load_index_async()

            # Initialize knowledge entry if new
            if knowledge_id not in index:
                index[knowledge_id] = {
                    "versions": [],
                    "current": None,
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                    "last_modified": datetime.now().isoformat()
                }

            # Check if knowledge is archived
            if index[knowledge_id].get("status") == "archived":
                raise ValueError(f"Cannot create version for archived knowledge: {knowledge_id}")

            # Generate version number
            version_num = len(index[knowledge_id]["versions"]) + 1
            version_id = f"{knowledge_id}_v{version_num}"

            # Enhanced metadata
            full_metadata = metadata or {}
            if source_agent:
                full_metadata["source_agent"] = source_agent
            if validation_score is not None:
                full_metadata["validation_score"] = validation_score
            if dependencies:
                full_metadata["dependencies"] = dependencies

            # Create version data
            version_data = {
                "version_id": version_id,
                "knowledge_id": knowledge_id,
                "version_num": version_num,
                "content": content,
                "metadata": full_metadata,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }

            # Calculate checksum and size
            content_str = json.dumps(version_data, sort_keys=True)
            version_data["checksum"] = hashlib.sha256(content_str.encode()).hexdigest()
            version_data["size_bytes"] = len(content_str.encode())

            # Validate against schema
            if JSONSCHEMA_AVAILABLE:
                jsonschema.validate(instance=version_data, schema=self.VERSION_SCHEMA)

            # Save version file
            version_file = self.versions_dir / f"{version_id}.json"
            await self._save_json_async(version_file, version_data)

            # Update index
            index[knowledge_id]["versions"].append({
                "id": version_id,
                "num": version_num,
                "created_at": version_data["created_at"],
                "checksum": version_data["checksum"]
            })
            index[knowledge_id]["current"] = version_id
            index[knowledge_id]["last_modified"] = now_ts()

            # Auto-archive old versions if limit exceeded
            if len(index[knowledge_id]["versions"]) > self.max_versions_per_knowledge:
                await self._auto_archive_old_versions(knowledge_id, index)

            await self._save_index_async(index)

            self.logger.info(f"Created version {version_id} for knowledge {knowledge_id}")
            return version_id

        except Exception as e:
            self.logger.exception(f"Failed to create version for {knowledge_id}: {e}")
            raise

    async def get_version(self, version_id: str, validate_integrity: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific version with optional integrity validation.

        Args:
            version_id: The version ID to retrieve
            validate_integrity: Whether to validate checksum

        Returns:
            Version data or None if not found
        """
        try:
            version_file = self.versions_dir / f"{version_id}.json"

            # Check archive if not in versions
            if not version_file.exists():
                version_file = self.archive_dir / f"{version_id}.json"
                if not version_file.exists():
                    return None

            version_data = await self._load_json_async(version_file)

            # Validate integrity if requested
            if validate_integrity and "checksum" in version_data:
                content_str = json.dumps({
                    k: v for k, v in version_data.items()
                    if k != "checksum" and k != "size_bytes"
                }, sort_keys=True)

                calculated_checksum = hashlib.sha256(content_str.encode()).hexdigest()
                if calculated_checksum != version_data["checksum"]:
                    self.logger.error(f"Checksum mismatch for version {version_id}")
                    return None

            return version_data

        except Exception as e:
            self.logger.exception(f"Failed to get version {version_id}: {e}")
            return None

    async def get_current_version(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get the current (latest) version of knowledge."""
        try:
            index = await self._load_index_async()
            if knowledge_id not in index:
                return None

            current_version_id = index[knowledge_id].get("current")
            if not current_version_id:
                return None

            return await self.get_version(current_version_id)

        except Exception as e:
            self.logger.exception(f"Failed to get current version for {knowledge_id}: {e}")
            return None

    async def archive_knowledge(
        self,
        knowledge_id: str,
        compression: bool = None
    ) -> bool:
        """
        Archive knowledge with optional compression.

        Args:
            knowledge_id: Knowledge to archive
            compression: Override default compression setting

        Returns:
            Success status
        """
        try:
            index = await self._load_index_async()
            if knowledge_id not in index:
                raise ValueError(f"Knowledge {knowledge_id} not found")

            use_compression = compression if compression is not None else self.enable_compression

            if use_compression:
                # Create compressed archive
                archive_name = f"{knowledge_id}_archive_{now_ts().replace(':', '-')}.zip"
                archive_path = self.archive_dir / archive_name

                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for version_info in index[knowledge_id]["versions"]:
                        version_id = version_info["id"]
                        version_file = self.versions_dir / f"{version_id}.json"
                        if version_file.exists():
                            zipf.write(version_file, f"{version_id}.json")
                            # Remove original file
                            version_file.unlink()
            else:
                # Move files individually
                for version_info in index[knowledge_id]["versions"]:
                    version_id = version_info["id"]
                    version_file = self.versions_dir / f"{version_id}.json"
                    if version_file.exists():
                        archive_file = self.archive_dir / f"{version_id}.json"
                        shutil.move(str(version_file), str(archive_file))

            # Update index
            index[knowledge_id]["status"] = "archived"
            index[knowledge_id]["archived_at"] = now_ts()
            index[knowledge_id]["archive_type"] = "compressed" if use_compression else "individual"

            await self._save_index_async(index)

            self.logger.info(f"Archived knowledge {knowledge_id} ({'compressed' if use_compression else 'individual'})")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to archive knowledge {knowledge_id}: {e}")
            return False

    async def list_versions(self, knowledge_id: str) -> List[Dict[str, Any]]:
        """List all versions of a knowledge item."""
        try:
            index = await self._load_index_async()
            if knowledge_id not in index:
                return []

            return index[knowledge_id]["versions"]

        except Exception as e:
            self.logger.exception(f"Failed to list versions for {knowledge_id}: {e}")
            return []

    async def get_knowledge_status(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a knowledge item."""
        try:
            index = await self._load_index_async()
            return index.get(knowledge_id)

        except Exception as e:
            self.logger.exception(f"Failed to get status for {knowledge_id}: {e}")
            return None

    async def cleanup_old_versions(self, days_old: int = None) -> int:
        """
        Clean up versions older than specified days.

        Args:
            days_old: Days threshold (uses config default if None)

        Returns:
            Number of versions archived
        """
        try:
            threshold_days = days_old or self.auto_archive_days
            cutoff_date = datetime.now() - timedelta(days=threshold_days)

            index = await self._load_index_async()
            archived_count = 0

            for knowledge_id, knowledge_info in index.items():
                if knowledge_info.get("status") == "archived":
                    continue

                versions_to_archive = []
                for version_info in knowledge_info["versions"]:
                    version_date = datetime.fromisoformat(version_info["created_at"])
                    if version_date < cutoff_date:
                        versions_to_archive.append(version_info)

                if versions_to_archive:
                    # Create archive for old versions
                    await self._archive_specific_versions(knowledge_id, versions_to_archive)
                    archived_count += len(versions_to_archive)

            self.logger.info(f"Archived {archived_count} versions older than {threshold_days} days")
            return archived_count

        except Exception as e:
            self.logger.exception(f"Failed to cleanup old versions: {e}")
            return 0

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge lifecycle tasks."""
        try:
            action = task.get("action")

            if action == "create_version":
                version_id = await self.create_version(
                    task["knowledge_id"],
                    task["content"],
                    task.get("metadata"),
                    task.get("source_agent"),
                    task.get("validation_score"),
                    task.get("dependencies")
                )
                return {"status": "success", "version_id": version_id}

            elif action == "get_version":
                version = await self.get_version(
                    task["version_id"],
                    task.get("validate_integrity", True)
                )
                return {"status": "success", "version": version}

            elif action == "get_current":
                version = await self.get_current_version(task["knowledge_id"])
                return {"status": "success", "version": version}

            elif action == "archive":
                success = await self.archive_knowledge(
                    task["knowledge_id"],
                    task.get("compression")
                )
                return {"status": "success" if success else "error"}

            elif action == "list_versions":
                versions = await self.list_versions(task["knowledge_id"])
                return {"status": "success", "versions": versions}

            elif action == "get_status":
                status = await self.get_knowledge_status(task["knowledge_id"])
                return {"status": "success", "knowledge_status": status}

            elif action == "cleanup":
                archived_count = await self.cleanup_old_versions(task.get("days_old"))
                return {"status": "success", "archived_count": archived_count}

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            self.logger.exception(f"Task execution failed: {e}")
            return {"status": "error", "message": str(e)}

    # Private helper methods

    async def _load_index_async(self) -> Dict[str, Any]:
        """Load version index asynchronously."""
        return await self._load_json_async(self.version_index_path, {})

    async def _save_index_async(self, index: Dict[str, Any]) -> None:
        """Save version index asynchronously."""
        await self._save_json_async(self.version_index_path, index)

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
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if AIOFILES_AVAILABLE:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            # Fallback to sync
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    async def _auto_archive_old_versions(self, knowledge_id: str, index: Dict[str, Any]) -> None:
        """Automatically archive old versions when limit is exceeded."""
        try:
            versions = index[knowledge_id]["versions"]
            if len(versions) <= self.max_versions_per_knowledge:
                return

            # Keep only the most recent versions
            versions_to_archive = versions[:-self.max_versions_per_knowledge]
            await self._archive_specific_versions(knowledge_id, versions_to_archive)

            # Update index
            index[knowledge_id]["versions"] = versions[-self.max_versions_per_knowledge:]

        except Exception as e:
            self.logger.exception(f"Failed to auto-archive old versions for {knowledge_id}: {e}")

    async def _archive_specific_versions(self, knowledge_id: str, versions_to_archive: List[Dict[str, Any]]) -> None:
        """Archive specific versions."""
        try:
            archive_name = f"{knowledge_id}_auto_archive_{now_ts().replace(':', '-')}.zip"
            archive_path = self.archive_dir / archive_name

            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for version_info in versions_to_archive:
                    version_id = version_info["id"]
                    version_file = self.versions_dir / f"{version_id}.json"
                    if version_file.exists():
                        zipf.write(version_file, f"{version_id}.json")
                        version_file.unlink()

            self.logger.info(f"Auto-archived {len(versions_to_archive)} versions for {knowledge_id}")

        except Exception as e:
            self.logger.exception(f"Failed to archive specific versions for {knowledge_id}: {e}")