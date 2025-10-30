#!/usr/bin/env python3
"""
Phase 7: Knowledge Quality, Validation & Lifecycle
- KnowledgeLifecycleAgent: Versioning, archival, obsolescence
- RollbackManager: Rollback and checkpointing support
"""
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from ..config import ROOT_DIR
from ..utils import now_ts, load_json, save_json

logger = logging.getLogger("kalki.agents.phase7")


class KnowledgeLifecycleAgent(BaseAgent):
    """
    Manages knowledge versioning, updates, archival, and obsolescence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="KnowledgeLifecycleAgent", config=config)
        self.knowledge_dir = Path(ROOT_DIR) / "knowledge_lifecycle"
        self.versions_dir = self.knowledge_dir / "versions"
        self.archive_dir = self.knowledge_dir / "archive"
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.version_index = self.knowledge_dir / "version_index.json"
    
    def initialize(self) -> bool:
        """Initialize knowledge lifecycle management"""
        try:
            if not self.version_index.exists():
                save_json(self.version_index, {})
            self.state = "ready"
            self.logger.info("KnowledgeLifecycleAgent initialized")
            return True
        except Exception as e:
            self.logger.exception(f"KnowledgeLifecycleAgent initialization failed: {e}")
            self.state = "error"
            return False
    
    def create_version(self, knowledge_id: str, content: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new version of knowledge"""
        try:
            index = load_json(self.version_index, {})
            
            if knowledge_id not in index:
                index[knowledge_id] = {"versions": [], "current_version": 0}
            
            version_num = len(index[knowledge_id]["versions"]) + 1
            version_id = f"{knowledge_id}_v{version_num}"
            
            version_data = {
                "version_id": version_id,
                "knowledge_id": knowledge_id,
                "version_num": version_num,
                "content": content,
                "metadata": metadata or {},
                "created_at": now_ts(),
                "status": "active"
            }
            
            version_file = self.versions_dir / f"{version_id}.json"
            save_json(version_file, version_data)
            
            index[knowledge_id]["versions"].append(version_id)
            index[knowledge_id]["current_version"] = version_num
            save_json(self.version_index, index)
            
            self.logger.info(f"Created version {version_id}")
            return version_id
        except Exception as e:
            self.logger.exception(f"Failed to create version: {e}")
            raise
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific version"""
        try:
            version_file = self.versions_dir / f"{version_id}.json"
            if not version_file.exists():
                return None
            return load_json(version_file)
        except Exception as e:
            self.logger.exception(f"Failed to get version: {e}")
            return None
    
    def archive_knowledge(self, knowledge_id: str):
        """Archive obsolete knowledge"""
        try:
            index = load_json(self.version_index, {})
            if knowledge_id not in index:
                raise ValueError(f"Knowledge {knowledge_id} not found")
            
            # Move all versions to archive
            for version_id in index[knowledge_id]["versions"]:
                version_file = self.versions_dir / f"{version_id}.json"
                if version_file.exists():
                    archive_file = self.archive_dir / f"{version_id}.json"
                    shutil.move(str(version_file), str(archive_file))
            
            index[knowledge_id]["status"] = "archived"
            index[knowledge_id]["archived_at"] = now_ts()
            save_json(self.version_index, index)
            
            self.logger.info(f"Archived knowledge {knowledge_id}")
        except Exception as e:
            self.logger.exception(f"Failed to archive knowledge: {e}")
            raise
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge lifecycle tasks"""
        action = task.get("action")
        
        if action == "create_version":
            version_id = self.create_version(
                task["knowledge_id"],
                task["content"],
                task.get("metadata")
            )
            return {"status": "success", "version_id": version_id}
        elif action == "get_version":
            version = self.get_version(task["version_id"])
            return {"status": "success", "version": version}
        elif action == "archive":
            self.archive_knowledge(task["knowledge_id"])
            return {"status": "success"}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class RollbackManager(BaseAgent):
    """
    Manages rollback and checkpointing for experiments and data changes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="RollbackManager", config=config)
        self.checkpoints_dir = Path(ROOT_DIR) / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_index = self.checkpoints_dir / "checkpoint_index.json"
    
    def initialize(self) -> bool:
        """Initialize rollback manager"""
        try:
            if not self.checkpoint_index.exists():
                save_json(self.checkpoint_index, {"checkpoints": []})
            self.state = "ready"
            self.logger.info("RollbackManager initialized")
            return True
        except Exception as e:
            self.logger.exception(f"RollbackManager initialization failed: {e}")
            self.state = "error"
            return False
    
    def create_checkpoint(self, checkpoint_name: str, state: Dict[str, Any]) -> str:
        """Create a checkpoint"""
        try:
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "name": checkpoint_name,
                "state": state,
                "created_at": now_ts()
            }
            
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            save_json(checkpoint_file, checkpoint_data)
            
            # Update index
            index = load_json(self.checkpoint_index, {"checkpoints": []})
            index["checkpoints"].append({
                "checkpoint_id": checkpoint_id,
                "name": checkpoint_name,
                "created_at": checkpoint_data["created_at"]
            })
            save_json(self.checkpoint_index, index)
            
            self.logger.info(f"Created checkpoint {checkpoint_id}")
            return checkpoint_id
        except Exception as e:
            self.logger.exception(f"Failed to create checkpoint: {e}")
            raise
    
    def rollback_to_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Rollback to a checkpoint"""
        try:
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            if not checkpoint_file.exists():
                raise ValueError(f"Checkpoint {checkpoint_id} not found")
            
            checkpoint_data = load_json(checkpoint_file)
            self.logger.info(f"Rolled back to checkpoint {checkpoint_id}")
            return checkpoint_data["state"]
        except Exception as e:
            self.logger.exception(f"Failed to rollback to checkpoint: {e}")
            raise
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        index = load_json(self.checkpoint_index, {"checkpoints": []})
        return index.get("checkpoints", [])
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback tasks"""
        action = task.get("action")
        
        if action == "create":
            checkpoint_id = self.create_checkpoint(task["name"], task["state"])
            return {"status": "success", "checkpoint_id": checkpoint_id}
        elif action == "rollback":
            state = self.rollback_to_checkpoint(task["checkpoint_id"])
            return {"status": "success", "state": state}
        elif action == "list":
            checkpoints = self.list_checkpoints()
            return {"status": "success", "checkpoints": checkpoints}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
