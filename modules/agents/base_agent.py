#!/usr/bin/env python3
"""
Base Agent class for all Kalki agents
Provides common functionality: logging, state management, event handling
"""
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime


class BaseAgent:
    """
    Base class for all Kalki agents
    Provides: unique ID, logging, state management, lifecycle hooks
    """
    
    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.state = "initialized"
        self.created_at = datetime.now()
        self.logger = logging.getLogger(f"kalki.agents.{self.name}")
        self.metadata = {}
        
    def initialize(self) -> bool:
        """
        Initialize agent resources
        Override in subclasses for custom initialization
        """
        try:
            self.state = "ready"
            self.logger.info(f"Agent {self.name} ({self.id}) initialized")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            self.state = "error"
            return False
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task
        Override in subclasses for custom execution logic
        """
        raise NotImplementedError(f"{self.name}.execute() must be implemented")
    
    def shutdown(self) -> bool:
        """
        Cleanup agent resources
        Override in subclasses for custom cleanup
        """
        try:
            self.state = "shutdown"
            self.logger.info(f"Agent {self.name} ({self.id}) shutdown")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown {self.name}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status
        """
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    def update_metadata(self, key: str, value: Any):
        """
        Update agent metadata
        """
        self.metadata[key] = value
        
    def __repr__(self):
        return f"<{self.__class__.__name__} id={self.id[:8]} state={self.state}>"
