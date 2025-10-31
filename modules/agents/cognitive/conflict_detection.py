"""
Conflict Detection Agent (Phase 6)
==================================

Detects conflicts in knowledge base with semantic analysis.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from difflib import SequenceMatcher
from modules.logging_config import get_logger

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.ConflictDetection")


class ConflictDetectionAgent(BaseAgent):
    """
    Detects conflicts in knowledge base
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="ConflictDetectionAgent",
            capabilities=[AgentCapability.CONFLICT_DETECTION, AgentCapability.VALIDATION],
            description="Knowledge conflict detection with semantic analysis",
            config=config or {}
        )
        self.conflicts = []
        self.knowledge_base = []

    async def initialize(self) -> bool:
        """Initialize conflict detection system"""
        try:
            logger.info("ConflictDetectionAgent initialized with semantic analysis")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize ConflictDetectionAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conflict detection tasks"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "detect":
            conflicts = self.detect_conflicts(params["knowledge_items"])
            return {"status": "success", "conflicts": conflicts}
        elif action == "resolve":
            self.resolve_conflict(
                params["conflict_id"],
                params["resolution"],
                params["chosen_item"]
            )
            return {"status": "success"}
        elif action == "list":
            return {"status": "success", "conflicts": self.conflicts}
        elif action == "add_knowledge":
            self.add_knowledge(params["items"])
            return {"status": "success"}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    def add_knowledge(self, items: List[Dict[str, Any]]):
        """Add items to the knowledge base"""
        try:
            for item in items:
                if item not in self.knowledge_base:
                    self.knowledge_base.append(item)
                    logger.debug(f"Added knowledge item: {item.get('id', 'unknown')}")
        except Exception as e:
            logger.exception(f"Failed to add knowledge: {e}")

    def detect_conflicts(self, knowledge_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect conflicts between knowledge items
        """
        try:
            conflicts = []
            items_to_check = knowledge_items + self.knowledge_base

            # Check for conflicts between all pairs
            for i, item1 in enumerate(items_to_check):
                for item2 in items_to_check[i+1:]:
                    conflict = self._detect_conflict_between(item1, item2)
                    if conflict:
                        conflict["conflict_id"] = f"conflict_{len(conflicts) + len(self.conflicts)}"
                        conflict["detected_at"] = datetime.utcnow().isoformat()
                        conflicts.append(conflict)

            self.conflicts.extend(conflicts)
            logger.info(f"Detected {len(conflicts)} conflicts")
            return conflicts

        except Exception as e:
            logger.exception(f"Failed to detect conflicts: {e}")
            return []

    def _detect_conflict_between(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect conflict between two knowledge items"""
        try:
            text1 = item1.get("text", "").lower().strip()
            text2 = item2.get("text", "").lower().strip()

            if not text1 or not text2:
                return None

            # Check for direct contradictions
            contradiction = self._check_contradiction(text1, text2)
            if contradiction:
                return {
                    "item1": item1,
                    "item2": item2,
                    "type": "contradiction",
                    "description": contradiction["description"],
                    "severity": contradiction["severity"],
                    "evidence": contradiction["evidence"]
                }

            # Check for logical inconsistencies
            inconsistency = self._check_logical_inconsistency(item1, item2)
            if inconsistency:
                return {
                    "item1": item1,
                    "item2": item2,
                    "type": "logical_inconsistency",
                    "description": inconsistency["description"],
                    "severity": "medium",
                    "evidence": inconsistency["evidence"]
                }

            # Check for temporal conflicts
            temporal_conflict = self._check_temporal_conflict(item1, item2)
            if temporal_conflict:
                return {
                    "item1": item1,
                    "item2": item2,
                    "type": "temporal_conflict",
                    "description": temporal_conflict["description"],
                    "severity": "high",
                    "evidence": temporal_conflict["evidence"]
                }

            return None

        except Exception as e:
            logger.exception(f"Conflict detection error: {e}")
            return None

    def _check_contradiction(self, text1: str, text2: str) -> Optional[Dict[str, Any]]:
        """Check for direct contradictions in text"""
        try:
            # Common contradiction patterns
            contradiction_pairs = [
                ("not", ""),
                ("never", "always"),
                ("impossible", "possible"),
                ("false", "true"),
                ("incorrect", "correct"),
                ("invalid", "valid")
            ]

            for neg, pos in contradiction_pairs:
                if neg in text1 and pos in text2:
                    # Check if they're referring to the same subject
                    similarity = SequenceMatcher(None, text1, text2).ratio()
                    if similarity > 0.6:  # High similarity suggests same subject
                        return {
                            "description": f"Direct contradiction detected: '{neg}' in one statement conflicts with '{pos}' in another",
                            "severity": "high",
                            "evidence": {
                                "text1": text1,
                                "text2": text2,
                                "conflicting_terms": [neg, pos],
                                "similarity_score": similarity
                            }
                        }

            # Check for numerical contradictions
            import re
            numbers1 = re.findall(r'\d+\.?\d*', text1)
            numbers2 = re.findall(r'\d+\.?\d*', text2)

            if numbers1 and numbers2:
                # If numbers are very different and context suggests they should be similar
                if abs(float(numbers1[0]) - float(numbers2[0])) > max(float(numbers1[0]), float(numbers2[0])) * 0.5:
                    return {
                        "description": "Numerical contradiction detected",
                        "severity": "medium",
                        "evidence": {
                            "text1": text1,
                            "text2": text2,
                            "numbers1": numbers1,
                            "numbers2": numbers2
                        }
                    }

            return None

        except Exception:
            return None

    def _check_logical_inconsistency(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for logical inconsistencies"""
        try:
            # Check for mutually exclusive categories
            category1 = item1.get("category", "")
            category2 = item2.get("category", "")

            if category1 and category2:
                mutually_exclusive = {
                    ("fact", "opinion"),
                    ("cause", "effect"),
                    ("necessary", "sufficient"),
                    ("deterministic", "probabilistic")
                }

                for cat1, cat2 in mutually_exclusive:
                    if (category1 == cat1 and category2 == cat2) or (category1 == cat2 and category2 == cat1):
                        return {
                            "description": f"Logical inconsistency: {category1} and {category2} are mutually exclusive categories",
                            "evidence": {
                                "item1_category": category1,
                                "item2_category": category2
                            }
                        }

            # Check for circular reasoning
            if self._detects_circular_reasoning(item1, item2):
                return {
                    "description": "Potential circular reasoning detected",
                    "evidence": {
                        "item1": item1.get("text", ""),
                        "item2": item2.get("text", "")
                    }
                }

            return None

        except Exception:
            return None

    def _check_temporal_conflict(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for temporal conflicts"""
        try:
            time1 = item1.get("timestamp") or item1.get("time")
            time2 = item2.get("timestamp") or item2.get("time")

            if time1 and time2:
                # Parse timestamps
                try:
                    dt1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
                    dt2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))

                    # Check for causality violations
                    text1 = item1.get("text", "").lower()
                    text2 = item2.get("text", "").lower()

                    # If one mentions "before" and times don't match
                    if "before" in text1 and dt1 > dt2:
                        return {
                            "description": "Temporal causality violation: event stated as 'before' actually occurs after",
                            "evidence": {
                                "text1": text1,
                                "time1": time1,
                                "text2": text2,
                                "time2": time2
                            }
                        }

                    if "after" in text1 and dt1 < dt2:
                        return {
                            "description": "Temporal causality violation: event stated as 'after' actually occurs before",
                            "evidence": {
                                "text1": text1,
                                "time1": time1,
                                "text2": text2,
                                "time2": time2
                            }
                        }

                except (ValueError, AttributeError):
                    pass

            return None

        except Exception:
            return None

    def _detects_circular_reasoning(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """Detect potential circular reasoning"""
        try:
            text1 = item1.get("text", "").lower()
            text2 = item2.get("text", "").lower()

            # Simple circular reasoning detection
            # If one statement's conclusion is the other's premise
            words1 = set(text1.split())
            words2 = set(text2.split())

            # High overlap might indicate circular reasoning
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            return overlap > 0.8

        except Exception:
            return False

    def resolve_conflict(self, conflict_id: str, resolution: str, chosen_item: int):
        """
        Resolve a detected conflict
        """
        try:
            conflict = next((c for c in self.conflicts if c["conflict_id"] == conflict_id), None)
            if not conflict:
                raise ValueError(f"Conflict {conflict_id} not found")

            conflict["resolution"] = resolution
            conflict["chosen_item"] = chosen_item
            conflict["resolved_at"] = datetime.utcnow().isoformat()

            logger.info(f"Resolved conflict {conflict_id} with resolution: {resolution}")

        except Exception as e:
            logger.exception(f"Failed to resolve conflict: {e}")
            raise

    async def shutdown(self) -> bool:
        """Shutdown the conflict detection agent"""
        try:
            logger.info("ConflictDetectionAgent shutting down")
            return True
        except Exception as e:
            logger.exception(f"Shutdown error: {e}")
            return False