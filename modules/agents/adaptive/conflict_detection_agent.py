#!/usr/bin/env python3
"""
ConflictDetectionAgent — Knowledge consistency and contradiction pruning
Modular design for detecting and resolving knowledge conflicts
"""
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path
from ..base_agent import BaseAgent


class ConflictDetectionAgent(BaseAgent):
    """
    Knowledge consistency and contradiction pruning.
    Detects conflicting information and provides resolution strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ConflictDetectionAgent", config=config)
        self.conflict_records = []
        self.conflict_resolution_strategies = {
            "confidence_weighted": self._resolve_by_confidence,
            "recency_based": self._resolve_by_recency,
            "consensus": self._resolve_by_consensus,
            "manual_review": self._resolve_manually
        }
        self.conflicts_file = Path.home() / "Desktop" / "Kalki" / "vector_db" / "conflict_records.json"
        self._load_conflict_records()

    def _load_conflict_records(self):
        """Load persisted conflict records from disk"""
        try:
            if self.conflicts_file.exists():
                with open(self.conflicts_file, 'r') as f:
                    data = json.load(f)
                    self.conflict_records = data.get('conflict_records', [])
                self.logger.info(f"Loaded {len(self.conflict_records)} conflict records from disk")
        except Exception as e:
            self.logger.warning(f"Failed to load conflict records: {e}")

    def _save_conflict_records(self):
        """Persist conflict records to disk"""
        try:
            self.conflicts_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'conflict_records': self.conflict_records[-500:],  # Keep last 500 records
                'last_updated': datetime.now().isoformat()
            }
            with open(self.conflicts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save conflict records: {e}")

    def detect_conflicts(self, knowledge_base: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect conflicts in a knowledge base
        """
        conflicts = []

        try:
            # Compare each pair of knowledge items
            for i, item1 in enumerate(knowledge_base):
                for j, item2 in enumerate(knowledge_base[i+1:], i+1):
                    if self._are_conflicting(item1, item2):
                        conflict = {
                            "conflict_id": f"conflict_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                            "item1": item1,
                            "item2": item2,
                            "conflict_type": self._classify_conflict(item1, item2),
                            "detected_at": datetime.now().isoformat(),
                            "resolution_status": "unresolved",
                            "resolution_strategy": None
                        }
                        conflicts.append(conflict)

            # Record conflicts
            for conflict in conflicts:
                self.conflict_records.append(conflict)

            self._save_conflict_records()

            self.logger.info(f"Detected {len(conflicts)} conflicts in knowledge base")
            return conflicts

        except Exception as e:
            self.logger.exception(f"Failed to detect conflicts: {e}")
            return []

    def _are_conflicting(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> bool:
        """
        Determine if two knowledge items conflict
        TODO: Upgrade to semantic similarity via embeddings once available
        """
        try:
            # Simple rule-based conflict detection
            text1 = self._extract_text_content(item1).lower()
            text2 = self._extract_text_content(item2).lower()

            # Direct contradictions
            contradiction_pairs = [
                ("true", "false"),
                ("yes", "no"),
                ("correct", "incorrect"),
                ("valid", "invalid"),
                ("success", "failure"),
                ("enabled", "disabled")
            ]

            for pos, neg in contradiction_pairs:
                if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                    return True

            # Conflicting numerical values (if both contain numbers)
            nums1 = self._extract_numbers(text1)
            nums2 = self._extract_numbers(text2)

            if nums1 and nums2:
                # Check for significantly different values
                for num1 in nums1:
                    for num2 in nums2:
                        if abs(num1 - num2) / max(abs(num1), abs(num2), 1) > 0.5:  # 50% difference threshold
                            return True

            # Conflicting temporal information
            if self._have_temporal_conflict(text1, text2):
                return True

            return False

        except Exception as e:
            self.logger.debug(f"Error checking conflict: {e}")
            return False

    def _extract_text_content(self, item: Dict[str, Any]) -> str:
        """Extract text content from knowledge item"""
        # Try common fields
        for field in ['content', 'text', 'description', 'summary', 'hypothesis']:
            if field in item and isinstance(item[field], str):
                return item[field]

        # Fallback to string representation
        return str(item)

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(num) for num in numbers]

    def _have_temporal_conflict(self, text1: str, text2: str) -> bool:
        """Check for temporal conflicts (e.g., 'before' vs 'after')"""
        before_words = ['before', 'prior', 'earlier', 'previously']
        after_words = ['after', 'later', 'subsequently', 'following']

        has_before_1 = any(word in text1 for word in before_words)
        has_after_1 = any(word in text1 for word in after_words)
        has_before_2 = any(word in text2 for word in before_words)
        has_after_2 = any(word in text2 for word in after_words)

        # Conflict if one says "before" and other says "after"
        return (has_before_1 and has_after_2) or (has_after_1 and has_before_2)

    def _classify_conflict(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> str:
        """Classify the type of conflict"""
        text1 = self._extract_text_content(item1).lower()
        text2 = self._extract_text_content(item2).lower()

        if any(word in text1 + text2 for word in ['true', 'false', 'yes', 'no']):
            return "logical_contradiction"
        elif self._extract_numbers(text1) and self._extract_numbers(text2):
            return "numerical_discrepancy"
        elif self._have_temporal_conflict(text1, text2):
            return "temporal_conflict"
        else:
            return "semantic_conflict"

    def resolve_conflict(self, conflict_id: str, strategy: str = "confidence_weighted") -> Dict[str, Any]:
        """
        Resolve a detected conflict using specified strategy
        """
        try:
            # Find the conflict
            conflict = None
            for c in self.conflict_records:
                if c["conflict_id"] == conflict_id:
                    conflict = c
                    break

            if not conflict:
                raise ValueError(f"Conflict {conflict_id} not found")

            if conflict["resolution_status"] != "unresolved":
                return {"status": "already_resolved", "conflict": conflict}

            # Apply resolution strategy
            if strategy not in self.conflict_resolution_strategies:
                strategy = "manual_review"

            resolution_result = self.conflict_resolution_strategies[strategy](conflict)

            # Update conflict record
            conflict["resolution_status"] = "resolved"
            conflict["resolution_strategy"] = strategy
            conflict["resolution_result"] = resolution_result
            conflict["resolved_at"] = datetime.now().isoformat()

            self._save_conflict_records()

            self.logger.info(f"Resolved conflict {conflict_id} using {strategy} strategy")
            return {
                "status": "success",
                "conflict": conflict,
                "resolution": resolution_result
            }

        except Exception as e:
            self.logger.exception(f"Failed to resolve conflict {conflict_id}: {e}")
            return {"status": "error", "message": str(e)}

    def _resolve_by_confidence(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by confidence weighting"""
        item1 = conflict["item1"]
        item2 = conflict["item2"]

        conf1 = item1.get("confidence", 0.5)
        conf2 = item2.get("confidence", 0.5)

        if conf1 > conf2:
            return {"winner": "item1", "confidence_diff": conf1 - conf2, "reason": "higher_confidence"}
        elif conf2 > conf1:
            return {"winner": "item2", "confidence_diff": conf2 - conf1, "reason": "higher_confidence"}
        else:
            return {"winner": "tie", "reason": "equal_confidence"}

    def _resolve_by_recency(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict by recency (newer wins)"""
        item1 = conflict["item1"]
        item2 = conflict["item2"]

        time1 = item1.get("timestamp") or item1.get("created_at", "")
        time2 = item2.get("timestamp") or item2.get("created_at", "")

        if time1 > time2:
            return {"winner": "item1", "reason": "more_recent"}
        elif time2 > time1:
            return {"winner": "item2", "reason": "more_recent"}
        else:
            return {"winner": "tie", "reason": "same_timestamp"}

    def _resolve_by_consensus(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve by checking consensus with other knowledge items"""
        # This would require access to broader knowledge base
        # For now, fall back to confidence-based resolution
        return self._resolve_by_confidence(conflict)

    def _resolve_manually(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Mark for manual review"""
        return {"winner": "manual_review_required", "reason": "requires_human_judgment"}

    def get_conflict_summary(self, status: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of conflicts
        """
        try:
            conflicts = self.conflict_records

            if status:
                conflicts = [c for c in conflicts if c.get("resolution_status") == status]

            # Group by conflict type
            type_counts = {}
            for conflict in conflicts:
                conflict_type = conflict.get("conflict_type", "unknown")
                type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1

            # Resolution strategy counts
            strategy_counts = {}
            for conflict in conflicts:
                strategy = conflict.get("resolution_strategy")
                if strategy:
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            summary = {
                "total_conflicts": len(conflicts),
                "by_type": type_counts,
                "by_resolution_strategy": strategy_counts,
                "unresolved_count": len([c for c in conflicts if c.get("resolution_status") == "unresolved"]),
                "resolved_count": len([c for c in conflicts if c.get("resolution_status") == "resolved"])
            }

            return summary

        except Exception as e:
            self.logger.exception(f"Failed to get conflict summary: {e}")
            return {"error": str(e)}

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conflict detection and resolution tasks"""
        action = task.get("action")

        if action == "detect":
            conflicts = self.detect_conflicts(task["knowledge_base"])
            return {"status": "success", "conflicts": conflicts}
        elif action == "resolve":
            result = self.resolve_conflict(task["conflict_id"], task.get("strategy", "confidence_weighted"))
            return result
        elif action == "summary":
            summary = self.get_conflict_summary(task.get("status"))
            return {"status": "success", "summary": summary}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}