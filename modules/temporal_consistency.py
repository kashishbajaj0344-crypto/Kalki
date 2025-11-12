# ============================================================
# Kalki v2.4 — temporal_consistency.py
# ------------------------------------------------------------
# Temporal Context Buffer: Cross-Session Continuity
# - Maintains reasoning continuity across sessions
# - Detects contradictions and logical evolutions
# - Cross-references conclusions from earlier designs
# - Ensures long-term project consistency
# ============================================================

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib
import difflib
import re

from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.TemporalConsistency")

@dataclass
class ReasoningConclusion:
    """A conclusion drawn during reasoning"""
    conclusion_id: str
    query: str
    conclusion: str
    confidence: float  # 0-1 scale
    reasoning_path: List[str]  # Steps that led to this conclusion
    domains_involved: List[str]  # Knowledge domains referenced
    timestamp: str
    session_id: str
    project_context: str = ""  # For long-term projects
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LogicalEvolution:
    """Evolution of a logical position over time"""
    topic: str
    original_conclusion: str
    current_conclusion: str
    evolution_type: str  # "refinement", "revision", "expansion", "contradiction"
    confidence_change: float
    reasoning_chain: List[str]
    timestamps: List[str]
    sessions_involved: List[str]

@dataclass
class Contradiction:
    """Detected contradiction between conclusions"""
    contradiction_id: str
    conclusion_a: str
    conclusion_b: str
    topic: str
    severity: float  # 0-1 scale
    resolution_suggestion: str
    timestamp: str
    sessions: List[str]
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectContinuity:
    """Continuity tracking for long-term projects"""
    project_id: str
    project_name: str
    start_date: str
    last_updated: str
    key_assumptions: List[str]
    core_conclusions: List[str]
    unresolved_questions: List[str]
    evolution_history: List[LogicalEvolution]
    contradiction_log: List[Contradiction]
    consistency_score: float  # 0-1 scale

class TemporalConsistencyBuffer:
    """
    Temporal Context Buffer: Cross-Session Continuity

    Maintains reasoning continuity across sessions:
    - Cross-references conclusions from earlier designs/reports
    - Detects contradictions or evolutions in logic
    - Ensures long-term projects remain internally consistent
    """

    def __init__(self, buffer_size: int = 5000, project_retention_days: int = 365):
        self.buffer_size = buffer_size
        self.project_retention_days = project_retention_days

        # Core data structures
        self.conclusions: List[ReasoningConclusion] = []
        self.logical_evolutions: Dict[str, LogicalEvolution] = {}
        self.contradictions: List[Contradiction] = []
        self.projects: Dict[str, ProjectContinuity] = {}

        # Indexing for fast lookup
        self.topic_index: Dict[str, List[str]] = defaultdict(list)  # topic -> conclusion_ids
        self.session_index: Dict[str, List[str]] = defaultdict(list)  # session -> conclusion_ids
        self.project_index: Dict[str, List[str]] = defaultdict(list)  # project -> conclusion_ids

        # Persistence
        self.data_dir = "data/temporal_context"
        self.conclusions_file = f"{self.data_dir}/conclusions.json"
        self.projects_file = f"{self.data_dir}/projects.json"
        self.evolutions_file = f"{self.data_dir}/evolutions.json"

        # Load existing data
        self._load_persistent_data()

        logger.info(f"Temporal Consistency Buffer initialized with buffer size: {buffer_size}")

    def _load_persistent_data(self):
        """Load persistent temporal context data"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            # Load conclusions
            if os.path.exists(self.conclusions_file):
                with open(self.conclusions_file, 'r') as f:
                    conclusions_data = json.load(f)
                    for item in conclusions_data.get('conclusions', []):
                        conclusion = ReasoningConclusion(**item)
                        self.conclusions.append(conclusion)
                        self._index_conclusion(conclusion)

            # Load projects
            if os.path.exists(self.projects_file):
                with open(self.projects_file, 'r') as f:
                    projects_data = json.load(f)
                    for project_id, project_data in projects_data.get('projects', {}).items():
                        project = ProjectContinuity(**project_data)
                        self.projects[project_id] = project

            # Load evolutions
            if os.path.exists(self.evolutions_file):
                with open(self.evolutions_file, 'r') as f:
                    evolutions_data = json.load(f)
                    for topic, evolution_data in evolutions_data.get('evolutions', {}).items():
                        evolution = LogicalEvolution(**evolution_data)
                        self.logical_evolutions[topic] = evolution

        except Exception as e:
            logger.warning(f"Failed to load persistent temporal data: {e}")

    def _save_persistent_data(self):
        """Save temporal context data persistently"""
        try:
            # Save conclusions (most recent 2000)
            conclusions_data = {
                'conclusions': [asdict(c) for c in self.conclusions[-2000:]],
                'last_updated': datetime.now().isoformat()
            }

            with open(self.conclusions_file, 'w') as f:
                json.dump(conclusions_data, f, indent=2)

            # Save projects
            projects_data = {
                'projects': {pid: asdict(p) for pid, p in self.projects.items()},
                'last_updated': datetime.now().isoformat()
            }

            with open(self.projects_file, 'w') as f:
                json.dump(projects_data, f, indent=2)

            # Save evolutions
            evolutions_data = {
                'evolutions': {topic: asdict(e) for topic, e in self.logical_evolutions.items()},
                'last_updated': datetime.now().isoformat()
            }

            with open(self.evolutions_file, 'w') as f:
                json.dump(evolutions_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save persistent temporal data: {e}")

    def _index_conclusion(self, conclusion: ReasoningConclusion):
        """Index a conclusion for fast lookup"""
        # Topic indexing
        topics = self._extract_topics(conclusion.conclusion)
        for topic in topics:
            self.topic_index[topic].append(conclusion.conclusion_id)

        # Session indexing
        self.session_index[conclusion.session_id].append(conclusion.conclusion_id)

        # Project indexing
        if conclusion.project_context:
            self.project_index[conclusion.project_context].append(conclusion.conclusion_id)

    async def add_conclusion(self,
                           query: str,
                           conclusion: str,
                           confidence: float,
                           reasoning_path: List[str],
                           domains_involved: List[str],
                           session_id: str,
                           project_context: str = "") -> ReasoningConclusion:
        """
        Add a new conclusion to the temporal buffer

        Args:
            query: The original query
            conclusion: The conclusion drawn
            confidence: Confidence level (0-1)
            reasoning_path: Steps that led to this conclusion
            domains_involved: Knowledge domains referenced
            session_id: Current session identifier
            project_context: Long-term project identifier (optional)

        Returns:
            The created conclusion object
        """

        conclusion_id = hashlib.md5(
            f"{query}{conclusion}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        conclusion_obj = ReasoningConclusion(
            conclusion_id=conclusion_id,
            query=query,
            conclusion=conclusion,
            confidence=confidence,
            reasoning_path=reasoning_path,
            domains_involved=domains_involved,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            project_context=project_context
        )

        # Add to buffer
        self.conclusions.append(conclusion_obj)

        # Maintain buffer size
        if len(self.conclusions) > self.buffer_size:
            removed = self.conclusions.pop(0)
            self._remove_from_index(removed)

        # Index the new conclusion
        self._index_conclusion(conclusion_obj)

        # Check for contradictions and evolutions
        await self._check_contradictions(conclusion_obj)
        await self._update_logical_evolutions(conclusion_obj)

        # Update project continuity if applicable
        if project_context:
            await self._update_project_continuity(project_context, conclusion_obj)

        # Persist data
        self._save_persistent_data()

        logger.info(f"Added conclusion {conclusion_id} to temporal buffer")

        return conclusion_obj

    def _remove_from_index(self, conclusion: ReasoningConclusion):
        """Remove a conclusion from all indices"""
        # Topic index
        topics = self._extract_topics(conclusion.conclusion)
        for topic in topics:
            if conclusion.conclusion_id in self.topic_index[topic]:
                self.topic_index[topic].remove(conclusion.conclusion_id)

        # Session index
        if conclusion.session_id in self.session_index:
            if conclusion.conclusion_id in self.session_index[conclusion.session_id]:
                self.session_index[conclusion.session_id].remove(conclusion.conclusion_id)

        # Project index
        if conclusion.project_context and conclusion.project_context in self.project_index:
            if conclusion.conclusion_id in self.project_index[conclusion.project_context]:
                self.project_index[conclusion.project_context].remove(conclusion.conclusion_id)

    async def get_relevant_context(self,
                                 query: str,
                                 session_id: str = "",
                                 project_context: str = "",
                                 max_results: int = 10) -> Dict[str, Any]:
        """
        Get relevant historical context for a query

        Args:
            query: Current query
            session_id: Current session (to exclude)
            project_context: Project context
            max_results: Maximum results to return

        Returns:
            Dictionary with relevant conclusions, contradictions, and evolutions
        """

        relevant_conclusions = []
        relevant_contradictions = []
        relevant_evolutions = []

        # Extract topics from query
        query_topics = self._extract_topics(query)

        # Find relevant conclusions by topic
        for topic in query_topics:
            if topic in self.topic_index:
                for conclusion_id in self.topic_index[topic][-20:]:  # Last 20 per topic
                    conclusion = self._get_conclusion_by_id(conclusion_id)
                    if conclusion and conclusion.session_id != session_id:
                        relevant_conclusions.append(conclusion)

        # Add project-specific conclusions
        if project_context and project_context in self.project_index:
            for conclusion_id in self.project_index[project_context][-10:]:
                conclusion = self._get_conclusion_by_id(conclusion_id)
                if conclusion and conclusion not in relevant_conclusions:
                    relevant_conclusions.append(conclusion)

        # Find relevant contradictions
        for contradiction in self.contradictions[-20:]:  # Recent contradictions
            if any(topic in contradiction.topic for topic in query_topics):
                relevant_contradictions.append(contradiction)

        # Find relevant evolutions
        for topic, evolution in self.logical_evolutions.items():
            if any(t in topic for t in query_topics):
                relevant_evolutions.append(evolution)

        # Sort and limit results
        relevant_conclusions.sort(key=lambda x: x.timestamp, reverse=True)
        relevant_conclusions = relevant_conclusions[:max_results]

        return {
            "relevant_conclusions": [asdict(c) for c in relevant_conclusions],
            "contradictions": [asdict(c) for c in relevant_contradictions],
            "logical_evolutions": [asdict(e) for e in relevant_evolutions],
            "continuity_score": self._calculate_continuity_score(relevant_conclusions, relevant_contradictions)
        }

    async def _check_contradictions(self, new_conclusion: ReasoningConclusion):
        """Check for contradictions with existing conclusions"""

        topics = self._extract_topics(new_conclusion.conclusion)

        for topic in topics:
            if topic in self.topic_index:
                # Check recent conclusions on this topic
                recent_conclusion_ids = self.topic_index[topic][-10:]  # Last 10

                for conclusion_id in recent_conclusion_ids:
                    existing = self._get_conclusion_by_id(conclusion_id)
                    if existing and existing.conclusion_id != new_conclusion.conclusion_id:
                        contradiction_score = self._calculate_contradiction_score(
                            existing.conclusion, new_conclusion.conclusion
                        )

                        if contradiction_score > 0.7:  # High contradiction threshold
                            contradiction = Contradiction(
                                contradiction_id=hashlib.md5(
                                    f"{existing.conclusion_id}{new_conclusion.conclusion_id}".encode()
                                ).hexdigest()[:16],
                                conclusion_a=existing.conclusion,
                                conclusion_b=new_conclusion.conclusion,
                                topic=topic,
                                severity=contradiction_score,
                                resolution_suggestion=self._suggest_resolution(existing, new_conclusion),
                                timestamp=datetime.now().isoformat(),
                                sessions=[existing.session_id, new_conclusion.session_id],
                                context={
                                    "topic": topic,
                                    "confidence_a": existing.confidence,
                                    "confidence_b": new_conclusion.confidence
                                }
                            )

                            self.contradictions.append(contradiction)
                            logger.warning(f"Contradiction detected: {contradiction.contradiction_id}")

    async def _update_logical_evolutions(self, new_conclusion: ReasoningConclusion):
        """Update logical evolutions based on new conclusion"""

        topics = self._extract_topics(new_conclusion.conclusion)

        for topic in topics:
            if topic not in self.logical_evolutions:
                # New topic evolution
                self.logical_evolutions[topic] = LogicalEvolution(
                    topic=topic,
                    original_conclusion=new_conclusion.conclusion,
                    current_conclusion=new_conclusion.conclusion,
                    evolution_type="initial",
                    confidence_change=0.0,
                    reasoning_chain=[new_conclusion.conclusion],
                    timestamps=[new_conclusion.timestamp],
                    sessions_involved=[new_conclusion.session_id]
                )
            else:
                # Update existing evolution
                evolution = self.logical_evolutions[topic]

                # Check if this represents an evolution
                similarity = self._calculate_similarity(evolution.current_conclusion, new_conclusion.conclusion)

                if similarity < 0.8:  # Significant change
                    evolution_type = self._classify_evolution(
                        evolution.current_conclusion, new_conclusion.conclusion
                    )

                    evolution.evolution_type = evolution_type
                    evolution.current_conclusion = new_conclusion.conclusion
                    evolution.confidence_change = new_conclusion.confidence - evolution.confidence_change
                    evolution.reasoning_chain.append(new_conclusion.conclusion)
                    evolution.timestamps.append(new_conclusion.timestamp)
                    evolution.sessions_involved.append(new_conclusion.session_id)

                    logger.info(f"Logical evolution detected for topic '{topic}': {evolution_type}")

    async def _update_project_continuity(self, project_id: str, conclusion: ReasoningConclusion):
        """Update project continuity tracking"""

        if project_id not in self.projects:
            # New project
            self.projects[project_id] = ProjectContinuity(
                project_id=project_id,
                project_name=f"Project {project_id}",
                start_date=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                key_assumptions=[],
                core_conclusions=[],
                unresolved_questions=[],
                evolution_history=[],
                contradiction_log=[],
                consistency_score=1.0
            )

        project = self.projects[project_id]
        project.last_updated = datetime.now().isoformat()

        # Add conclusion to core conclusions if high confidence
        if conclusion.confidence > 0.8:
            if conclusion.conclusion not in project.core_conclusions:
                project.core_conclusions.append(conclusion.conclusion)

        # Update consistency score
        project_contradictions = [c for c in self.contradictions
                                if project_id in c.sessions]

        if project_contradictions:
            # Reduce consistency score based on contradictions
            contradiction_penalty = min(0.5, len(project_contradictions) * 0.1)
            project.consistency_score = max(0.0, project.consistency_score - contradiction_penalty)

        # Clean up old projects
        cutoff_date = datetime.now() - timedelta(days=self.project_retention_days)
        projects_to_remove = []

        for pid, proj in self.projects.items():
            if datetime.fromisoformat(proj.last_updated) < cutoff_date:
                projects_to_remove.append(pid)

        for pid in projects_to_remove:
            del self.projects[pid]
            logger.info(f"Removed expired project: {pid}")

    def _get_conclusion_by_id(self, conclusion_id: str) -> Optional[ReasoningConclusion]:
        """Get conclusion by ID"""
        for conclusion in self.conclusions:
            if conclusion.conclusion_id == conclusion_id:
                return conclusion
        return None

    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        # Simple topic extraction - could be enhanced with NLP
        topic_keywords = {
            "design": ["design", "create", "build", "develop", "construct"],
            "physics": ["physics", "force", "energy", "quantum", "relativity"],
            "biology": ["biology", "cell", "organism", "evolution", "dna"],
            "ai": ["ai", "artificial intelligence", "machine learning", "neural"],
            "ethics": ["ethics", "moral", "safety", "responsible", "bias"],
            "engineering": ["engineering", "mechanical", "electrical", "software"],
            "mathematics": ["math", "equation", "algorithm", "computation"],
            "psychology": ["psychology", "mind", "behavior", "cognitive", "emotion"]
        }

        topics = []
        text_lower = text.lower()

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)

        return list(set(topics))  # Remove duplicates

    def _calculate_contradiction_score(self, text_a: str, text_b: str) -> float:
        """Calculate contradiction score between two texts"""
        # Simple contradiction detection - could be enhanced
        contradiction_indicators = [
            # Direct opposites
            ("possible", "impossible"),
            ("safe", "dangerous"),
            ("effective", "ineffective"),
            ("reliable", "unreliable"),
            ("ethical", "unethical"),
            # Conflicting statements
            ("always", "never"),
            ("all", "none"),
            ("every", "no"),
        ]

        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        contradiction_score = 0.0

        for pos, neg in contradiction_indicators:
            if pos in text_a_lower and neg in text_b_lower:
                contradiction_score += 0.3
            elif neg in text_a_lower and pos in text_b_lower:
                contradiction_score += 0.3

        # Semantic similarity (lower similarity = higher contradiction potential)
        similarity = difflib.SequenceMatcher(None, text_a_lower, text_b_lower).ratio()
        contradiction_score += (1.0 - similarity) * 0.4

        return min(1.0, contradiction_score)

    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate semantic similarity between texts"""
        return difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

    def _classify_evolution(self, old_text: str, new_text: str) -> str:
        """Classify the type of logical evolution"""
        similarity = self._calculate_similarity(old_text, new_text)

        if similarity > 0.9:
            return "refinement"  # Minor changes
        elif similarity > 0.7:
            return "expansion"   # Added information
        elif self._calculate_contradiction_score(old_text, new_text) > 0.6:
            return "contradiction"  # Direct contradiction
        else:
            return "revision"    # Significant change

    def _suggest_resolution(self, conclusion_a: ReasoningConclusion, conclusion_b: ReasoningConclusion) -> str:
        """Suggest resolution for a contradiction"""
        if conclusion_a.confidence > conclusion_b.confidence:
            return f"Prioritize earlier conclusion (higher confidence: {conclusion_a.confidence:.2f})"
        elif conclusion_b.confidence > conclusion_a.confidence:
            return f"Prioritize newer conclusion (higher confidence: {conclusion_b.confidence:.2f})"
        else:
            return "Further investigation needed - gather additional evidence"

    def _calculate_continuity_score(self, conclusions: List[ReasoningConclusion],
                                  contradictions: List[Contradiction]) -> float:
        """Calculate continuity score for a set of conclusions"""
        if not conclusions:
            return 1.0

        base_score = 0.8  # Start with high continuity

        # Reduce score for contradictions
        contradiction_penalty = len(contradictions) * 0.1
        base_score -= contradiction_penalty

        # Reduce score for low confidence conclusions
        low_confidence_penalty = sum(1 for c in conclusions if c.confidence < 0.6) * 0.05
        base_score -= low_confidence_penalty

        # Increase score for recent, high-confidence conclusions
        recent_bonus = sum(c.confidence for c in conclusions[-5:] if c.confidence > 0.8) * 0.02
        base_score += recent_bonus

        return max(0.0, min(1.0, base_score))

    def get_continuity_report(self, project_id: str = "") -> Dict[str, Any]:
        """Get comprehensive continuity report"""
        if project_id and project_id in self.projects:
            project = self.projects[project_id]
            return {
                "project_continuity": asdict(project),
                "active_conclusions": len(self.conclusions),
                "total_contradictions": len(self.contradictions),
                "logical_evolutions": len(self.logical_evolutions),
                "system_health": "good" if project.consistency_score > 0.7 else "needs_attention"
            }
        else:
            return {
                "total_conclusions": len(self.conclusions),
                "total_contradictions": len(self.contradictions),
                "total_evolutions": len(self.logical_evolutions),
                "active_projects": len(self.projects),
                "overall_continuity_score": self._calculate_overall_continuity(),
                "recent_activity": [asdict(c) for c in self.conclusions[-5:]]
            }

    def _calculate_overall_continuity(self) -> float:
        """Calculate overall system continuity score"""
        if not self.conclusions:
            return 1.0

        recent_conclusions = self.conclusions[-50:]  # Last 50 conclusions
        recent_contradictions = [c for c in self.contradictions
                               if datetime.fromisoformat(c.timestamp) >
                               datetime.now() - timedelta(days=7)]

        continuity_score = 0.85  # Base score

        # Penalty for contradictions
        contradiction_penalty = len(recent_contradictions) * 0.05
        continuity_score -= contradiction_penalty

        # Bonus for high-confidence recent conclusions
        confidence_bonus = sum(c.confidence for c in recent_conclusions) / len(recent_conclusions) * 0.1
        continuity_score += confidence_bonus

        return max(0.0, min(1.0, continuity_score))

    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old temporal data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        # Remove old conclusions
        original_count = len(self.conclusions)
        self.conclusions = [c for c in self.conclusions
                          if datetime.fromisoformat(c.timestamp) > cutoff_date]

        removed_count = original_count - len(self.conclusions)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old conclusions")

        # Rebuild indices
        self.topic_index.clear()
        self.session_index.clear()
        self.project_index.clear()

        for conclusion in self.conclusions:
            self._index_conclusion(conclusion)

        # Save cleaned data
        self._save_persistent_data()

# Global temporal consistency buffer instance
_temporal_buffer = None

def get_temporal_consistency_buffer() -> TemporalConsistencyBuffer:
    """Get the global temporal consistency buffer instance"""
    global _temporal_buffer
    if _temporal_buffer is None:
        _temporal_buffer = TemporalConsistencyBuffer()
    return _temporal_buffer

async def add_temporal_conclusion(query: str,
                                conclusion: str,
                                confidence: float,
                                reasoning_path: List[str],
                                domains_involved: List[str],
                                session_id: str,
                                project_context: str = "") -> Dict[str, Any]:
    """Convenience function for adding temporal conclusions"""
    buffer = get_temporal_consistency_buffer()
    conclusion_obj = await buffer.add_conclusion(
        query, conclusion, confidence, reasoning_path,
        domains_involved, session_id, project_context
    )
    return {
        "conclusion": asdict(conclusion_obj),
        "continuity_report": buffer.get_continuity_report(project_context)
    }

async def get_temporal_context(query: str,
                             session_id: str = "",
                             project_context: str = "",
                             max_results: int = 10) -> Dict[str, Any]:
    """Convenience function for retrieving temporal context"""
    buffer = get_temporal_consistency_buffer()
    return await buffer.get_relevant_context(query, session_id, project_context, max_results)