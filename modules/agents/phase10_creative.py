#!/usr/bin/env python3
"""
Phase 10: Creative Cognition & Synthetic Intuition
- CreativeAgent: Generative invention and dream mode
- PatternRecognitionAgent: Insight discovery
- IdeaFusionAgent: Cross-domain synthesis
"""
import logging
import random
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from ..utils import now_ts

logger = logging.getLogger("kalki.agents.phase10")


class CreativeAgent(BaseAgent):
    """
    Generates creative ideas and inventions
    Includes Dream Mode for exploration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="CreativeAgent", config=config)
        self.ideas = []
        self.dream_mode = False
    
    def generate_idea(self, domain: str, constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a creative idea in a domain"""
        try:
            idea_id = f"idea_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            idea = {
                "idea_id": idea_id,
                "domain": domain,
                "constraints": constraints or {},
                "description": self._generate_creative_description(domain, constraints),
                "novelty_score": random.uniform(0.5, 1.0),
                "feasibility_score": random.uniform(0.3, 0.9),
                "created_at": now_ts()
            }
            
            self.ideas.append(idea)
            self.logger.info(f"Generated idea {idea_id} in domain {domain}")
            return idea
        except Exception as e:
            self.logger.exception(f"Failed to generate idea: {e}")
            raise
    
    def _generate_creative_description(self, domain: str, constraints: Optional[Dict[str, Any]] = None) -> str:
        """Generate creative description"""
        # Simplified creative generation (can be enhanced with LLM)
        templates = {
            "technology": f"A novel approach to {domain} that combines emerging technologies",
            "art": f"An innovative artistic expression in {domain}",
            "science": f"A breakthrough hypothesis in {domain} research",
            "business": f"A disruptive business model for {domain}"
        }
        return templates.get(domain, f"A creative concept in {domain}")
    
    def enable_dream_mode(self):
        """Enable dream mode for unconstrained exploration"""
        self.dream_mode = True
        self.logger.info("Dream mode enabled")
    
    def disable_dream_mode(self):
        """Disable dream mode"""
        self.dream_mode = False
        self.logger.info("Dream mode disabled")
    
    def dream(self, theme: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate multiple ideas in dream mode"""
        if not self.dream_mode:
            self.enable_dream_mode()
        
        try:
            dreams = []
            domains = ["technology", "art", "science", "business"]
            
            for _ in range(random.randint(3, 7)):
                domain = random.choice(domains)
                idea = self.generate_idea(domain)
                idea["dream_theme"] = theme
                dreams.append(idea)
            
            self.logger.info(f"Generated {len(dreams)} dream ideas")
            return dreams
        except Exception as e:
            self.logger.exception(f"Dream generation failed: {e}")
            return []
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute creative tasks"""
        action = task.get("action")
        
        if action == "generate":
            idea = self.generate_idea(task["domain"], task.get("constraints"))
            return {"status": "success", "idea": idea}
        elif action == "dream":
            dreams = self.dream(task.get("theme"))
            return {"status": "success", "dreams": dreams}
        elif action == "list":
            return {"status": "success", "ideas": self.ideas}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class PatternRecognitionAgent(BaseAgent):
    """
    Discovers patterns and insights in data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="PatternRecognitionAgent", config=config)
        self.patterns = []
    
    def analyze_data(self, data: List[Dict[str, Any]], analysis_type: str = "general") -> List[Dict[str, Any]]:
        """Analyze data for patterns"""
        try:
            patterns = self._detect_patterns(data, analysis_type)
            
            for pattern in patterns:
                pattern["detected_at"] = now_ts()
                self.patterns.append(pattern)
            
            self.logger.info(f"Detected {len(patterns)} patterns in {analysis_type} analysis")
            return patterns
        except Exception as e:
            self.logger.exception(f"Pattern analysis failed: {e}")
            return []
    
    def _detect_patterns(self, data: List[Dict[str, Any]], analysis_type: str) -> List[Dict[str, Any]]:
        """Detect patterns in data"""
        # Simplified pattern detection (can be enhanced with ML)
        patterns = []
        
        if len(data) > 2:
            patterns.append({
                "pattern_id": f"pattern_{len(self.patterns)}",
                "type": "frequency",
                "description": f"Recurring elements in {analysis_type} data",
                "confidence": 0.75,
                "data_points": len(data)
            })
        
        return patterns
    
    def get_insights(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get insights from detected patterns"""
        try:
            relevant_patterns = self.patterns
            if domain:
                relevant_patterns = [p for p in self.patterns if domain in p.get("description", "")]
            
            insights = [
                {
                    "insight": f"Pattern {p['pattern_id']} suggests {p['description']}",
                    "confidence": p["confidence"],
                    "pattern_id": p["pattern_id"]
                }
                for p in relevant_patterns
            ]
            
            return insights
        except Exception as e:
            self.logger.exception(f"Failed to get insights: {e}")
            return []
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern recognition tasks"""
        action = task.get("action")
        
        if action == "analyze":
            patterns = self.analyze_data(task["data"], task.get("analysis_type", "general"))
            return {"status": "success", "patterns": patterns}
        elif action == "insights":
            insights = self.get_insights(task.get("domain"))
            return {"status": "success", "insights": insights}
        elif action == "list":
            return {"status": "success", "patterns": self.patterns}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class IdeaFusionAgent(BaseAgent):
    """
    Combines ideas from different domains for cross-domain synthesis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="IdeaFusionAgent", config=config)
        self.fusions = []
    
    def fuse_ideas(self, ideas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fuse multiple ideas into a novel concept"""
        try:
            if len(ideas) < 2:
                raise ValueError("Need at least 2 ideas to fuse")
            
            fusion_id = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Extract domains and key concepts
            domains = [idea.get("domain", "unknown") for idea in ideas]
            
            fusion = {
                "fusion_id": fusion_id,
                "source_ideas": [idea.get("idea_id", f"idea_{i}") for i, idea in enumerate(ideas)],
                "domains": domains,
                "description": self._create_fusion_description(ideas, domains),
                "novelty_score": self._calculate_fusion_novelty(ideas),
                "cross_domain_value": len(set(domains)) / len(domains),
                "created_at": now_ts()
            }
            
            self.fusions.append(fusion)
            self.logger.info(f"Created fusion {fusion_id} from {len(ideas)} ideas")
            return fusion
        except Exception as e:
            self.logger.exception(f"Idea fusion failed: {e}")
            raise
    
    def _create_fusion_description(self, ideas: List[Dict[str, Any]], domains: List[str]) -> str:
        """Create description for fused idea"""
        domain_list = ", ".join(set(domains))
        return f"A cross-domain innovation combining insights from {domain_list}"
    
    def _calculate_fusion_novelty(self, ideas: List[Dict[str, Any]]) -> float:
        """Calculate novelty score for fusion"""
        # Average novelty of source ideas plus bonus for cross-domain
        avg_novelty = sum(idea.get("novelty_score", 0.5) for idea in ideas) / len(ideas)
        domains = set(idea.get("domain") for idea in ideas)
        cross_domain_bonus = 0.1 * (len(domains) - 1)
        return min(1.0, avg_novelty + cross_domain_bonus)
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute idea fusion tasks"""
        action = task.get("action")
        
        if action == "fuse":
            fusion = self.fuse_ideas(task["ideas"])
            return {"status": "success", "fusion": fusion}
        elif action == "list":
            return {"status": "success", "fusions": self.fusions}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
