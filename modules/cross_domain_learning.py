"""
Cross-Domain Learning
Facilitates knowledge transfer between domains.

Example:
- Construction learns project management from game dev
- Robotics learns simulation from aerospace
- All domains learn estimation from construction
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from modules.domains.domain_registry import DomainRegistry
from modules.meta_learning_system import MetaLearningSystem
from modules.llm import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class TransferableSkill:
    """A skill that can be transferred between domains"""
    skill_name: str
    source_domains: List[str]
    target_domains: List[str]
    transfer_confidence: float
    adaptation_notes: str = ""


class CrossDomainLearning:
    """
    Facilitates learning across domains.
    
    Example:
    - Construction learns project management from game dev
    - Robotics learns simulation from aerospace
    - All domains learn estimation from construction
    """
    
    def __init__(
        self,
        domain_registry: DomainRegistry,
        meta_learning: MetaLearningSystem,
        llm_engine: LLMEngine
    ):
        self.domain_registry = domain_registry
        self.meta_learning = meta_learning
        self.llm_engine = llm_engine
        self.transferable_skills: Dict[str, TransferableSkill] = {}
        self.transfer_history: List[Dict[str, Any]] = []
        self._identify_transferable_skills()
    
    def _identify_transferable_skills(self):
        """Identify skills that transfer across domains"""
        self.transferable_skills = {
            "project_management": TransferableSkill(
                skill_name="project_management",
                source_domains=["construction", "game_dev", "robotics"],
                target_domains=["construction", "game_dev", "robotics", "aerospace", "power_systems"],
                transfer_confidence=0.9
            ),
            "estimation": TransferableSkill(
                skill_name="estimation",
                source_domains=["construction"],
                target_domains=["game_dev", "robotics", "aerospace", "power_systems"],
                transfer_confidence=0.8
            ),
            "simulation": TransferableSkill(
                skill_name="simulation",
                source_domains=["robotics", "aerospace"],
                target_domains=["robotics", "aerospace", "power_systems"],
                transfer_confidence=0.85
            ),
            "design_patterns": TransferableSkill(
                skill_name="design_patterns",
                source_domains=["game_dev", "robotics"],
                target_domains=["game_dev", "robotics", "aerospace"],
                transfer_confidence=0.75
            ),
            "safety_analysis": TransferableSkill(
                skill_name="safety_analysis",
                source_domains=["construction", "aerospace", "power_systems"],
                target_domains=["construction", "aerospace", "power_systems", "robotics"],
                transfer_confidence=0.9
            ),
        }
    
    async def transfer_skill(
        self,
        source_domain: str,
        target_domain: str,
        skill: str
    ) -> Dict[str, Any]:
        """
        Transfer a skill from one domain to another using Llama 3.1 8B.
        
        Example: Transfer "estimation" from construction to game dev
        """
        if skill not in self.transferable_skills:
            return {"error": f"Skill {skill} not transferable"}
        
        skill_info = self.transferable_skills[skill]
        
        if source_domain not in skill_info.source_domains:
            return {"error": f"Source domain {source_domain} doesn't have {skill}"}
        
        if target_domain not in skill_info.target_domains:
            return {"error": f"Target domain {target_domain} cannot receive {skill}"}
        
        # Get knowledge from source domain
        source_domain_obj = self.domain_registry.get_domain(source_domain)
        if not source_domain_obj:
            return {"error": f"Source domain {source_domain} not found"}
        
        # Extract knowledge (this would need to be implemented in BaseDomain)
        source_knowledge = await self._extract_domain_knowledge(source_domain_obj, skill)
        
        # Adapt knowledge for target domain using Llama 3.1 8B
        adapted_knowledge = await self._adapt_knowledge(
            source_knowledge=source_knowledge,
            source_domain=source_domain,
            target_domain=target_domain,
            skill=skill
        )
        
        # Apply to target domain
        target_domain_obj = self.domain_registry.get_domain(target_domain)
        if not target_domain_obj:
            return {"error": f"Target domain {target_domain} not found"}
        
        await self._apply_knowledge(target_domain_obj, skill, adapted_knowledge)
        
        # Record transfer
        transfer_record = {
            "skill": skill,
            "source": source_domain,
            "target": target_domain,
            "knowledge_transferred": len(adapted_knowledge),
            "confidence": skill_info.transfer_confidence,
            "timestamp": datetime.now().isoformat()
        }
        self.transfer_history.append(transfer_record)
        
        logger.info(f"✅ Transferred {skill} from {source_domain} to {target_domain}")
        
        return transfer_record
    
    async def _extract_domain_knowledge(
        self,
        domain_obj: Any,
        skill: str
    ) -> List[Dict[str, Any]]:
        """Extract knowledge from a domain"""
        # This would need to be implemented in BaseDomain
        # For now, return empty list
        return []
    
    async def _adapt_knowledge(
        self,
        source_knowledge: List[Dict],
        source_domain: str,
        target_domain: str,
        skill: str
    ) -> List[Dict]:
        """Adapt knowledge from one domain to another using Llama 3.1 8B"""
        adapted = []
        
        for item in source_knowledge:
            adaptation_prompt = f"""Adapt this {skill} knowledge from {source_domain} domain to {target_domain} domain.

Source Knowledge:
{item}

Provide the adapted version for {target_domain} domain, maintaining the core principles but adjusting for domain-specific context."""
            
            # Use Llama 3.1 8B to adapt
            adapted_item = await self.llm_engine.generate(
                prompt=adaptation_prompt,
                max_tokens=500,
                temperature=0.7
            )
            
            if isinstance(adapted_item, dict):
                adapted_text = adapted_item.get("text", str(adapted_item))
            else:
                adapted_text = str(adapted_item)
            
            adapted.append({
                "original": item,
                "adapted": adapted_text,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "skill": skill
            })
        
        return adapted
    
    async def _apply_knowledge(
        self,
        target_domain_obj: Any,
        skill: str,
        adapted_knowledge: List[Dict]
    ):
        """Apply adapted knowledge to target domain"""
        # This would need to be implemented in BaseDomain
        logger.info(f"Applied {len(adapted_knowledge)} knowledge items to {target_domain_obj.name}")
    
    def get_transferable_skills(self) -> Dict[str, TransferableSkill]:
        """Get all transferable skills"""
        return self.transferable_skills
    
    def get_transfer_history(self) -> List[Dict[str, Any]]:
        """Get history of knowledge transfers"""
        return self.transfer_history


