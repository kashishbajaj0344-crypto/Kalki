"""
Real-Time Learning System
Enables Kalki to learn and adapt in real-time from every interaction.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from modules.llm import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class LearningExample:
    """A learning example for few-shot learning"""
    input: str
    output: str
    context: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningUpdate:
    """An update to the learning system"""
    update_type: str  # "feedback", "correction", "preference"
    domain: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class RealTimeLearningSystem:
    """
    Real-time learning system that enables Kalki to:
    - Learn from feedback in real-time
    - Adapt to new tasks with few-shot learning
    - Transfer knowledge between domains instantly
    - Ask clarifying questions when uncertain (active learning)
    """
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.learning_examples: Dict[str, List[LearningExample]] = {}  # domain -> examples
        self.feedback_history: List[LearningUpdate] = []
        self.adaptation_cache: Dict[str, Any] = {}
        
        # Few-shot learning cache
        self.few_shot_cache: Dict[str, List[LearningExample]] = {}
        
        # Active learning: track uncertainty
        self.uncertainty_threshold = 0.6
        self.uncertainty_history: List[Dict[str, Any]] = []
    
    async def online_update(
        self,
        feedback: Dict[str, Any],
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Update model in real-time from feedback.
        
        Args:
            feedback: Feedback data with 'input', 'output', 'quality_score', etc.
            domain: Domain context
        
        Returns:
            Update result with success status
        """
        logger.info(f"🔄 Real-time learning update for {domain} domain")
        
        # Store feedback
        update = LearningUpdate(
            update_type=feedback.get('type', 'feedback'),
            domain=domain,
            data=feedback
        )
        self.feedback_history.append(update)
        
        # Create learning example
        example = LearningExample(
            input=feedback.get('input', ''),
            output=feedback.get('output', ''),
            context=feedback.get('context', {}),
            quality_score=feedback.get('quality_score', 1.0)
        )
        
        # Add to domain examples
        if domain not in self.learning_examples:
            self.learning_examples[domain] = []
        self.learning_examples[domain].append(example)
        
        # Update few-shot cache
        if domain not in self.few_shot_cache:
            self.few_shot_cache[domain] = []
        
        # Keep top N examples by quality
        self.few_shot_cache[domain].append(example)
        self.few_shot_cache[domain].sort(key=lambda x: x.quality_score, reverse=True)
        self.few_shot_cache[domain] = self.few_shot_cache[domain][:10]  # Keep top 10
        
        # Update adaptation cache
        cache_key = f"{domain}_{feedback.get('task_type', 'general')}"
        if cache_key not in self.adaptation_cache:
            self.adaptation_cache[cache_key] = {
                "examples": [],
                "patterns": {},
                "last_updated": datetime.now()
            }
        
        self.adaptation_cache[cache_key]["examples"].append(example)
        self.adaptation_cache[cache_key]["last_updated"] = datetime.now()
        
        logger.info(f"✅ Learning update stored: {len(self.learning_examples[domain])} examples for {domain}")
        
        return {
            "status": "success",
            "domain": domain,
            "total_examples": len(self.learning_examples[domain]),
            "cache_updated": True
        }
    
    async def few_shot_adapt(
        self,
        examples: List[Dict[str, Any]],
        task: str,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Adapt to new task from few examples (1-5 examples).
        
        Args:
            examples: List of example dicts with 'input' and 'output'
            task: Task description
            domain: Domain context
        
        Returns:
            Adaptation result with learned patterns
        """
        logger.info(f"📚 Few-shot learning: {len(examples)} examples for task: {task[:50]}...")
        
        # Format examples for few-shot prompting
        example_text = ""
        for i, ex in enumerate(examples[:5], 1):  # Limit to 5 examples
            example_text += f"Example {i}:\n"
            example_text += f"Input: {ex.get('input', '')}\n"
            example_text += f"Output: {ex.get('output', '')}\n\n"
        
        # Generate few-shot prompt
        prompt = f"""Based on these examples, learn the pattern and apply it to new tasks.

Examples:
{example_text}

Task: {task}

Based on the pattern in the examples, provide the output for this task."""
        
        # Use LLM to learn pattern
        result = await self.llm_engine.generate(
            prompt=prompt,
            context={"domain": domain, "task": task},
            task="few_shot_learning"
        )
        
        # Extract learned pattern
        pattern_prompt = f"""Extract the pattern from these examples:

{example_text}

What is the common pattern or rule?"""
        
        pattern = await self.llm_engine.generate(
            prompt=pattern_prompt,
            context={"domain": domain},
            task="pattern_extraction"
        )
        
        # Store pattern
        pattern_key = f"{domain}_{task}"
        self.adaptation_cache[pattern_key] = {
            "pattern": str(pattern),
            "examples": examples,
            "learned_at": datetime.now()
        }
        
        return {
            "status": "success",
            "task": task,
            "learned_pattern": str(pattern),
            "result": str(result),
            "examples_used": len(examples)
        }
    
    async def active_learning_query(
        self,
        task: str,
        current_answer: str,
        confidence: float,
        domain: str = "general"
    ) -> Optional[str]:
        """
        Generate clarifying question when uncertain (active learning).
        
        Args:
            task: Task description
            current_answer: Current answer (may be uncertain)
            confidence: Confidence score (0-1)
            domain: Domain context
        
        Returns:
            Clarifying question if uncertain, None otherwise
        """
        if confidence >= self.uncertainty_threshold:
            return None  # Confident enough, no question needed
        
        logger.info(f"❓ Active learning: Generating clarifying question (confidence: {confidence:.2f})")
        
        # Track uncertainty
        self.uncertainty_history.append({
            "task": task,
            "confidence": confidence,
            "domain": domain,
            "timestamp": datetime.now()
        })
        
        # Generate clarifying question
        prompt = f"""I'm working on this task: {task}

My current answer: {current_answer}

I'm not very confident (confidence: {confidence:.0%}). What clarifying question should I ask to better understand what the user wants?

Generate a single, specific clarifying question that would help me provide a better answer."""
        
        question = await self.llm_engine.generate(
            prompt=prompt,
            context={"domain": domain, "task": task},
            task="active_learning"
        )
        
        return str(question).strip()
    
    async def transfer_knowledge(
        self,
        source_domain: str,
        target_domain: str,
        knowledge_type: str = "pattern"
    ) -> Dict[str, Any]:
        """
        Transfer knowledge between domains instantly.
        
        Args:
            source_domain: Source domain
            target_domain: Target domain
            knowledge_type: Type of knowledge to transfer
        
        Returns:
            Transfer result
        """
        logger.info(f"🔄 Transferring knowledge: {source_domain} → {target_domain}")
        
        # Get source knowledge
        source_examples = self.learning_examples.get(source_domain, [])
        if not source_examples:
            return {
                "status": "error",
                "message": f"No knowledge found in {source_domain}"
            }
        
        # Adapt examples to target domain
        adapted_examples = []
        for ex in source_examples[:5]:  # Use top 5 examples
            # Use LLM to adapt example to target domain
            adaptation_prompt = f"""Adapt this {source_domain} example to {target_domain}:

Source ({source_domain}):
Input: {ex.input}
Output: {ex.output}

Adapt it to {target_domain} domain while preserving the pattern."""
            
            adapted = await self.llm_engine.generate(
                prompt=adaptation_prompt,
                context={"source": source_domain, "target": target_domain},
                task="knowledge_transfer"
            )
            
            # Parse adapted example
            adapted_examples.append({
                "input": adapted,  # Simplified - would parse properly
                "output": adapted,
                "source": source_domain,
                "target": target_domain
            })
        
        # Store in target domain
        if target_domain not in self.learning_examples:
            self.learning_examples[target_domain] = []
        
        for ex in adapted_examples:
            self.learning_examples[target_domain].append(
                LearningExample(
                    input=ex["input"],
                    output=ex["output"],
                    context={"transferred_from": source_domain}
                )
            )
        
        return {
            "status": "success",
            "source": source_domain,
            "target": target_domain,
            "examples_transferred": len(adapted_examples)
        }
    
    def get_learning_stats(self, domain: str = None) -> Dict[str, Any]:
        """Get learning statistics"""
        if domain:
            examples = self.learning_examples.get(domain, [])
            return {
                "domain": domain,
                "total_examples": len(examples),
                "avg_quality": sum(e.quality_score for e in examples) / len(examples) if examples else 0,
                "uncertainty_queries": len([u for u in self.uncertainty_history if u.get("domain") == domain])
            }
        else:
            return {
                "total_domains": len(self.learning_examples),
                "total_examples": sum(len(ex) for ex in self.learning_examples.values()),
                "total_feedback": len(self.feedback_history),
                "uncertainty_queries": len(self.uncertainty_history)
            }

