"""
Advanced Reasoning Engine
Implements cutting-edge reasoning architectures: CoT, ToT, Self-Consistency, ReAct, Reflexion
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from modules.llm import LLMEngine

logger = logging.getLogger(__name__)


class ReasoningMethod(Enum):
    """Reasoning methods available"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    SELF_CONSISTENCY = "self_consistency"
    REACT = "react"
    REFLEXION = "reflexion"


@dataclass
class ReasoningStep:
    """A single step in reasoning"""
    step_number: int
    thought: str
    confidence: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class ReasoningPath:
    """A complete reasoning path"""
    path_id: str
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    evaluation_score: float = 0.0


class AdvancedReasoningEngine:
    """
    Advanced reasoning engine with multiple reasoning architectures.
    
    Implements:
    - Chain-of-Thought (CoT): Step-by-step reasoning
    - Tree-of-Thought (ToT): Multiple reasoning paths, best selection
    - Self-Consistency: Multiple answers, consensus
    - ReAct: Reasoning + Acting (interleaved)
    - Reflexion: Self-critique and improvement loop
    """
    
    def __init__(self, llm_engine):
        """Initialize with LLMEngine (can be LLMEngine or LlamaEngine)"""
        self.llm_engine = llm_engine
        self.reasoning_history: List[Dict[str, Any]] = []
        
    async def reason(
        self,
        query: str,
        method: ReasoningMethod = ReasoningMethod.CHAIN_OF_THOUGHT,
        context: Optional[Dict[str, Any]] = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Reason about a query using specified method.
        
        Args:
            query: The question or problem to reason about
            method: Reasoning method to use
            context: Additional context
            domain: Domain context (construction, game_dev, etc.)
        
        Returns:
            Dict with reasoning steps, answer, and confidence
        """
        if method == ReasoningMethod.CHAIN_OF_THOUGHT:
            return await self.chain_of_thought(query, context, domain)
        elif method == ReasoningMethod.TREE_OF_THOUGHT:
            return await self.tree_of_thought(query, context, domain)
        elif method == ReasoningMethod.SELF_CONSISTENCY:
            return await self.self_consistency(query, context, domain)
        elif method == ReasoningMethod.REACT:
            return await self.react(query, context, domain)
        elif method == ReasoningMethod.REFLEXION:
            return await self.reflexion(query, context, domain)
        else:
            raise ValueError(f"Unknown reasoning method: {method}")
    
    async def chain_of_thought(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Chain-of-Thought reasoning: Step-by-step reasoning.
        
        Generates intermediate reasoning steps before final answer.
        """
        logger.info(f"🧠 Chain-of-Thought reasoning: {query[:60]}...")
        
        prompt = f"""You are an expert {domain} professional. Solve this problem step by step.

Problem: {query}

Think through this step by step:
1. First, understand what is being asked
2. Identify the key information needed
3. Break down the problem into smaller parts
4. Solve each part systematically
5. Combine the solutions
6. Verify your answer

Provide your reasoning step by step, then give your final answer.

Reasoning:
"""
        
        response = await self.llm_engine.generate(
            prompt=prompt,
            context=context or {},
            task=f"{domain}_reasoning"
        )
        
        # Parse response to extract steps
        steps = self._parse_cot_response(response)
        
        return {
            "method": "chain_of_thought",
            "query": query,
            "steps": steps,
            "final_answer": steps[-1]["answer"] if steps else str(response),
            "confidence": self._calculate_confidence(steps),
            "reasoning": response
        }
    
    async def tree_of_thought(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        domain: str = "general",
        max_depth: int = 3,
        branching_factor: int = 3
    ) -> Dict[str, Any]:
        """
        Tree-of-Thought reasoning: Generate multiple reasoning paths, evaluate, select best.
        
        Creates a tree of reasoning paths, evaluates each, selects the best.
        """
        logger.info(f"🌳 Tree-of-Thought reasoning: {query[:60]}...")
        
        # Generate multiple reasoning paths
        paths = []
        for i in range(branching_factor):
            path = await self._generate_reasoning_path(query, context, domain, max_depth)
            # Evaluate path
            path.evaluation_score = await self._evaluate_path(path, query, domain)
            paths.append(path)
        
        # Select best path
        best_path = max(paths, key=lambda p: p.evaluation_score)
        
        return {
            "method": "tree_of_thought",
            "query": query,
            "paths_generated": len(paths),
            "best_path": {
                "path_id": best_path.path_id,
                "steps": [{"step": s.step_number, "thought": s.thought, "confidence": s.confidence} 
                          for s in best_path.steps],
                "final_answer": best_path.final_answer,
                "confidence": best_path.confidence,
                "evaluation_score": best_path.evaluation_score
            },
            "all_paths": [
                {
                    "path_id": p.path_id,
                    "evaluation_score": p.evaluation_score,
                    "final_answer": p.final_answer
                }
                for p in paths
            ]
        }
    
    async def self_consistency(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        domain: str = "general",
        num_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Self-Consistency: Generate multiple answers, find consensus.
        
        Generates multiple independent answers, finds the most consistent one.
        """
        logger.info(f"🔄 Self-Consistency reasoning: {query[:60]}... (generating {num_samples} samples)")
        
        # Generate multiple independent answers
        answers = []
        for i in range(num_samples):
            response = await self.llm_engine.generate(
                prompt=f"Answer this {domain} question: {query}",
                context=context or {},
                task=f"{domain}_reasoning"
            )
            
            # Extract answer from response
            answer = self._extract_answer(response)
            answers.append({
                "sample": i + 1,
                "answer": answer,
                "raw_response": str(response)
            })
        
        # Find consensus answer
        consensus = self._find_consensus(answers)
        
        return {
            "method": "self_consistency",
            "query": query,
            "num_samples": num_samples,
            "answers": answers,
            "consensus_answer": consensus["answer"],
            "consensus_confidence": consensus["confidence"],
            "agreement_rate": consensus["agreement_rate"]
        }
    
    async def react(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        domain: str = "general",
        max_iterations: int = 5
    ) -> Dict[str, Any]:
        """
        ReAct (Reasoning + Acting): Interleaved reasoning and tool use.
        
        Alternates between reasoning about the problem and taking actions.
        """
        logger.info(f"⚡ ReAct reasoning: {query[:60]}...")
        
        reasoning_steps = []
        actions_taken = []
        observation = None
        
        for iteration in range(max_iterations):
            # Think
            think_prompt = f"""You are solving: {query}
            
Current context: {json.dumps(context or {}, indent=2)}
Previous observations: {observation or "None"}

Think about what to do next. What information do you need? What action should you take?

Thought:"""
            
            thought = await self.llm_engine.generate(
                prompt=think_prompt,
                context=context or {},
                task=f"{domain}_reasoning"
            )
            reasoning_steps.append({
                "iteration": iteration + 1,
                "thought": str(thought)
            })
            
            # Act (simplified - in real implementation, would use tools)
            action_prompt = f"""Based on your thought: {thought}

What action should you take? (e.g., "search for information", "calculate", "analyze")

Action:"""
            
            action = await self.llm_engine.generate(
                prompt=action_prompt,
                context=context or {},
                task=f"{domain}_action"
            )
            actions_taken.append({
                "iteration": iteration + 1,
                "action": str(action)
            })
            
            # Observe (simplified - in real implementation, would execute action)
            observation = f"Action '{action}' completed. Result: [simulated result]"
            
            # Check if we have enough information
            if iteration >= 2:  # Simplified stopping condition
                break
        
        # Final answer
        final_prompt = f"""Based on your reasoning and actions:

Query: {query}
Thoughts: {json.dumps(reasoning_steps, indent=2)}
Actions: {json.dumps(actions_taken, indent=2)}
Observations: {observation}

What is your final answer?

Answer:"""
        
        final_answer = await self.llm_engine.generate(
            prompt=final_prompt,
            context=context or {},
            task=f"{domain}_reasoning"
        )
        
        return {
            "method": "react",
            "query": query,
            "reasoning_steps": reasoning_steps,
            "actions_taken": actions_taken,
            "observations": observation,
            "final_answer": str(final_answer),
            "iterations": len(reasoning_steps)
        }
    
    async def reflexion(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        domain: str = "general",
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Reflexion: Self-critique and improvement loop.
        
        Generates answer, critiques it, improves, repeats.
        """
        logger.info(f"🪞 Reflexion reasoning: {query[:60]}...")
        
        iterations = []
        current_answer = None
        
        for iteration in range(max_iterations):
            # Generate answer
            if iteration == 0:
                prompt = f"Answer this {domain} question: {query}"
            else:
                prompt = f"""Answer this {domain} question: {query}

Previous attempt: {current_answer}
Critique: {iterations[-1]['critique']}

Improve your answer based on the critique."""
            
            answer = await self.llm_engine.generate(
                prompt=prompt,
                context=context or {},
                task=f"{domain}_reasoning"
            )
            current_answer = str(answer)
            
            # Critique answer
            critique_prompt = f"""Critique this answer to the question "{query}":

Answer: {current_answer}

Provide constructive criticism. What's good? What could be improved? What's missing?

Critique:"""
            
            critique = await self.llm_engine.generate(
                prompt=critique_prompt,
                context=context or {},
                task=f"{domain}_critique"
            )
            
            iterations.append({
                "iteration": iteration + 1,
                "answer": current_answer,
                "critique": str(critique)
            })
            
            # Check if critique suggests major improvements needed
            if "good" in str(critique).lower() and "excellent" in str(critique).lower():
                break
        
        return {
            "method": "reflexion",
            "query": query,
            "iterations": iterations,
            "final_answer": current_answer,
            "improvement_cycles": len(iterations)
        }
    
    # Helper methods
    
    async def _generate_reasoning_path(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        domain: str,
        max_depth: int
    ) -> ReasoningPath:
        """Generate a single reasoning path"""
        import uuid
        path_id = str(uuid.uuid4())[:8]
        steps = []
        
        for depth in range(max_depth):
            step_prompt = f"""Reasoning step {depth + 1} for: {query}

Previous steps: {json.dumps([s.thought for s in steps], indent=2) if steps else "None"}

What is the next logical step in your reasoning?

Step {depth + 1}:"""
            
            response = await self.llm_engine.generate(
                prompt=step_prompt,
                context=context or {},
                task=f"{domain}_reasoning"
            )
            
            steps.append(ReasoningStep(
                step_number=depth + 1,
                thought=str(response),
                confidence=0.8  # Simplified
            ))
        
        # Final answer
        final_prompt = f"""Based on your reasoning steps: {json.dumps([s.thought for s in steps], indent=2)}

What is your final answer to: {query}?

Final Answer:"""
        
        final_answer = await self.llm_engine.generate(
            prompt=final_prompt,
            context=context or {},
            task=f"{domain}_reasoning"
        )
        
        return ReasoningPath(
            path_id=path_id,
            steps=steps,
            final_answer=str(final_answer),
            confidence=0.8
        )
    
    async def _evaluate_path(
        self,
        path: ReasoningPath,
        query: str,
        domain: str
    ) -> float:
        """Evaluate a reasoning path"""
        eval_prompt = f"""Evaluate this reasoning path for answering: {query}

Reasoning steps:
{json.dumps([{"step": s.step_number, "thought": s.thought} for s in path.steps], indent=2)}

Final answer: {path.final_answer}

Rate this reasoning path from 0.0 to 1.0 based on:
- Logical coherence
- Completeness
- Correctness
- Clarity

Score (0.0-1.0):"""
        
        response = await self.llm_engine.generate(
            prompt=eval_prompt,
            context={},
            task="evaluation"
        )
        
        # Extract score from response
        import re
        score_match = re.search(r'(\d+\.?\d*)', str(response))
        if score_match:
            return float(score_match.group(1))
        return 0.5
    
    def _parse_cot_response(self, response: Any) -> List[Dict[str, Any]]:
        """Parse Chain-of-Thought response into steps"""
        response_str = str(response)
        steps = []
        
        # Try to extract numbered steps
        import re
        step_pattern = r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]+)*)'
        matches = re.findall(step_pattern, response_str)
        
        for i, (num, content) in enumerate(matches):
            steps.append({
                "step": int(num),
                "thought": content.strip(),
                "answer": content.strip() if i == len(matches) - 1 else None
            })
        
        if not steps:
            # Fallback: treat entire response as one step
            steps.append({
                "step": 1,
                "thought": response_str,
                "answer": response_str
            })
        
        return steps
    
    def _calculate_confidence(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate confidence from reasoning steps"""
        if not steps:
            return 0.5
        
        # More steps = higher confidence (up to a point)
        num_steps = len(steps)
        base_confidence = min(0.9, 0.5 + (num_steps * 0.1))
        
        return base_confidence
    
    def _extract_answer(self, response: Any) -> str:
        """Extract answer from response"""
        response_str = str(response)
        
        # Try to find answer markers
        answer_patterns = [
            r'answer[:\s]+(.+?)(?:\n\n|\Z)',
            r'final[:\s]+(.+?)(?:\n\n|\Z)',
            r'conclusion[:\s]+(.+?)(?:\n\n|\Z)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response_str, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: return first 200 chars
        return response_str[:200].strip()
    
    def _find_consensus(self, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find consensus among multiple answers"""
        from collections import Counter
        
        # Extract answer texts
        answer_texts = [a["answer"] for a in answers]
        
        # Find most common answer (simplified - in reality would use semantic similarity)
        answer_counter = Counter(answer_texts)
        most_common = answer_counter.most_common(1)[0]
        
        consensus_answer = most_common[0]
        agreement_count = most_common[1]
        agreement_rate = agreement_count / len(answers)
        
        return {
            "answer": consensus_answer,
            "confidence": agreement_rate,
            "agreement_rate": agreement_rate,
            "votes": agreement_count,
            "total": len(answers)
        }

