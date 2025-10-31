"""
Phase 16 - Iterative Reasoning & Chaining
Multi-step reasoning with checkpointing and rule-based inference.
Enhanced with memory integration, meta-reasoning, and LLM support.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..config import DIRS
from ..agents.memory import EpisodicMemory
from ..llm import ask_kalki

logger = logging.getLogger("Kalki.Reasoner")


class StepType(Enum):
    """Type of reasoning step."""
    OBSERVATION = "observation"
    INFERENCE = "inference"
    DEDUCTION = "deduction"
    CONCLUSION = "conclusion"
    CHECKPOINT = "checkpoint"
    LLM_ASSISTED = "llm_assisted"


@dataclass
class Step:
    """Represents a single step in reasoning trace."""
    step_id: int
    step_type: StepType
    content: str
    facts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningTrace:
    """Trace of reasoning steps with enhanced metadata."""
    problem: str
    steps: List[Step] = field(default_factory=list)
    conclusion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_step(self, step_type: StepType, content: str, facts: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> Step:
        """Add a step to the trace."""
        step = Step(
            step_id=len(self.steps),
            step_type=step_type,
            content=content,
            facts=facts or [],
            metadata=metadata or {}
        )
        self.steps.append(step)
        return step

    def get_facts(self) -> List[str]:
        """Get all facts accumulated in the trace."""
        facts = []
        for step in self.steps:
            facts.extend(step.facts)
        return facts

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            'problem': self.problem,
            'steps': [
                {
                    'step_id': s.step_id,
                    'step_type': s.step_type.value,
                    'content': s.content,
                    'facts': s.facts,
                    'metadata': s.metadata,
                    'timestamp': s.timestamp.isoformat()
                }
                for s in self.steps
            ],
            'conclusion': self.conclusion,
            'metadata': self.metadata,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

    def save_trace(self, path: Path) -> None:
        """
        Save reasoning trace to file for persistence.

        Args:
            path: Path to save the trace
        """
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.debug(f"[Reasoner] Saved trace to {path}")
        except Exception as e:
            logger.warning(f"[Reasoner] Failed to save trace: {e}")


@dataclass
class Rule:
    """Represents an if-then rule for inference."""
    name: str
    conditions: List[Callable[[List[str]], bool]]  # Functions that check facts
    action: Callable[[List[str]], List[str]]  # Function that generates new facts
    description: str = ""

    def matches(self, facts: List[str]) -> bool:
        """Check if all conditions are satisfied by the facts."""
        return all(condition(facts) for condition in self.conditions)

    def apply(self, facts: List[str]) -> List[str]:
        """Apply the rule and return new facts."""
        if self.matches(facts):
            return self.action(facts)
        return []


class InferenceEngine:
    """Enhanced forward and backward-chaining inference engine."""

    def __init__(self):
        """Initialize inference engine."""
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self.rules.append(rule)

    def infer(self, facts: List[str], max_iterations: int = 10) -> List[str]:
        """
        Apply forward chaining to derive new facts.

        Args:
            facts: Initial fact base
            max_iterations: Maximum number of inference iterations

        Returns:
            All facts (initial + derived)
        """
        all_facts = set(facts)

        for _ in range(max_iterations):
            new_facts = set()

            # Try to apply each rule
            for rule in self.rules:
                derived = rule.apply(list(all_facts))
                new_facts.update(derived)

            # If no new facts were derived, we're done
            if not new_facts - all_facts:
                break

            all_facts.update(new_facts)

        return list(all_facts)

    def infer_backward(self, goal: str, facts: List[str], max_iterations: int = 10) -> List[str]:
        """
        Apply backward chaining to achieve a goal.

        Args:
            goal: Goal to achieve
            facts: Initial fact base
            max_iterations: Maximum number of inference iterations

        Returns:
            All facts (initial + derived) that help achieve the goal
        """
        all_facts = set(facts)
        goal_achieved = False

        for _ in range(max_iterations):
            if goal_achieved:
                break

            new_facts = set()

            # Try rules that could help achieve the goal
            for rule in self.rules:
                # Check if this rule could produce facts related to the goal
                if goal.lower() in rule.description.lower() or goal.lower() in rule.name.lower():
                    if rule.matches(list(all_facts)):
                        derived = rule.apply(list(all_facts))
                        new_facts.update(derived)
                        # Check if goal is achieved
                        if any(goal.lower() in fact.lower() for fact in derived):
                            goal_achieved = True

            # If no new facts were derived, we're done
            if not new_facts - all_facts:
                break

            all_facts.update(new_facts)

        return list(all_facts)


class Reasoner:
    """Multi-step chain-of-thought reasoner with checkpointing and LLM support."""

    def __init__(self, memory_store: Optional[EpisodicMemory] = None,
                 enable_llm: bool = False):
        """
        Initialize reasoner with enhanced capabilities.

        Args:
            memory_store: Episodic memory for checkpointing
            enable_llm: Whether to enable LLM-assisted reasoning
        """
        self.memory_store = memory_store
        self.enable_llm = enable_llm
        self.inference_engine = InferenceEngine()
        self._setup_default_rules()

        logger.info("[Reasoner] Initialized with memory integration and LLM support")

    async def reason(self, problem: str, max_steps: int = 10,
                    checkpoint_every: int = 3, mode: str = "forward") -> ReasoningTrace:
        """
        Perform multi-step reasoning on a problem with enhanced features.

        Args:
            problem: Problem description
            max_steps: Maximum number of reasoning steps
            checkpoint_every: Checkpoint trace every N steps
            mode: Reasoning mode ("forward" or "backward")

        Returns:
            ReasoningTrace with reasoning steps and metadata
        """
        logger.info(f"[Reasoner] Started reasoning for problem: {problem[:50]}...")
        start_time = datetime.now()

        trace = ReasoningTrace(problem=problem)

        # Initial observation
        trace.add_step(
            StepType.OBSERVATION,
            f"Problem: {problem}",
            facts=[f"problem:{problem}"]
        )

        # Parse problem into initial facts
        initial_facts = self._extract_facts(problem)
        trace.add_step(
            StepType.OBSERVATION,
            f"Extracted {len(initial_facts)} initial facts",
            facts=initial_facts
        )

        # Perform reasoning steps
        current_facts = initial_facts.copy()
        goal_achieved = False

        for step_num in range(max_steps):
            derived_facts = []

            if mode == "backward":
                # Backward chaining toward a goal
                goal = self._extract_goal(problem)
                new_facts = self.inference_engine.infer_backward(goal, current_facts, max_iterations=1)
                derived_facts = [f for f in new_facts if f not in current_facts]
                if derived_facts and any(goal.lower() in fact.lower() for fact in derived_facts):
                    goal_achieved = True
            else:
                # Forward chaining
                new_facts = self.inference_engine.infer(current_facts, max_iterations=1)
                derived_facts = [f for f in new_facts if f not in current_facts]

            if derived_facts:
                trace.add_step(
                    StepType.INFERENCE,
                    f"Derived {len(derived_facts)} new facts using {mode} chaining",
                    facts=derived_facts,
                    metadata={"reasoning_mode": mode}
                )
                current_facts = new_facts
            else:
                # Try LLM assistance if enabled and no progress
                if self.enable_llm and not derived_facts:
                    llm_facts = await self._llm_assisted_inference(current_facts, problem)
                    if llm_facts:
                        trace.add_step(
                            StepType.LLM_ASSISTED,
                            f"LLM-assisted inference derived {len(llm_facts)} facts",
                            facts=llm_facts,
                            metadata={"llm_used": True}
                        )
                        current_facts.extend(llm_facts)
                        continue

                # No new facts, reasoning is complete
                break

            # Checkpoint if needed
            if (step_num + 1) % checkpoint_every == 0:
                await self._checkpoint(trace)
                trace.add_step(StepType.CHECKPOINT, f"Checkpoint at step {step_num + 1}")

            # Check if goal achieved in backward mode
            if mode == "backward" and goal_achieved:
                break

        # Draw conclusion
        conclusion = self._draw_conclusion(trace.get_facts(), mode, goal_achieved)
        trace.conclusion = conclusion
        trace.add_step(StepType.CONCLUSION, conclusion)
        trace.completed_at = datetime.now()

        # Add meta-reasoning metadata for self-reflection
        trace.metadata["meta"] = {
            "steps_taken": len(trace.steps),
            "facts_derived": len(trace.get_facts()),
            "reasoning_mode": mode,
            "goal_achieved": goal_achieved if mode == "backward" else None,
            "confidence": round(min(1.0, len(trace.get_facts()) / 10), 2),
            "duration_seconds": (trace.completed_at - trace.started_at).total_seconds(),
            "llm_used": any(step.step_type == StepType.LLM_ASSISTED for step in trace.steps)
        }

        # Final checkpoint with enhanced memory integration
        await self._checkpoint(trace)

        # Save trace to persistent storage
        trace_path = Path(DIRS['vector_db']) / f"trace_{trace.started_at.timestamp()}.json"
        trace.save_trace(trace_path)

        logger.info(f"[Reasoner] Completed reasoning: {len(trace.steps)} steps, {len(trace.get_facts())} facts, confidence: {trace.metadata['meta']['confidence']}")

        return trace

    def _extract_facts(self, problem: str) -> List[str]:
        """
        Extract facts from problem description using enhanced heuristics.

        Args:
            problem: Problem description

        Returns:
            List of facts
        """
        facts = []

        # Enhanced keyword-based extraction
        keywords = {
            "is": "property",
            "has": "property",
            "contains": "contains",
            "needs": "requires",
            "requires": "requires",
            "must": "constraint",
            "cannot": "constraint",
            "should": "recommendation",
        }

        words = problem.lower().split()
        problem_sentences = problem.split('.')

        for sentence in problem_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Extract facts from each sentence
            for i, word in enumerate(words):
                if word in keywords:
                    context = " ".join(words[max(0, i-2):min(len(words), i+3)])
                    facts.append(f"fact:{context}")

        # Always add the problem itself as a fact
        if not facts:
            facts.append(f"fact:{problem.lower()}")

        return facts

    def _extract_goal(self, problem: str) -> str:
        """
        Extract the main goal from a problem description.

        Args:
            problem: Problem description

        Returns:
            Extracted goal
        """
        # Simple goal extraction - look for action-oriented phrases
        goal_indicators = ["find", "determine", "solve", "calculate", "prove", "show", "explain"]

        words = problem.lower().split()
        for i, word in enumerate(words):
            if word in goal_indicators:
                # Extract the goal phrase
                goal_phrase = " ".join(words[i:i+5])  # Next 5 words
                return goal_phrase.strip()

        # Fallback: use the whole problem as goal
        return problem.lower()

    def _draw_conclusion(self, facts: List[str], mode: str = "forward",
                        goal_achieved: bool = False) -> str:
        """
        Draw a conclusion from accumulated facts with mode awareness.

        Args:
            facts: All accumulated facts
            mode: Reasoning mode
            goal_achieved: Whether goal was achieved (backward mode)

        Returns:
            Conclusion string
        """
        if not facts:
            return "No conclusion could be drawn due to insufficient facts."

        fact_count = len(facts)

        # Mode-specific conclusions
        if mode == "backward":
            if goal_achieved:
                return f"Goal achieved through backward chaining. Derived {fact_count} supporting facts."
            else:
                return f"Goal not achieved despite backward chaining analysis. Found {fact_count} facts."

        # Forward chaining conclusions
        if any("solved" in f.lower() for f in facts):
            return f"Problem appears solved. Analysis derived {fact_count} facts supporting the solution."
        elif any("impossible" in f.lower() or "contradiction" in f.lower() for f in facts):
            return f"Problem analysis revealed contradictions. {fact_count} facts examined, no valid solution found."
        elif any("uncertain" in f.lower() for f in facts):
            return f"Analysis complete but uncertain. Derived {fact_count} facts with inconclusive results."
        else:
            confidence = min(1.0, fact_count / 10)
            return f"Analysis complete. Derived {fact_count} facts with {confidence:.1%} confidence in the reasoning chain."

    async def _checkpoint(self, trace: ReasoningTrace) -> None:
        """
        Checkpoint reasoning trace to episodic memory with enhanced metadata.

        Args:
            trace: Reasoning trace to checkpoint
        """
        if not self.memory_store:
            return

        try:
            # Use enhanced episodic memory integration
            await self.memory_store.add_event_async(
                event_type="reasoning_checkpoint",
                data=trace.to_dict(),
                metadata={
                    "category": "reasoning_trace",
                    "tags": ["reasoning", "checkpoint", trace.problem[:30], f"mode_{trace.metadata.get('meta', {}).get('reasoning_mode', 'unknown')}"],
                    "steps": len(trace.steps),
                    "facts": len(trace.get_facts()),
                    "confidence": trace.metadata.get("meta", {}).get("confidence", 0.0)
                }
            )
            logger.debug(f"[Reasoner] Checkpointed trace with {len(trace.steps)} steps")
        except Exception as e:
            logger.warning(f"[Reasoner] Failed to checkpoint: {e}")

    async def _llm_assisted_inference(self, current_facts: List[str], problem: str) -> List[str]:
        """
        Use LLM to assist with inference when rule-based reasoning stalls.

        Args:
            current_facts: Current fact base
            problem: Original problem

        Returns:
            List of new facts from LLM assistance
        """
        if not self.enable_llm:
            return []

        try:
            # Prepare prompt for LLM
            facts_str = "\n".join(f"- {fact}" for fact in current_facts[-5:])  # Last 5 facts
            prompt = f"""
Given this problem: {problem}

Current facts established:
{facts_str}

What logical inferences or new facts can be derived from this information?
Provide 1-3 specific, logical conclusions or facts that follow from the current state.
Be concise and focus on logical deductions.

Response format: One fact per line, starting with "fact:" or "inference:"
"""

            # Use async LLM call
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, ask_kalki, prompt)

            # Parse response into facts
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            new_facts = []

            for line in lines:
                if line.startswith(('fact:', 'inference:')):
                    # Clean up the fact
                    fact = line.split(':', 1)[1].strip()
                    if fact:
                        new_facts.append(f"llm_fact:{fact}")

            logger.debug(f"[Reasoner] LLM assisted with {len(new_facts)} new facts")
            return new_facts

        except Exception as e:
            logger.warning(f"[Reasoner] LLM assistance failed: {e}")
            return []

    def _setup_default_rules(self) -> None:
        """Set up enhanced default inference rules."""

        # Rule 1: If problem requires X and we have X, mark as solved
        def has_requirement(facts: List[str]) -> bool:
            return any("requires" in f or "needs" in f for f in facts)

        def check_requirement_met(facts: List[str]) -> bool:
            requirements = [f for f in facts if "requires" in f or "needs" in f]
            available = [f for f in facts if "has" in f or "contains" in f]
            # Simplified check - in practice, this would be more sophisticated
            return len(requirements) > 0 and len(available) > 0

        def mark_solved(facts: List[str]) -> List[str]:
            return ["status:solved", "inference:requirements_met"]

        self.inference_engine.add_rule(Rule(
            name="requirement_satisfaction",
            conditions=[has_requirement, check_requirement_met],
            action=mark_solved,
            description="Mark problem as solved when requirements are met"
        ))

        # Rule 2: Detect contradictions
        def has_contradiction(facts: List[str]) -> bool:
            # Simple contradiction detection
            positive_facts = [f for f in facts if "has" in f or "is" in f]
            negative_facts = [f for f in facts if "cannot" in f or "not" in f]
            # Check for direct contradictions (simplified)
            for pos in positive_facts:
                for neg in negative_facts:
                    if any(word in pos.lower() and word in neg.lower()
                          for word in ["possible", "available", "feasible"]):
                        return True
            return False

        def mark_contradiction(facts: List[str]) -> List[str]:
            return ["status:contradiction", "inference:inconsistent_facts"]

        self.inference_engine.add_rule(Rule(
            name="contradiction_detection",
            conditions=[has_contradiction],
            action=mark_contradiction,
            description="Detect logical contradictions in facts"
        ))

        # Rule 3: Transitive property reasoning
        def has_transitive_relation(facts: List[str]) -> bool:
            return any("contains" in f for f in facts)  # Simplified

        def apply_transitive_reasoning(facts: List[str]) -> List[str]:
            # Very simplified transitive reasoning
            new_facts = []
            containers = [f for f in facts if "contains" in f]
            if len(containers) >= 2:
                new_facts.append("inference:multiple_containers_found")
            return new_facts

        self.inference_engine.add_rule(Rule(
            name="transitive_reasoning",
            conditions=[has_transitive_relation],
            action=apply_transitive_reasoning,
            description="Apply transitive reasoning to relationships"
        ))

# [Kalki v2.3 — reasoner/reasoner.py v1.0]