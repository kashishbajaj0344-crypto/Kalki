"""
Phase 16 - Iterative Reasoning & Chaining
Multi-step reasoning with checkpointing and rule-based inference.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum


class StepType(Enum):
    """Type of reasoning step."""
    OBSERVATION = "observation"
    INFERENCE = "inference"
    DEDUCTION = "deduction"
    CONCLUSION = "conclusion"
    CHECKPOINT = "checkpoint"


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
    """Trace of reasoning steps."""
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
                    'metadata': s.metadata
                }
                for s in self.steps
            ],
            'conclusion': self.conclusion,
            'metadata': self.metadata
        }


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
    """Simple forward-chaining inference engine."""
    
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


class Reasoner:
    """Multi-step chain-of-thought reasoner with checkpointing."""
    
    def __init__(self, memory_store=None):
        """
        Initialize reasoner.
        
        Args:
            memory_store: Optional memory store for checkpointing
        """
        self.memory_store = memory_store
        self.inference_engine = InferenceEngine()
        self._setup_default_rules()
    
    def reason(self, problem: str, max_steps: int = 10,
               checkpoint_every: int = 3) -> ReasoningTrace:
        """
        Perform multi-step reasoning on a problem.
        
        Args:
            problem: Problem description
            max_steps: Maximum number of reasoning steps
            checkpoint_every: Checkpoint trace every N steps
            
        Returns:
            ReasoningTrace with reasoning steps
        """
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
        
        for step_num in range(max_steps):
            # Apply inference engine
            new_facts = self.inference_engine.infer(current_facts, max_iterations=1)
            derived_facts = [f for f in new_facts if f not in current_facts]
            
            if derived_facts:
                trace.add_step(
                    StepType.INFERENCE,
                    f"Derived {len(derived_facts)} new facts",
                    facts=derived_facts
                )
                current_facts = new_facts
            else:
                # No new facts, reasoning is complete
                break
            
            # Checkpoint if needed
            if (step_num + 1) % checkpoint_every == 0:
                self._checkpoint(trace)
                trace.add_step(StepType.CHECKPOINT, f"Checkpoint at step {step_num + 1}")
        
        # Draw conclusion
        conclusion = self._draw_conclusion(trace.get_facts())
        trace.conclusion = conclusion
        trace.add_step(StepType.CONCLUSION, conclusion)
        trace.completed_at = datetime.now()
        
        # Final checkpoint
        self._checkpoint(trace)
        
        return trace
    
    def _extract_facts(self, problem: str) -> List[str]:
        """
        Extract facts from problem description.
        
        This is a simple heuristic extractor. In a real system,
        this could use NLP techniques.
        
        Args:
            problem: Problem description
            
        Returns:
            List of facts
        """
        facts = []
        
        # Simple keyword-based extraction
        keywords = {
            "is": "property",
            "has": "property",
            "contains": "contains",
            "needs": "requires",
            "requires": "requires",
        }
        
        words = problem.lower().split()
        
        for i, word in enumerate(words):
            if word in keywords:
                # Create a fact from context
                context = " ".join(words[max(0, i-2):min(len(words), i+3)])
                facts.append(f"fact:{context}")
        
        # Always add the problem itself as a fact
        if not facts:
            facts.append(f"fact:{problem.lower()}")
        
        return facts
    
    def _draw_conclusion(self, facts: List[str]) -> str:
        """
        Draw a conclusion from accumulated facts.
        
        Args:
            facts: All accumulated facts
            
        Returns:
            Conclusion string
        """
        # Simple conclusion based on fact count and content
        if not facts:
            return "No conclusion could be drawn."
        
        # Check for specific patterns in facts
        if any("solved" in f.lower() for f in facts):
            return "Problem appears to be solved based on derived facts."
        elif any("impossible" in f.lower() or "contradiction" in f.lower() for f in facts):
            return "Problem may not have a solution due to contradictions."
        else:
            return f"Analysis complete. Derived {len(facts)} total facts."
    
    def _checkpoint(self, trace: ReasoningTrace) -> None:
        """
        Checkpoint reasoning trace to memory.
        
        Args:
            trace: Reasoning trace to checkpoint
        """
        if not self.memory_store:
            return
        
        checkpoint_data = trace.to_dict()
        checkpoint_key = f"reasoning_checkpoint_{trace.started_at.timestamp()}"
        
        self.memory_store.put(
            checkpoint_key,
            checkpoint_data,
            metadata={'type': 'reasoning_checkpoint'}
        )
    
    def _setup_default_rules(self) -> None:
        """Set up default inference rules."""
        
        # Rule: If problem requires X and we have X, mark as solved
        def has_requirement(facts: List[str]) -> bool:
            return any("requires" in f for f in facts)
        
        def check_requirement_met(facts: List[str]) -> bool:
            requirements = [f for f in facts if "requires" in f]
            has_facts = [f for f in facts if f.startswith("fact:") and "has" in f]
            # Simplified check
            return len(requirements) > 0 and len(has_facts) > 0
        
        def mark_solved(facts: List[str]) -> List[str]:
            return ["status:solved"]
        
        self.inference_engine.add_rule(Rule(
            name="requirement_met",
            conditions=[has_requirement, check_requirement_met],
            action=mark_solved,
            description="Mark problem as solved when requirements are met"
        ))
