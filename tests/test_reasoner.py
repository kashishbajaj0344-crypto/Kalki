"""
Unit tests for Phase 16 - Iterative Reasoning
"""

import unittest

from modules.reasoner import (
    Reasoner, ReasoningTrace, Step, StepType,
    InferenceEngine, Rule
)
from modules.memory import InMemoryStore


class TestReasoningTrace(unittest.TestCase):
    """Tests for ReasoningTrace."""
    
    def test_create_trace(self):
        """Test creating a reasoning trace."""
        trace = ReasoningTrace(problem="Test problem")
        
        self.assertEqual(trace.problem, "Test problem")
        self.assertEqual(len(trace.steps), 0)
        self.assertIsNone(trace.conclusion)
    
    def test_add_step(self):
        """Test adding steps to trace."""
        trace = ReasoningTrace(problem="Test")
        
        step = trace.add_step(
            StepType.OBSERVATION,
            "Initial observation",
            facts=["fact1", "fact2"]
        )
        
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(step.step_id, 0)
        self.assertEqual(step.step_type, StepType.OBSERVATION)
        self.assertEqual(step.content, "Initial observation")
    
    def test_get_facts(self):
        """Test getting all facts from trace."""
        trace = ReasoningTrace(problem="Test")
        
        trace.add_step(StepType.OBSERVATION, "Step 1", facts=["fact1", "fact2"])
        trace.add_step(StepType.INFERENCE, "Step 2", facts=["fact3"])
        
        all_facts = trace.get_facts()
        
        self.assertEqual(len(all_facts), 3)
        self.assertIn("fact1", all_facts)
        self.assertIn("fact2", all_facts)
        self.assertIn("fact3", all_facts)
    
    def test_to_dict(self):
        """Test converting trace to dictionary."""
        trace = ReasoningTrace(problem="Test")
        trace.add_step(StepType.OBSERVATION, "Step 1")
        trace.conclusion = "Test conclusion"
        
        trace_dict = trace.to_dict()
        
        self.assertEqual(trace_dict['problem'], "Test")
        self.assertEqual(trace_dict['conclusion'], "Test conclusion")
        self.assertEqual(len(trace_dict['steps']), 1)


class TestRule(unittest.TestCase):
    """Tests for Rule."""
    
    def test_rule_matches(self):
        """Test rule matching."""
        def has_python(facts):
            return any("python" in f.lower() for f in facts)
        
        def has_programming(facts):
            return any("programming" in f.lower() for f in facts)
        
        def derive_language(facts):
            return ["language:python"]
        
        rule = Rule(
            name="test_rule",
            conditions=[has_python, has_programming],
            action=derive_language
        )
        
        # Should match
        self.assertTrue(rule.matches(["Python is a programming language"]))
        
        # Should not match (missing "programming")
        self.assertFalse(rule.matches(["Python is great"]))
    
    def test_rule_apply(self):
        """Test applying a rule."""
        def always_true(facts):
            return True
        
        def add_fact(facts):
            return ["new_fact"]
        
        rule = Rule(
            name="test_rule",
            conditions=[always_true],
            action=add_fact
        )
        
        new_facts = rule.apply(["existing_fact"])
        
        self.assertIn("new_fact", new_facts)


class TestInferenceEngine(unittest.TestCase):
    """Tests for InferenceEngine."""
    
    def test_add_rule(self):
        """Test adding rules to engine."""
        engine = InferenceEngine()
        
        rule = Rule(
            name="test",
            conditions=[lambda f: True],
            action=lambda f: []
        )
        
        engine.add_rule(rule)
        
        self.assertEqual(len(engine.rules), 1)
    
    def test_infer_basic(self):
        """Test basic inference."""
        engine = InferenceEngine()
        
        # Rule: if "A" in facts, add "B"
        def has_a(facts):
            return "A" in facts
        
        def add_b(facts):
            return ["B"] if has_a(facts) else []
        
        rule = Rule(
            name="A_implies_B",
            conditions=[has_a],
            action=add_b
        )
        
        engine.add_rule(rule)
        
        result = engine.infer(["A"])
        
        self.assertIn("A", result)
        self.assertIn("B", result)
    
    def test_infer_chain(self):
        """Test chaining multiple rules."""
        engine = InferenceEngine()
        
        # Rule 1: A -> B
        engine.add_rule(Rule(
            name="A_to_B",
            conditions=[lambda f: "A" in f],
            action=lambda f: ["B"]
        ))
        
        # Rule 2: B -> C
        engine.add_rule(Rule(
            name="B_to_C",
            conditions=[lambda f: "B" in f],
            action=lambda f: ["C"]
        ))
        
        result = engine.infer(["A"], max_iterations=5)
        
        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("C", result)


class TestReasoner(unittest.TestCase):
    """Tests for Reasoner."""
    
    def setUp(self):
        """Create fresh reasoner for each test."""
        self.store = InMemoryStore()
        self.reasoner = Reasoner(self.store)
    
    def test_reason_basic(self):
        """Test basic reasoning."""
        trace = self.reasoner.reason("Solve a simple problem")
        
        self.assertIsNotNone(trace)
        self.assertEqual(trace.problem, "Solve a simple problem")
        self.assertGreater(len(trace.steps), 0)
        self.assertIsNotNone(trace.conclusion)
    
    def test_reason_with_steps(self):
        """Test reasoning produces multiple steps."""
        trace = self.reasoner.reason("This problem requires analysis", max_steps=5)
        
        # Should have at least observation and conclusion steps
        self.assertGreaterEqual(len(trace.steps), 2)
        
        # Check step types
        step_types = [s.step_type for s in trace.steps]
        self.assertIn(StepType.OBSERVATION, step_types)
        self.assertIn(StepType.CONCLUSION, step_types)
    
    def test_checkpointing(self):
        """Test that reasoning is checkpointed to memory."""
        trace = self.reasoner.reason("Test problem", max_steps=10, checkpoint_every=2)
        
        # Check that trace was completed (which triggers final checkpoint)
        self.assertIsNotNone(trace.completed_at)
        
        # Check that at least one checkpoint was saved to memory
        # (final checkpoint is always saved)
        from modules.memory import MemoryQuery
        query = MemoryQuery(filter={'type': 'reasoning_checkpoint'})
        checkpoints = self.store.query(query)
        self.assertGreater(len(checkpoints), 0)
    
    def test_extract_facts(self):
        """Test fact extraction from problem."""
        facts = self.reasoner._extract_facts("The system is broken and needs repair")
        
        self.assertGreater(len(facts), 0)
        self.assertTrue(any("is" in f.lower() or "needs" in f.lower() for f in facts))
    
    def test_draw_conclusion(self):
        """Test drawing conclusions from facts."""
        facts = ["fact1", "fact2", "status:solved"]
        conclusion = self.reasoner._draw_conclusion(facts)
        
        self.assertIn("solved", conclusion.lower())


if __name__ == '__main__':
    unittest.main()
