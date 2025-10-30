"""
Unit tests for Phase 15 - Hierarchical Planner
"""

import unittest

from modules.planner import (
    Task, TaskGraph, TaskStatus,
    Planner, Scheduler, PlanningContext
)
from modules.memory import InMemoryStore


class TestTaskGraph(unittest.TestCase):
    """Tests for TaskGraph."""
    
    def setUp(self):
        """Create fresh task graph for each test."""
        self.graph = TaskGraph()
    
    def test_add_task(self):
        """Test adding tasks to graph."""
        task_id = self.graph.add_task("Test task")
        
        self.assertIsNotNone(task_id)
        self.assertIn(task_id, self.graph.tasks)
        
        task = self.graph.get_task(task_id)
        self.assertEqual(task.description, "Test task")
        self.assertEqual(task.status, TaskStatus.PENDING)
    
    def test_add_subtask(self):
        """Test adding subtasks with parent relationship."""
        parent_id = self.graph.add_task("Parent task")
        child_id = self.graph.add_task("Child task", parent_task=parent_id)
        
        parent = self.graph.get_task(parent_id)
        child = self.graph.get_task(child_id)
        
        self.assertIn(child_id, parent.subtasks)
        self.assertEqual(child.parent_task, parent_id)
    
    def test_update_status(self):
        """Test updating task status."""
        task_id = self.graph.add_task("Test task")
        
        result = self.graph.update_status(task_id, TaskStatus.IN_PROGRESS)
        self.assertTrue(result)
        
        task = self.graph.get_task(task_id)
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks with satisfied dependencies."""
        task1_id = self.graph.add_task("Task 1")
        task2_id = self.graph.add_task("Task 2", dependencies=[task1_id])
        task3_id = self.graph.add_task("Task 3")
        
        # Initially, only tasks without dependencies are ready
        ready = self.graph.get_ready_tasks()
        ready_ids = [t.task_id for t in ready]
        
        self.assertIn(task1_id, ready_ids)
        self.assertIn(task3_id, ready_ids)
        self.assertNotIn(task2_id, ready_ids)
        
        # After completing task1, task2 becomes ready
        self.graph.update_status(task1_id, TaskStatus.COMPLETED)
        ready = self.graph.get_ready_tasks()
        ready_ids = [t.task_id for t in ready]
        
        self.assertIn(task2_id, ready_ids)
    
    def test_get_subtasks(self):
        """Test getting subtasks of a task."""
        parent_id = self.graph.add_task("Parent")
        child1_id = self.graph.add_task("Child 1", parent_task=parent_id)
        child2_id = self.graph.add_task("Child 2", parent_task=parent_id)
        
        subtasks = self.graph.get_subtasks(parent_id)
        subtask_ids = [t.task_id for t in subtasks]
        
        self.assertEqual(len(subtasks), 2)
        self.assertIn(child1_id, subtask_ids)
        self.assertIn(child2_id, subtask_ids)
    
    def test_is_complete(self):
        """Test checking if graph is complete."""
        task1_id = self.graph.add_task("Task 1")
        task2_id = self.graph.add_task("Task 2")
        
        self.assertFalse(self.graph.is_complete())
        
        self.graph.update_status(task1_id, TaskStatus.COMPLETED)
        self.assertFalse(self.graph.is_complete())
        
        self.graph.update_status(task2_id, TaskStatus.COMPLETED)
        self.assertTrue(self.graph.is_complete())
    
    def test_get_statistics(self):
        """Test getting task statistics."""
        self.graph.add_task("Task 1")
        task2_id = self.graph.add_task("Task 2")
        self.graph.add_task("Task 3")
        
        self.graph.update_status(task2_id, TaskStatus.COMPLETED)
        
        stats = self.graph.get_statistics()
        
        self.assertEqual(stats['total'], 3)
        self.assertEqual(stats['pending'], 2)
        self.assertEqual(stats['completed'], 1)


class TestPlanner(unittest.TestCase):
    """Tests for Planner."""
    
    def setUp(self):
        """Create fresh planner for each test."""
        self.store = InMemoryStore()
        self.planner = Planner(self.store)
    
    def test_plan_basic(self):
        """Test basic planning."""
        graph = self.planner.plan("Search and analyze data")
        
        self.assertIsNotNone(graph)
        self.assertGreater(len(graph.tasks), 0)
    
    def test_plan_creates_subtasks(self):
        """Test that planning creates subtasks."""
        graph = self.planner.plan("Search and analyze information")
        
        # Should have root task plus subtasks
        self.assertGreater(len(graph.tasks), 1)
        
        # Root task should have subtasks
        root_task = graph.tasks["task_1"]
        self.assertGreater(len(root_task.subtasks), 0)
    
    def test_plan_with_context(self):
        """Test planning with context."""
        context = PlanningContext(
            goal="Test goal",
            constraints={"time_limit": 60},
            available_capabilities={"search", "analysis"},
            metadata={"priority": "high"}
        )
        
        graph = self.planner.plan("Search and analyze", context=context)
        self.assertIsNotNone(graph)
    
    def test_refine_task(self):
        """Test refining a task into subtasks."""
        task = Task(
            task_id="test_task",
            description="Search for information"
        )
        
        subtasks = self.planner.refine(task)
        
        self.assertGreater(len(subtasks), 0)
        for subtask in subtasks:
            self.assertEqual(subtask.parent_task, task.task_id)
    
    def test_infer_capabilities(self):
        """Test capability inference from task description."""
        capabilities = self.planner._infer_capabilities("Search and analyze data")
        
        self.assertIn("search", capabilities)
        self.assertIn("analysis", capabilities)
    
    def test_memory_integration(self):
        """Test that planning events are logged to memory."""
        self.planner.plan("Test goal")
        
        # Check episodic memory
        events = self.planner.episodic.get_recent_episodes(event_type="planning")
        
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0].event_type, "planning")


class TestScheduler(unittest.TestCase):
    """Tests for Scheduler."""
    
    def setUp(self):
        """Create fresh scheduler for each test."""
        self.scheduler = Scheduler()
    
    def test_register_agent(self):
        """Test registering agents."""
        self.scheduler.register_agent("agent1", {"search", "analysis"})
        
        self.assertIn("agent1", self.scheduler.agent_capabilities)
        self.assertEqual(self.scheduler.agent_capabilities["agent1"], {"search", "analysis"})
    
    def test_assign_task(self):
        """Test assigning a task to an agent."""
        self.scheduler.register_agent("agent1", {"search", "analysis"})
        
        task = Task(
            task_id="test_task",
            description="Search for data",
            required_capabilities={"search"}
        )
        
        agent_id = self.scheduler.assign_task(task)
        
        self.assertEqual(agent_id, "agent1")
        self.assertEqual(task.assigned_agent, "agent1")
    
    def test_assign_task_no_match(self):
        """Test assigning a task when no agent matches."""
        self.scheduler.register_agent("agent1", {"search"})
        
        task = Task(
            task_id="test_task",
            description="Analyze data",
            required_capabilities={"analysis", "machine_learning"}
        )
        
        agent_id = self.scheduler.assign_task(task)
        
        self.assertIsNone(agent_id)
    
    def test_assign_tasks_multiple(self):
        """Test assigning multiple tasks."""
        self.scheduler.register_agent("agent1", {"search"})
        self.scheduler.register_agent("agent2", {"analysis"})
        
        graph = TaskGraph()
        task1_id = graph.add_task("Search", required_capabilities={"search"})
        task2_id = graph.add_task("Analyze", required_capabilities={"analysis"})
        
        assignments = self.scheduler.assign_tasks(graph)
        
        self.assertEqual(len(assignments), 2)
        self.assertEqual(assignments[task1_id], "agent1")
        self.assertEqual(assignments[task2_id], "agent2")


if __name__ == '__main__':
    unittest.main()
