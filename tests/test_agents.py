"""
Unit and integration tests for Phase 17 - Multi-Agent Coordination
"""

import unittest
import time
from threading import Event

from modules.agents import (
    Agent, AgentContext, AgentResult, AgentStatus,
    Message, MessageBus,
    AgentRegistry, AgentRunner,
    SearchAgent, ExecutorAgent, SafetyAgent, ReasoningAgent
)
from modules.planner import Task, TaskStatus
from modules.memory import InMemoryStore, EpisodicMemory


class TestMessageBus(unittest.TestCase):
    """Tests for MessageBus."""
    
    def setUp(self):
        """Create fresh message bus for each test."""
        self.store = InMemoryStore()
        self.episodic = EpisodicMemory(self.store)
        self.bus = MessageBus(self.episodic)
    
    def test_publish_message(self):
        """Test publishing a message."""
        msg_id = self.bus.publish("agent1", "Hello", recipient="agent2")
        
        self.assertIsNotNone(msg_id)
        self.assertTrue(msg_id.startswith("msg_"))
    
    def test_subscribe_and_receive(self):
        """Test subscribing and receiving messages."""
        received = []
        
        def callback(msg: Message):
            received.append(msg)
        
        self.bus.subscribe("agent2", callback)
        self.bus.publish("agent1", "Test message", recipient="agent2")
        
        # Give time for delivery
        time.sleep(0.01)
        
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].content, "Test message")
    
    def test_broadcast_message(self):
        """Test broadcasting to all subscribers."""
        received_a = []
        received_b = []
        
        def callback_a(msg):
            received_a.append(msg)
        
        def callback_b(msg):
            received_b.append(msg)
        
        self.bus.subscribe("agent_a", callback_a)
        self.bus.subscribe("agent_b", callback_b)
        
        # Broadcast from agent_a (should not receive own message)
        self.bus.publish("agent_a", "Broadcast", recipient=None)
        
        time.sleep(0.01)
        
        self.assertEqual(len(received_a), 0)  # Sender doesn't receive
        self.assertEqual(len(received_b), 1)  # Other subscriber receives
    
    def test_message_stored_in_memory(self):
        """Test that messages are stored in episodic memory."""
        self.bus.publish("agent1", "Test", recipient="agent2")
        
        events = self.episodic.get_recent_episodes(event_type="agent_message")
        
        self.assertGreater(len(events), 0)
        self.assertEqual(events[0].event_type, "agent_message")


class TestAgentRegistry(unittest.TestCase):
    """Tests for AgentRegistry."""
    
    def setUp(self):
        """Create fresh registry for each test."""
        self.registry = AgentRegistry()
    
    def test_register_agent(self):
        """Test registering an agent."""
        agent = SearchAgent("search1")
        self.registry.register(agent)
        
        self.assertIn("search1", self.registry.agents)
    
    def test_unregister_agent(self):
        """Test unregistering an agent."""
        agent = SearchAgent("search1")
        self.registry.register(agent)
        
        result = self.registry.unregister("search1")
        
        self.assertTrue(result)
        self.assertNotIn("search1", self.registry.agents)
    
    def test_get_agent(self):
        """Test getting an agent by ID."""
        agent = ExecutorAgent("exec1")
        self.registry.register(agent)
        
        retrieved = self.registry.get_agent("exec1")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.agent_id, "exec1")
    
    def test_find_capable_agents(self):
        """Test finding agents with required capabilities."""
        self.registry.register(SearchAgent("search1"))
        self.registry.register(ExecutorAgent("exec1"))
        self.registry.register(SafetyAgent("safety1"))
        
        # Find agents with search capability
        capable = self.registry.find_capable_agents({"search"})
        
        self.assertEqual(len(capable), 1)
        self.assertEqual(capable[0].agent_id, "search1")
    
    def test_get_available_agents(self):
        """Test getting idle agents."""
        agent1 = SearchAgent("search1")
        agent2 = ExecutorAgent("exec1")
        agent2.status = AgentStatus.BUSY
        
        self.registry.register(agent1)
        self.registry.register(agent2)
        
        available = self.registry.get_available_agents()
        
        self.assertEqual(len(available), 1)
        self.assertEqual(available[0].agent_id, "search1")


class TestAgentRunner(unittest.TestCase):
    """Tests for AgentRunner."""
    
    def setUp(self):
        """Create fresh runner for each test."""
        self.registry = AgentRegistry()
        self.store = InMemoryStore()
        self.episodic = EpisodicMemory(self.store)
        self.bus = MessageBus(self.episodic)
        self.runner = AgentRunner(self.registry, self.bus)
        
        # Register sample agents
        self.registry.register(SearchAgent("search1"))
        self.registry.register(ExecutorAgent("exec1"))
        self.registry.register(SafetyAgent("safety1"))
    
    def test_execute_task_sync(self):
        """Test synchronous task execution."""
        task = Task(
            task_id="task1",
            description="Search for information",
            required_capabilities={"search"}
        )
        
        context = AgentContext(
            task_id=task.task_id,
            task_description=task.description,
            task_data={"query": "test query"}
        )
        
        result = self.runner.execute_task(task, context)
        
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertIn("results", result.result)
    
    def test_execute_task_no_capable_agent(self):
        """Test execution when no capable agent exists."""
        task = Task(
            task_id="task1",
            description="Impossible task",
            required_capabilities={"nonexistent_capability"}
        )
        
        result = self.runner.execute_task(task)
        
        self.assertIsNone(result)
    
    def test_execute_task_async(self):
        """Test asynchronous task execution."""
        task = Task(
            task_id="task1",
            description="Execute analysis",
            required_capabilities={"execution"}
        )
        
        result_received = Event()
        received_result = []
        
        def callback(result):
            received_result.append(result)
            result_received.set()
        
        started = self.runner.execute_task_async(task, callback=callback)
        
        self.assertTrue(started)
        
        # Wait for completion (with timeout)
        result_received.wait(timeout=1.0)
        
        self.assertEqual(len(received_result), 1)
        self.assertTrue(received_result[0].success)


class TestSampleAgents(unittest.TestCase):
    """Tests for sample agent implementations."""
    
    def test_search_agent(self):
        """Test SearchAgent execution."""
        agent = SearchAgent()
        
        task = Task(task_id="t1", description="Search for Python")
        context = AgentContext(
            task_id=task.task_id,
            task_description=task.description,
            task_data={"query": "Python programming"}
        )
        
        result = agent.execute(task, context)
        
        self.assertTrue(result.success)
        self.assertIn("results", result.result)
        self.assertGreater(len(result.result["results"]), 0)
    
    def test_executor_agent(self):
        """Test ExecutorAgent execution."""
        agent = ExecutorAgent()
        
        task = Task(task_id="t1", description="Analyze data")
        context = AgentContext(
            task_id=task.task_id,
            task_description=task.description
        )
        
        result = agent.execute(task, context)
        
        self.assertTrue(result.success)
        self.assertIn("analysis", result.result)
    
    def test_safety_agent_pass(self):
        """Test SafetyAgent with safe task."""
        agent = SafetyAgent()
        
        task = Task(task_id="t1", description="Process normal data")
        context = AgentContext(
            task_id=task.task_id,
            task_description=task.description
        )
        
        result = agent.execute(task, context)
        
        self.assertTrue(result.success)
        self.assertTrue(result.result["safe"])
    
    def test_safety_agent_fail(self):
        """Test SafetyAgent with unsafe task."""
        agent = SafetyAgent()
        
        task = Task(task_id="t1", description="Execute malicious code")
        context = AgentContext(
            task_id=task.task_id,
            task_description=task.description
        )
        
        result = agent.execute(task, context)
        
        self.assertFalse(result.success)
        self.assertFalse(result.result["safe"])
        self.assertIn("violations", result.result)
    
    def test_reasoning_agent(self):
        """Test ReasoningAgent execution."""
        agent = ReasoningAgent()
        
        task = Task(task_id="t1", description="Reason about problem")
        context = AgentContext(
            task_id=task.task_id,
            task_description=task.description,
            task_data={"facts": ["fact1", "fact2"]}
        )
        
        result = agent.execute(task, context)
        
        self.assertTrue(result.success)
        self.assertIn("reasoning_steps", result.result)
        self.assertIn("conclusion", result.result)


class TestMultiAgentIntegration(unittest.TestCase):
    """Integration tests for multi-agent coordination."""
    
    def setUp(self):
        """Set up multi-agent system."""
        self.registry = AgentRegistry()
        self.store = InMemoryStore()
        self.episodic = EpisodicMemory(self.store)
        self.bus = MessageBus(self.episodic)
        self.runner = AgentRunner(self.registry, self.bus)
        
        # Register multiple agents
        self.registry.register(SearchAgent("search1"))
        self.registry.register(ExecutorAgent("exec1"))
        self.registry.register(SafetyAgent("safety1"))
        self.registry.register(ReasoningAgent("reason1"))
    
    def test_multi_agent_workflow(self):
        """Test coordinated multi-agent workflow."""
        # Task 1: Search
        task1 = Task(
            task_id="t1",
            description="Search for data",
            required_capabilities={"search"}
        )
        
        result1 = self.runner.execute_task(task1)
        self.assertTrue(result1.success)
        
        # Task 2: Execute with search results
        task2 = Task(
            task_id="t2",
            description="Process search results",
            required_capabilities={"execution"}
        )
        
        context2 = AgentContext(
            task_id=task2.task_id,
            task_description=task2.description,
            task_data={"input": result1.result}
        )
        
        result2 = self.runner.execute_task(task2, context2)
        self.assertTrue(result2.success)
        
        # Task 3: Validate safety
        task3 = Task(
            task_id="t3",
            description="Validate results",
            required_capabilities={"validation"}
        )
        
        result3 = self.runner.execute_task(task3)
        self.assertTrue(result3.success)
    
    def test_message_based_coordination(self):
        """Test agents coordinating via message bus."""
        messages_received = []
        
        def agent_callback(msg):
            messages_received.append(msg)
        
        # Subscribe agents to messages
        self.bus.subscribe("exec1", agent_callback)
        self.bus.subscribe("safety1", agent_callback)
        
        # Search agent sends broadcast
        self.bus.publish(
            "search1",
            {"results_ready": True, "count": 10},
            message_type="search_complete"
        )
        
        time.sleep(0.01)
        
        # Both agents should receive the message
        self.assertGreaterEqual(len(messages_received), 2)


if __name__ == '__main__':
    unittest.main()
