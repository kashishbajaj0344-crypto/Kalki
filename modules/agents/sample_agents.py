"""
Sample agent implementations for demonstration.
"""

import time
from typing import Set
from .base import Agent, AgentContext, AgentResult


class SearchAgent(Agent):
    """Agent that simulates search operations."""
    
    def __init__(self, agent_id: str = "search_agent"):
        """Initialize search agent."""
        super().__init__(agent_id, capabilities={"search", "query", "data_collection"})
    
    def execute(self, task, context: AgentContext) -> AgentResult:
        """
        Execute a search task.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            AgentResult with search results
        """
        try:
            # Simulate search operation
            query = context.task_data.get("query", context.task_description)
            
            # Mock search results
            results = {
                "query": query,
                "results": [
                    {"title": f"Result 1 for {query}", "score": 0.95},
                    {"title": f"Result 2 for {query}", "score": 0.87},
                    {"title": f"Result 3 for {query}", "score": 0.72}
                ],
                "total_found": 3
            }
            
            # Simulate processing time
            time.sleep(0.01)
            
            return AgentResult(
                success=True,
                result=results,
                metadata={"agent_type": "search"}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))


class ExecutorAgent(Agent):
    """Agent that executes generic tasks."""
    
    def __init__(self, agent_id: str = "executor_agent"):
        """Initialize executor agent."""
        super().__init__(
            agent_id,
            capabilities={"execution", "data_processing", "analysis"}
        )
    
    def execute(self, task, context: AgentContext) -> AgentResult:
        """
        Execute a generic task.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            AgentResult with execution outcome
        """
        try:
            # Process the task based on description
            description = context.task_description.lower()
            
            result_data = {
                "task_id": context.task_id,
                "description": context.task_description,
                "status": "completed"
            }
            
            # Add task-specific processing
            if "analyze" in description:
                result_data["analysis"] = {
                    "key_findings": ["Finding 1", "Finding 2"],
                    "confidence": 0.85
                }
            elif "process" in description:
                result_data["processed_items"] = 42
            
            # Simulate processing time
            time.sleep(0.01)
            
            return AgentResult(
                success=True,
                result=result_data,
                metadata={"agent_type": "executor"}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))


class SafetyAgent(Agent):
    """Agent that validates safety constraints."""
    
    def __init__(self, agent_id: str = "safety_agent"):
        """Initialize safety agent."""
        super().__init__(
            agent_id,
            capabilities={"validation", "safety", "testing"}
        )
        self.forbidden_keywords = {"malicious", "harmful", "dangerous"}
    
    def execute(self, task, context: AgentContext) -> AgentResult:
        """
        Validate task safety.
        
        Args:
            task: Task to validate
            context: Execution context
            
        Returns:
            AgentResult with safety validation
        """
        try:
            description = context.task_description.lower()
            data = str(context.task_data).lower()
            
            # Check for forbidden keywords
            violations = []
            for keyword in self.forbidden_keywords:
                if keyword in description or keyword in data:
                    violations.append(keyword)
            
            if violations:
                return AgentResult(
                    success=False,
                    result={
                        "safe": False,
                        "violations": violations,
                        "message": f"Safety violations detected: {', '.join(violations)}"
                    },
                    metadata={"agent_type": "safety"}
                )
            
            # Passed safety check
            return AgentResult(
                success=True,
                result={
                    "safe": True,
                    "message": "Task passed safety validation"
                },
                metadata={"agent_type": "safety"}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))


class ReasoningAgent(Agent):
    """Agent that performs reasoning tasks."""
    
    def __init__(self, agent_id: str = "reasoning_agent"):
        """Initialize reasoning agent."""
        super().__init__(
            agent_id,
            capabilities={"reasoning", "analysis", "synthesis"}
        )
    
    def execute(self, task, context: AgentContext) -> AgentResult:
        """
        Execute a reasoning task.
        
        Args:
            task: Task to execute
            context: Execution context
            
        Returns:
            AgentResult with reasoning outcome
        """
        try:
            # Simple reasoning simulation
            facts = context.task_data.get("facts", [])
            problem = context.task_description
            
            # Mock reasoning process
            reasoning_steps = [
                {"step": 1, "action": "Analyze problem statement"},
                {"step": 2, "action": f"Consider {len(facts)} known facts"},
                {"step": 3, "action": "Draw conclusions"}
            ]
            
            conclusion = f"Based on analysis of '{problem}', concluded that task is feasible."
            
            return AgentResult(
                success=True,
                result={
                    "reasoning_steps": reasoning_steps,
                    "conclusion": conclusion,
                    "confidence": 0.8
                },
                metadata={"agent_type": "reasoning"}
            )
            
        except Exception as e:
            return AgentResult(success=False, error=str(e))
