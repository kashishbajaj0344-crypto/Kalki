"""
Professional Workflow System
Multi-step professional workflows with dependencies and parallel execution.

Enables complex professional processes:
- Construction: design → validate → schedule → estimate
- Game Dev: design → prototype → implement → test
- Robotics: design → simulate → build → test
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from modules.professional_team_orchestrator import ProfessionalRole, ProfessionalTeamOrchestrator
from modules.llm import LLMEngine

logger = logging.getLogger(__name__)


class WorkflowStepStatus(Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """A step in a professional workflow"""
    name: str
    role: ProfessionalRole
    action: str
    inputs: List[str]  # Required input step names
    outputs: List[str]  # Output names this step produces
    validation: Optional[Callable] = None
    timeout_seconds: int = 300
    retry_count: int = 0
    status: WorkflowStepStatus = WorkflowStepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class ProfessionalWorkflow:
    """Represents a professional workflow"""
    name: str
    description: str
    steps: List[WorkflowStep]
    domain: str
    created_at: datetime = field(default_factory=datetime.now)
    workflow_id: str = field(default_factory=lambda: f"workflow_{datetime.now().timestamp()}")


class ProfessionalWorkflowExecutor:
    """
    Executes professional workflows with dependency management and parallel execution.
    
    Supports:
    - Multi-step workflows with dependencies
    - Parallel execution of independent steps
    - Step validation
    - Error handling and retries
    - Workflow state tracking
    """
    
    def __init__(
        self,
        team_orchestrator: ProfessionalTeamOrchestrator,
        llm_engine: LLMEngine
    ):
        self.team_orchestrator = team_orchestrator
        self.llm_engine = llm_engine
        self.workflow_history: List[Dict[str, Any]] = []
    
    async def execute_workflow(
        self,
        workflow: ProfessionalWorkflow,
        context: Dict[str, Any],
        initial_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a professional workflow.
        
        Example construction workflow:
        1. Architect: Design layout
        2. Engineer: Validate structure
        3. PM: Create schedule
        4. Cost Estimator: Provide budget
        
        Args:
            workflow: Workflow definition
            context: Context information
            initial_data: Initial data for workflow
        
        Returns:
            Complete workflow results
        """
        logger.info(f"🔄 Executing workflow: {workflow.name}")
        logger.info(f"   Steps: {len(workflow.steps)}")
        
        # Initialize step states
        step_states: Dict[str, WorkflowStep] = {
            step.name: step for step in workflow.steps
        }
        
        # Track outputs from completed steps
        workflow_outputs: Dict[str, Any] = initial_data or {}
        
        # Execute steps respecting dependencies
        completed_steps = set()
        failed_steps = set()
        max_iterations = len(workflow.steps) * 2  # Safety limit
        iteration = 0
        
        while len(completed_steps) < len(workflow.steps) and iteration < max_iterations:
            iteration += 1
            
            # Find steps ready to execute (dependencies met)
            ready_steps = [
                step for step in workflow.steps
                if step.name not in completed_steps
                and step.name not in failed_steps
                and step.status == WorkflowStepStatus.PENDING
                and all(
                    dep in completed_steps
                    for dep in step.inputs
                )
            ]
            
            if not ready_steps:
                # Check if we're stuck
                remaining = [
                    s for s in workflow.steps
                    if s.name not in completed_steps and s.name not in failed_steps
                ]
                if remaining:
                    logger.warning(f"⚠️ Workflow deadlock: {len(remaining)} steps cannot execute")
                    # Mark remaining as failed
                    for step in remaining:
                        step.status = WorkflowStepStatus.FAILED
                        step.error = "Dependencies not met or workflow deadlock"
                        failed_steps.add(step.name)
                break
            
            # Execute ready steps in parallel
            step_tasks = []
            for step in ready_steps:
                step.status = WorkflowStepStatus.READY
                task = self._execute_workflow_step(
                    step=step,
                    context={**context, **workflow_outputs},
                    workflow_outputs=workflow_outputs
                )
                step_tasks.append((step.name, task))
            
            # Wait for all ready steps
            if step_tasks:
                step_results = await asyncio.gather(
                    *[task for _, task in step_tasks],
                    return_exceptions=True
                )
                
                # Process results
                for (step_name, _), result in zip(step_tasks, step_results):
                    step = step_states[step_name]
                    
                    if isinstance(result, Exception):
                        step.status = WorkflowStepStatus.FAILED
                        step.error = str(result)
                        failed_steps.add(step_name)
                        logger.error(f"❌ Step {step_name} failed: {result}")
                    else:
                        step.status = WorkflowStepStatus.COMPLETED
                        step.result = result
                        step.completed_at = datetime.now()
                        completed_steps.add(step_name)
                        
                        # Store outputs
                        for output_name in step.outputs:
                            workflow_outputs[output_name] = result.get(output_name, result)
                        
                        logger.info(f"✅ Step {step_name} completed")
        
        # Generate workflow summary using Llama 3.1 8B
        summary = await self._generate_workflow_summary(
            workflow=workflow,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            outputs=workflow_outputs
        )
        
        # Record workflow execution
        workflow_record = {
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "domain": workflow.domain,
            "steps_total": len(workflow.steps),
            "steps_completed": len(completed_steps),
            "steps_failed": len(failed_steps),
            "outputs": workflow_outputs,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        self.workflow_history.append(workflow_record)
        
        return {
            "workflow_id": workflow.workflow_id,
            "workflow_name": workflow.name,
            "status": "completed" if len(failed_steps) == 0 else "partial",
            "steps_completed": len(completed_steps),
            "steps_failed": len(failed_steps),
            "outputs": workflow_outputs,
            "summary": summary,
            "step_results": {
                step.name: {
                    "status": step.status.value,
                    "result": step.result,
                    "error": step.error
                }
                for step in workflow.steps
            }
        }
    
    async def _execute_workflow_step(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        workflow_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step"""
        step.status = WorkflowStepStatus.RUNNING
        step.started_at = datetime.now()
        
        try:
            # Use team orchestrator to execute role work
            result = await self.team_orchestrator.coordinate_team_task(
                task=step.action,
                required_roles=[step.role],
                context=context,
                domain=context.get('domain', 'general')
            )
            
            # Extract outputs
            step_outputs = {}
            if step.outputs:
                role_result = result.get('role_results', {}).get(step.role.value, {})
                for output_name in step.outputs:
                    step_outputs[output_name] = role_result.get(output_name, role_result)
            
            # Validate if validation function provided
            if step.validation:
                validation_result = await step.validation(step_outputs, context)
                if not validation_result.get('valid', True):
                    raise ValueError(f"Validation failed: {validation_result.get('error', 'Unknown error')}")
            
            return {
                **step_outputs,
                "step_name": step.name,
                "role": step.role.value,
                "result": result
            }
            
        except Exception as e:
            step.status = WorkflowStepStatus.FAILED
            step.error = str(e)
            raise
    
    async def _generate_workflow_summary(
        self,
        workflow: ProfessionalWorkflow,
        completed_steps: set,
        failed_steps: set,
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate workflow summary using Llama 3.1 8B"""
        summary_prompt = f"""Generate a professional summary for a {workflow.domain} workflow execution.

Workflow: {workflow.name}
Description: {workflow.description}

Steps Completed: {len(completed_steps)}/{len(workflow.steps)}
Steps Failed: {len(failed_steps)}

Outputs Generated:
{outputs}

Provide:
1. Executive summary of workflow execution
2. Key achievements
3. Any issues or concerns
4. Recommendations for next steps

Format as professional workflow report."""

        summary_response = await self.llm_engine.generate(
            prompt=summary_prompt,
            max_tokens=500,
            temperature=0.5
        )
        
        if isinstance(summary_response, dict):
            summary_text = summary_response.get("text", str(summary_response))
        else:
            summary_text = str(summary_response)
        
        return {
            "summary": summary_text,
            "steps_completed": len(completed_steps),
            "steps_failed": len(failed_steps),
            "success_rate": len(completed_steps) / len(workflow.steps) if workflow.steps else 0
        }
    
    async def generate_workflow_from_requirements(
        self,
        requirements: str,
        domain: str,
        context: Dict[str, Any]
    ) -> ProfessionalWorkflow:
        """
        Generate a professional workflow from requirements using Llama 3.1 8B.
        
        Args:
            requirements: Natural language requirements
            domain: Domain context
            context: Additional context
        
        Returns:
            Generated workflow
        """
        workflow_prompt = f"""Generate a professional workflow for a {domain} project based on these requirements:

Requirements:
{requirements}

Context:
{context}

Generate a multi-step workflow with:
1. Step names
2. Professional roles for each step
3. Actions to perform
4. Dependencies between steps
5. Expected outputs

Return as structured workflow definition."""

        workflow_response = await self.llm_engine.generate(
            prompt=workflow_prompt,
            max_tokens=1000,
            temperature=0.7
        )
        
        if isinstance(workflow_response, dict):
            workflow_text = workflow_response.get("text", str(workflow_response))
        else:
            workflow_text = str(workflow_response)
        
        # Parse workflow from response (simplified - would need more robust parsing)
        # For now, create a default workflow structure
        steps = self._parse_workflow_steps(workflow_text, domain)
        
        return ProfessionalWorkflow(
            name=f"{domain}_workflow",
            description=requirements,
            steps=steps,
            domain=domain
        )
    
    def _parse_workflow_steps(
        self,
        workflow_text: str,
        domain: str
    ) -> List[WorkflowStep]:
        """Parse workflow steps from LLM response"""
        # This is a simplified parser - would need more robust implementation
        steps = []
        
        # Default workflows by domain
        if domain == "construction":
            steps = [
                WorkflowStep(
                    name="design",
                    role=ProfessionalRole.ARCHITECT,
                    action="Design building layout and specifications",
                    inputs=[],
                    outputs=["design_spec"]
                ),
                WorkflowStep(
                    name="validate",
                    role=ProfessionalRole.STRUCTURAL_ENGINEER,
                    action="Validate structural integrity",
                    inputs=["design"],
                    outputs=["validation_report"]
                ),
                WorkflowStep(
                    name="schedule",
                    role=ProfessionalRole.PROJECT_MANAGER,
                    action="Create project schedule",
                    inputs=["design", "validate"],
                    outputs=["schedule"]
                ),
                WorkflowStep(
                    name="estimate",
                    role=ProfessionalRole.COST_ESTIMATOR,
                    action="Provide cost estimate",
                    inputs=["design", "schedule"],
                    outputs=["cost_estimate"]
                )
            ]
        elif domain == "game_dev":
            steps = [
                WorkflowStep(
                    name="design",
                    role=ProfessionalRole.GAME_DESIGNER,
                    action="Design game mechanics and systems",
                    inputs=[],
                    outputs=["game_design"]
                ),
                WorkflowStep(
                    name="prototype",
                    role=ProfessionalRole.PROGRAMMER,
                    action="Create prototype",
                    inputs=["design"],
                    outputs=["prototype"]
                ),
                WorkflowStep(
                    name="implement",
                    role=ProfessionalRole.PROGRAMMER,
                    action="Implement full game",
                    inputs=["prototype"],
                    outputs=["game_code"]
                ),
                WorkflowStep(
                    name="test",
                    role=ProfessionalRole.QA_TESTER,
                    action="Test game functionality",
                    inputs=["implement"],
                    outputs=["test_report"]
                )
            ]
        # Add more domain workflows as needed
        
        return steps
    
    def get_workflow_history(self) -> List[Dict[str, Any]]:
        """Get workflow execution history"""
        return self.workflow_history


