"""
Professional Team Orchestrator
Coordinates multiple professional roles working together on domain tasks.

Enables domains to simulate complete professional teams:
- Construction: Architect + Engineer + PM + Inspector
- Game Dev: Designer + Programmer + Artist + Sound Engineer
- Robotics: Mechanical + Control + Simulation Engineers
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from modules.agents.agent_manager import AgentManager
from modules.agents.base_agent import BaseAgent, AgentCapability
from modules.llm import LLMEngine

logger = logging.getLogger(__name__)


class ProfessionalRole(Enum):
    """Professional roles that domains can use"""
    # Construction roles
    ARCHITECT = "architect"
    STRUCTURAL_ENGINEER = "structural_engineer"
    MEP_ENGINEER = "mep_engineer"
    PROJECT_MANAGER = "project_manager"
    COST_ESTIMATOR = "cost_estimator"
    SCHEDULER = "scheduler"
    INSPECTOR = "inspector"
    SAFETY_OFFICER = "safety_officer"
    
    # Game Dev roles
    GAME_DESIGNER = "game_designer"
    PROGRAMMER = "programmer"
    ARTIST = "artist"
    SOUND_ENGINEER = "sound_engineer"
    QA_TESTER = "qa_tester"
    
    # Engineering roles (Robotics, Aerospace, Power Systems)
    MECHANICAL_ENGINEER = "mechanical_engineer"
    CONTROL_ENGINEER = "control_engineer"
    SYSTEMS_ENGINEER = "systems_engineer"
    TEST_ENGINEER = "test_engineer"
    THERMAL_ENGINEER = "thermal_engineer"
    ELECTRICAL_ENGINEER = "electrical_engineer"
    
    # General roles
    DESIGNER = "designer"
    ANALYST = "analyst"
    VALIDATOR = "validator"
    OPTIMIZER = "optimizer"


@dataclass
class RoleAssignment:
    """Assignment of an agent to a professional role"""
    role: ProfessionalRole
    agent: BaseAgent
    agent_capability: AgentCapability
    domain: str
    assigned_at: datetime = field(default_factory=datetime.now)


@dataclass
class TeamTaskResult:
    """Result from a team task execution"""
    role: ProfessionalRole
    result: Dict[str, Any]
    confidence: float
    execution_time: float
    dependencies_met: bool = True


class ProfessionalTeamOrchestrator:
    """
    Orchestrates multiple professional roles working together.
    
    Domains use this to simulate a team of professionals:
    - Construction: Architect + Engineer + PM + Inspector
    - Game Dev: Designer + Programmer + Artist + Sound Engineer
    - Robotics: Mechanical + Control + Simulation Engineers
    """
    
    def __init__(self, agent_manager: AgentManager, llm_engine: LLMEngine):
        """
        Initialize the professional team orchestrator.
        
        Args:
            agent_manager: AgentManager instance for agent coordination
            llm_engine: LLMEngine for role-based prompt generation
        """
        self.agent_manager = agent_manager
        self.llm_engine = llm_engine
        self.role_assignments: Dict[ProfessionalRole, RoleAssignment] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        self.role_prompts_cache: Dict[str, str] = {}
        
        logger.info("Professional Team Orchestrator initialized")
    
    async def process(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task using the professional team.
        This method makes ProfessionalTeamOrchestrator compatible with orchestrator's agent execution model.
        
        Args:
            task: Task dictionary with 'query' or 'description' key
            context: Context dictionary
        
        Returns:
            Result dictionary with 'answer' or 'result' key
        """
        try:
            # Extract task description
            task_description = task.get('query') or task.get('description') or task.get('task', '')
            if not task_description:
                return {"status": "error", "error": "No task description provided"}
            
            # Determine required roles from task or context
            required_roles = task.get('required_roles', [])
            if not required_roles:
                # Auto-detect roles based on task keywords
                task_lower = task_description.lower()
                if any(word in task_lower for word in ['design', 'layout', 'plan']):
                    required_roles = [ProfessionalRole.ARCHITECT, ProfessionalRole.DESIGNER]
                elif any(word in task_lower for word in ['analyze', 'validate', 'check']):
                    required_roles = [ProfessionalRole.ANALYST, ProfessionalRole.VALIDATOR]
                elif any(word in task_lower for word in ['schedule', 'plan', 'timeline']):
                    required_roles = [ProfessionalRole.PROJECT_MANAGER, ProfessionalRole.SCHEDULER]
                else:
                    # Default to general roles
                    required_roles = [ProfessionalRole.ANALYST]
            
            # Convert string roles to ProfessionalRole enum if needed
            if required_roles and isinstance(required_roles[0], str):
                required_roles = [ProfessionalRole(r) for r in required_roles if r in [role.value for role in ProfessionalRole]]
            
            # Coordinate team task
            result = await self.coordinate_team_task(
                task=task_description,
                required_roles=required_roles,
                context=context,
                domain=context.get('domain', 'general')
            )
            
            # Format result for orchestrator compatibility
            if isinstance(result, dict):
                # Extract consensus or first role result
                if 'team_consensus' in result:
                    return {
                        "status": "success",
                        "answer": result.get('team_consensus', {}).get('consensus', ''),
                        "result": result,
                        "confidence": result.get('team_consensus', {}).get('confidence', 0.8)
                    }
                elif result:
                    # Get first role result
                    first_result = next(iter(result.values())) if result else None
                    if first_result and hasattr(first_result, 'result'):
                        return {
                            "status": "success",
                            "answer": first_result.result.get('answer', str(first_result.result)),
                            "result": result,
                            "confidence": first_result.confidence
                        }
            
            return {
                "status": "success",
                "answer": str(result),
                "result": result,
                "confidence": 0.7
            }
            
        except Exception as e:
            logger.error(f"ProfessionalTeamOrchestrator.process failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "result": {}
            }
    
    async def assign_role(
        self,
        role: ProfessionalRole,
        agent: Optional[BaseAgent] = None,
        agent_capability: Optional[AgentCapability] = None,
        domain: str = "general"
    ) -> bool:
        """
        Assign an agent to a professional role.
        
        Args:
            role: Professional role to assign
            agent: Specific agent to assign (optional)
            agent_capability: Capability to find agent by (optional)
            domain: Domain context for the role
        
        Returns:
            True if assignment successful
        """
        try:
            # If agent not provided, find by capability
            if agent is None:
                if agent_capability is None:
                    # Map role to default capability
                    agent_capability = self._get_default_capability_for_role(role)
                
                # Find agent with this capability
                agents = self.agent_manager.find_agents_by_capability(agent_capability)
                if not agents:
                    logger.warning(f"No agent found for capability {agent_capability}")
                    return False
                agent = agents[0]  # Use first available agent
            
            # Create assignment
            assignment = RoleAssignment(
                role=role,
                agent=agent,
                agent_capability=agent_capability or AgentCapability.REASONING,
                domain=domain
            )
            
            self.role_assignments[role] = assignment
            logger.info(f"✅ Assigned {role.value} role to agent {agent.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to assign role {role.value}: {e}")
            return False
    
    def _get_default_capability_for_role(self, role: ProfessionalRole) -> AgentCapability:
        """Map professional role to default agent capability"""
        role_capability_map = {
            ProfessionalRole.ARCHITECT: AgentCapability.DESIGN,
            ProfessionalRole.STRUCTURAL_ENGINEER: AgentCapability.ANALYSIS,
            ProfessionalRole.MEP_ENGINEER: AgentCapability.ANALYSIS,
            ProfessionalRole.PROJECT_MANAGER: AgentCapability.PLANNING,
            ProfessionalRole.COST_ESTIMATOR: AgentCapability.ANALYSIS,
            ProfessionalRole.SCHEDULER: AgentCapability.PLANNING,
            ProfessionalRole.INSPECTOR: AgentCapability.VALIDATION,
            ProfessionalRole.SAFETY_OFFICER: AgentCapability.RISK_ASSESSMENT,
            ProfessionalRole.GAME_DESIGNER: AgentCapability.CREATIVE_SYNTHESIS,
            ProfessionalRole.PROGRAMMER: AgentCapability.REASONING,
            ProfessionalRole.ARTIST: AgentCapability.CREATIVE_SYNTHESIS,
            ProfessionalRole.SOUND_ENGINEER: AgentCapability.CREATIVE_SYNTHESIS,
            ProfessionalRole.QA_TESTER: AgentCapability.VALIDATION,
            ProfessionalRole.MECHANICAL_ENGINEER: AgentCapability.DESIGN,
            ProfessionalRole.CONTROL_ENGINEER: AgentCapability.REASONING,
            ProfessionalRole.SYSTEMS_ENGINEER: AgentCapability.ORCHESTRATION,
            ProfessionalRole.TEST_ENGINEER: AgentCapability.VALIDATION,
            ProfessionalRole.THERMAL_ENGINEER: AgentCapability.ANALYSIS,
            ProfessionalRole.ELECTRICAL_ENGINEER: AgentCapability.ANALYSIS,
            ProfessionalRole.DESIGNER: AgentCapability.DESIGN,
            ProfessionalRole.ANALYST: AgentCapability.ANALYSIS,
            ProfessionalRole.VALIDATOR: AgentCapability.VALIDATION,
            ProfessionalRole.OPTIMIZER: AgentCapability.OPTIMIZATION,
        }
        return role_capability_map.get(role, AgentCapability.REASONING)
    
    async def coordinate_team_task(
        self,
        task: str,
        required_roles: List[ProfessionalRole],
        context: Dict[str, Any],
        dependencies: Optional[Dict[str, List[str]]] = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """
        Coordinate multiple professionals working on a task.
        
        Example for construction:
        - Architect designs the layout
        - Engineer validates structural integrity
        - PM creates schedule
        - Cost estimator provides budget
        
        Args:
            task: Description of the task
            required_roles: List of professional roles needed
            context: Context information for the task
            dependencies: Optional dependencies between roles (role_name -> [required_roles])
            domain: Domain context
        
        Returns:
            Dict with results from each role and team consensus
        """
        logger.info(f"👥 Coordinating team task: {task}")
        logger.info(f"   Required roles: {[r.value for r in required_roles]}")
        
        # Check all roles are assigned
        missing_roles = [r for r in required_roles if r not in self.role_assignments]
        if missing_roles:
            logger.warning(f"⚠️ Missing role assignments: {[r.value for r in missing_roles]}")
            # Try to auto-assign missing roles
            for role in missing_roles:
                await self.assign_role(role, domain=domain)
        
        # Execute roles respecting dependencies
        results: Dict[str, TeamTaskResult] = {}
        completed_roles = set()
        
        # Build dependency graph
        if dependencies is None:
            dependencies = {}
        
        # Execute in rounds (respecting dependencies)
        max_rounds = len(required_roles) * 2  # Safety limit
        round_num = 0
        
        while len(completed_roles) < len(required_roles) and round_num < max_rounds:
            round_num += 1
            
            # Find roles ready to execute (dependencies met)
            ready_roles = [
                role for role in required_roles
                if role not in completed_roles
                and all(
                    dep_role in completed_roles
                    for dep_role in dependencies.get(role.value, [])
                )
            ]
            
            if not ready_roles:
                logger.warning("⚠️ Workflow deadlock: no roles ready to execute")
                break
            
            # Execute ready roles in parallel
            role_tasks = []
            for role in ready_roles:
                if role in self.role_assignments:
                    task_obj = self._execute_role_work(
                        role=role,
                        task=task,
                        context={**context, **{k: v.result for k, v in results.items()}},
                        domain=domain
                    )
                    role_tasks.append((role, task_obj))
            
            # Wait for all ready roles
            if role_tasks:
                role_results = await asyncio.gather(
                    *[task for _, task in role_tasks],
                    return_exceptions=True
                )
                
                # Store results
                for (role, _), result in zip(role_tasks, role_results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Role {role.value} failed: {result}")
                        results[role.value] = TeamTaskResult(
                            role=role,
                            result={"error": str(result)},
                            confidence=0.0,
                            execution_time=0.0,
                            dependencies_met=False
                        )
                    else:
                        results[role.value] = result
                        completed_roles.add(role)
                        logger.info(f"✅ {role.value} completed")
        
        # Get team consensus if multiple roles
        team_consensus = None
        if len(required_roles) > 1 and all(r.value in results for r in required_roles):
            team_consensus = await self._get_team_consensus(
                results=results,
                task=task,
                context=context,
                domain=domain
            )
        
        # Record workflow
        workflow_record = {
            "task": task,
            "roles": [r.value for r in required_roles],
            "results": {k: {
                "confidence": v.confidence,
                "execution_time": v.execution_time,
                "summary": str(v.result)[:200]
            } for k, v in results.items()},
            "consensus": team_consensus,
            "timestamp": datetime.now().isoformat(),
            "domain": domain
        }
        self.workflow_history.append(workflow_record)
        
        return {
            "task": task,
            "role_results": {k: v.result for k, v in results.items()},
            "team_consensus": team_consensus,
            "workflow_id": len(self.workflow_history) - 1,
            "all_roles_completed": len(completed_roles) == len(required_roles)
        }
    
    async def _execute_role_work(
        self,
        role: ProfessionalRole,
        task: str,
        context: Dict[str, Any],
        domain: str
    ) -> TeamTaskResult:
        """Execute a professional role's work"""
        start_time = datetime.now()
        
        assignment = self.role_assignments[role]
        agent = assignment.agent
        
        # Generate role-specific prompt using Llama 3.1 8B
        role_prompt = await self._get_role_prompt(role, task, context, domain)
        
        # Execute agent work
        try:
            agent_result = await agent.execute({
                "action": "professional_work",
                "role": role.value,
                "task": task,
                "context": context,
                "prompt": role_prompt,
                "domain": domain
            })
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Extract confidence if available
            confidence = agent_result.get("confidence", 0.7)
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except:
                    confidence = 0.7
            
            return TeamTaskResult(
                role=role,
                result=agent_result,
                confidence=confidence,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error executing role {role.value}: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            return TeamTaskResult(
                role=role,
                result={"error": str(e)},
                confidence=0.0,
                execution_time=execution_time
            )
    
    async def _get_role_prompt(
        self,
        role: ProfessionalRole,
        task: str,
        context: Dict[str, Any],
        domain: str
    ) -> str:
        """Generate role-specific prompt using Llama 3.1 8B"""
        # Check cache
        cache_key = f"{role.value}_{domain}_{task[:50]}"
        if cache_key in self.role_prompts_cache:
            return self.role_prompts_cache[cache_key]
        
        # Get role description
        role_description = self._get_role_description(role, domain)
        
        # Generate prompt using LLM
        prompt_generation = f"""Generate a professional prompt for a {role.value} working on a {domain} project.

Role: {role.value}
Role Description: {role_description}

Task: {task}

Context: {context}

Generate a detailed, professional prompt that:
1. Clearly defines the {role.value}'s responsibilities for this task
2. Provides domain-specific guidance ({domain})
3. Includes relevant context and constraints
4. Specifies the expected output format

Return only the prompt, no explanations."""

        try:
            llm_response = await self.llm_engine.generate(
                prompt=prompt_generation,
                max_tokens=300,
                temperature=0.7
            )
            
            # Extract prompt from response
            if isinstance(llm_response, dict):
                prompt = llm_response.get("text", str(llm_response))
            else:
                prompt = str(llm_response)
            
            # Cache it
            self.role_prompts_cache[cache_key] = prompt
            return prompt
            
        except Exception as e:
            logger.warning(f"Failed to generate role prompt with LLM: {e}, using template")
            # Fallback to template
            return self._get_template_prompt(role, task, context, domain)
    
    def _get_role_description(self, role: ProfessionalRole, domain: str) -> str:
        """Get description of a professional role"""
        descriptions = {
            ProfessionalRole.ARCHITECT: f"""
            You are a professional architect specializing in {domain} projects. Your role:
            - Design functional and aesthetic layouts
            - Ensure code compliance and building standards
            - Create architectural drawings and specifications
            - Consider user experience, flow, and accessibility
            - Balance aesthetics with functionality and cost
            """,
            ProfessionalRole.STRUCTURAL_ENGINEER: f"""
            You are a professional structural engineer specializing in {domain} projects. Your role:
            - Validate structural integrity and safety
            - Calculate loads, stresses, and material requirements
            - Ensure compliance with structural codes and standards
            - Design structural systems and components
            - Verify safety factors and failure modes
            """,
            ProfessionalRole.PROJECT_MANAGER: f"""
            You are a professional project manager specializing in {domain} projects. Your role:
            - Create realistic schedules and timelines
            - Identify dependencies and critical paths
            - Manage risks and mitigation strategies
            - Coordinate team communication and deliverables
            - Track progress and adjust plans as needed
            """,
            ProfessionalRole.COST_ESTIMATOR: f"""
            You are a professional cost estimator specializing in {domain} projects. Your role:
            - Estimate material, labor, and equipment costs
            - Account for market conditions and location factors
            - Provide detailed cost breakdowns
            - Identify cost-saving opportunities
            - Validate estimates against historical data
            """,
            ProfessionalRole.GAME_DESIGNER: f"""
            You are a professional game designer. Your role:
            - Design game mechanics and systems
            - Create engaging gameplay loops
            - Balance game difficulty and progression
            - Design user interfaces and experiences
            - Ensure fun and player engagement
            """,
            ProfessionalRole.PROGRAMMER: f"""
            You are a professional programmer. Your role:
            - Write clean, efficient, maintainable code
            - Implement game mechanics and systems
            - Optimize performance and memory usage
            - Follow best practices and design patterns
            - Test and debug code thoroughly
            """,
            # Add more role descriptions as needed
        }
        
        default_description = f"""
        You are a professional {role.value} working on a {domain} project.
        Apply your expertise to complete the assigned task with high quality.
        """
        
        return descriptions.get(role, default_description)
    
    def _get_template_prompt(
        self,
        role: ProfessionalRole,
        task: str,
        context: Dict[str, Any],
        domain: str
    ) -> str:
        """Fallback template prompt if LLM generation fails"""
        role_desc = self._get_role_description(role, domain)
        return f"""{role_desc}

Task: {task}

Context:
{context}

Please complete this task as a professional {role.value}, providing:
1. Your analysis and approach
2. Your recommendations or solution
3. Any concerns or considerations
4. Expected outcomes

Provide your response in a professional, detailed format."""
    
    async def _get_team_consensus(
        self,
        results: Dict[str, TeamTaskResult],
        task: str,
        context: Dict[str, Any],
        domain: str
    ) -> Dict[str, Any]:
        """Get consensus from multiple professional roles using Llama 3.1 8B"""
        # Prepare role results summary
        role_summaries = []
        for role_name, result in results.items():
            role_summaries.append(
                f"{role_name}: {str(result.result)[:200]} (confidence: {result.confidence:.2f})"
            )
        
        consensus_prompt = f"""As a team coordinator, synthesize consensus from multiple professionals working on a {domain} project.

Task: {task}

Professional Inputs:
{chr(10).join(role_summaries)}

Context: {context}

Provide:
1. A unified consensus recommendation
2. Areas of agreement between professionals
3. Any conflicts or disagreements and how to resolve them
4. Final recommendation with confidence level

Return as JSON with keys: consensus, agreements, conflicts, recommendation, confidence."""

        try:
            llm_response = await self.llm_engine.generate(
                prompt=consensus_prompt,
                max_tokens=500,
                temperature=0.5
            )
            
            if isinstance(llm_response, dict):
                consensus_text = llm_response.get("text", str(llm_response))
            else:
                consensus_text = str(llm_response)
            
            # Try to parse JSON from response
            import json
            import re
            json_match = re.search(r'\{.*\}', consensus_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # Fallback to text response
            return {
                "consensus": consensus_text,
                "confidence": 0.7,
                "source": "llm_synthesis"
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate consensus: {e}")
            return {
                "consensus": "Team completed work, manual review recommended",
                "confidence": 0.5,
                "source": "fallback"
            }
    
    def get_team_status(self) -> Dict[str, Any]:
        """Get current team status"""
        return {
            "assigned_roles": [r.value for r in self.role_assignments.keys()],
            "total_workflows": len(self.workflow_history),
            "recent_workflows": self.workflow_history[-5:] if self.workflow_history else []
        }

