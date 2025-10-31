from typing import Dict, Any, List
import logging

from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ..creative import CreativeAgent, DreamModeAgent, IdeaFusionAgent, PatternRecognitionAgent


class CopilotAgent(BaseAgent):
    """Copilot agent that orchestrates plan → execute → feedback cycles"""

    def __init__(self, agent_manager=None, config: Dict[str, Any] = None):
        super().__init__(
            name="CopilotAgent",
            capabilities=[AgentCapability.ORCHESTRATION],
            description="Orchestrates complex multi-agent workflows with planning and feedback loops",
            config=config or {}
        )
        self.agent_manager = agent_manager
        self.logger = logging.getLogger("kalki.agent.CopilotAgent")

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        params = task.get("params", {})

        if action == "orchestrate_workflow":
            return await self._orchestrate_workflow(params)
        elif action == "plan_execute_feedback":
            return await self._plan_execute_feedback_cycle(params)
        elif action == "generate_ideas":
            return await self.generate_creative_ideas(
                params.get("domain", "technology"),
                params.get("count", 5)
            )
        elif action == "creative_exploration":
            return await self.creative_exploration_workflow(params.get("theme", ""))
        elif action == "idea_to_plan":
            return await self.idea_to_plan_workflow(params.get("idea_id", ""))
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _orchestrate_workflow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate a complex workflow using multiple agents"""
        try:
            workflow_name = params.get("workflow", "default")
            goal = params.get("goal", "")
            max_iterations = params.get("max_iterations", 3)

            self.logger.info(f"Starting workflow orchestration: {workflow_name}")

            # Define workflow based on type
            if workflow_name == "document_processing":
                result = await self._document_processing_workflow(goal, max_iterations)
            elif workflow_name == "research_analysis":
                result = await self._research_analysis_workflow(goal, max_iterations)
            else:
                result = await self._generic_workflow(goal, max_iterations)

            return result

        except Exception as e:
            self.logger.exception(f"Workflow orchestration error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _plan_execute_feedback_cycle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a plan → execute → feedback cycle with session tracking"""
        try:
            goal = params.get("goal", "")
            max_cycles = params.get("max_cycles", 3)
            user_id = params.get("user_id", "default_user")

            self.logger.info(f"Starting plan-execute-feedback cycle for: {goal}")

            # Create or get active session
            session_id = await self._ensure_active_session(user_id, goal)
            
            cycle_results = []
            current_plan = None

            for cycle in range(max_cycles):
                self.logger.info(f"Starting cycle {cycle + 1}/{max_cycles}")

                # Update session with current cycle
                await self._update_session_context(session_id, {
                    "cycle": cycle + 1,
                    "phase": "planning",
                    "goal": goal
                })

                # Phase 1: Plan
                plan_result = await self._get_planning_assistance(goal, current_plan, cycle_results)
                if plan_result["status"] != "success":
                    await self._update_session_context(session_id, {
                        "cycle": cycle + 1,
                        "phase": "planning_failed",
                        "error": plan_result.get("error")
                    })
                    return {
                        "status": "error",
                        "error": f"Planning failed in cycle {cycle + 1}: {plan_result.get('error')}"
                    }

                current_plan = plan_result.get("plan", [])

                # Update session with plan
                await self._update_session_context(session_id, {
                    "cycle": cycle + 1,
                    "phase": "plan_created",
                    "plan": current_plan
                })

                # Phase 2: Execute
                execution_result = await self._execute_plan_with_agents(current_plan, goal)
                
                # Store execution event in memory
                await self._store_execution_event(session_id, goal, current_plan, execution_result)
                
                cycle_results.append({
                    "cycle": cycle + 1,
                    "plan": current_plan,
                    "execution": execution_result
                })

                # Update session with execution results
                await self._update_session_context(session_id, {
                    "cycle": cycle + 1,
                    "phase": "execution_completed",
                    "execution_status": execution_result.get("status"),
                    "completed_steps": execution_result.get("completed_steps", 0)
                })

                # Phase 3: Feedback and assessment
                feedback_result = await self._assess_execution_quality(execution_result, goal)

                # Check if we should continue or are satisfied
                if feedback_result.get("satisfied", False):
                    self.logger.info(f"Goal satisfied after {cycle + 1} cycles")
                    await self._update_session_context(session_id, {
                        "cycle": cycle + 1,
                        "phase": "goal_satisfied",
                        "final_result": "success"
                    })
                    break

                # Adjust plan based on feedback for next cycle
                current_plan = await self._refine_plan_based_on_feedback(current_plan, feedback_result)

            # Close session on completion
            await self._close_session(session_id, cycle_results)

            return {
                "status": "success",
                "goal": goal,
                "cycles_completed": len(cycle_results),
                "final_plan": current_plan,
                "cycle_results": cycle_results,
                "session_id": session_id,
                "satisfied": cycle_results[-1]["execution"].get("status") == "success" if cycle_results else False
            }

        except Exception as e:
            self.logger.exception(f"Plan-execute-feedback cycle error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _get_planning_assistance(self, goal: str, previous_plan: List[Dict] = None,
                                     cycle_history: List[Dict] = None) -> Dict[str, Any]:
        """Get planning assistance from PlannerAgent"""
        if not self.agent_manager:
            return {"status": "error", "error": "No agent manager available"}

        planning_params = {
            "goal": goal,
            "max_steps": 5
        }

        # Include context from previous cycles
        if previous_plan:
            planning_params["previous_plan"] = previous_plan
        if cycle_history:
            planning_params["cycle_history"] = cycle_history

        planning_task = {
            "action": "reasoning_guided_plan",
            "params": planning_params
        }

        return await self.agent_manager.execute_by_capability(
            AgentCapability.PLANNING,
            planning_task,
            strategy="optimal"
        )

    async def _execute_plan_with_agents(self, plan: List[Dict], goal: str) -> Dict[str, Any]:
        """Execute a plan using multiple agents"""
        if not self.agent_manager:
            return {"status": "error", "error": "No agent manager available"}

        execution_task = {
            "action": "plan_and_execute",
            "params": {
                "goal": goal,
                "plan": plan
            }
        }

        return await self.agent_manager.execute_by_capability(
            AgentCapability.PLANNING,
            execution_task,
            strategy="optimal"
        )

    async def _assess_execution_quality(self, execution_result: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Assess the quality of execution and determine if goal is satisfied"""
        if not self.agent_manager:
            return {"satisfied": False, "feedback": "No agent manager available"}

        assessment_task = {
            "action": "reason",
            "params": {
                "query": f"Assess if this execution result satisfies the goal: '{goal}'. Result: {execution_result}",
                "steps": 2
            }
        }

        assessment = await self.agent_manager.execute_by_capability(
            AgentCapability.REASONING,
            assessment_task,
            strategy="optimal"
        )

        if assessment.get("status") == "success":
            answer = assessment.get("answer", "").lower()
            satisfied = "yes" in answer or "satisfied" in answer or "achieved" in answer
            return {
                "satisfied": satisfied,
                "assessment": answer,
                "feedback": assessment
            }
        else:
            return {
                "satisfied": False,
                "assessment": "Assessment failed",
                "feedback": assessment
            }

    async def _refine_plan_based_on_feedback(self, current_plan: List[Dict],
                                           feedback: Dict[str, Any]) -> List[Dict]:
        """Refine the plan based on execution feedback"""
        if not self.agent_manager:
            return current_plan

        refinement_task = {
            "action": "reason",
            "params": {
                "query": f"Based on this feedback, suggest improvements to the plan: {feedback}. Current plan: {current_plan}",
                "steps": 2
            }
        }

        refinement = await self.agent_manager.execute_by_capability(
            AgentCapability.REASONING,
            refinement_task,
            strategy="optimal"
        )

        if refinement.get("status") == "success":
            # For now, return the original plan - in a full implementation,
            # this would parse the refinement suggestions and modify the plan
            self.logger.info(f"Plan refinement suggested: {refinement.get('answer', '')}")
            return current_plan
        else:
            return current_plan

    async def _document_processing_workflow(self, goal: str, max_iterations: int) -> Dict[str, Any]:
        """Document processing workflow: ingest → embed → search → analyze"""
        workflow_steps = [
            {"agent": "DocumentIngestAgent", "action": "ingest", "params": {"source": goal}},
            {"agent": "SearchAgent", "action": "embed", "params": {"wait_for_ingestion": True}},
            {"agent": "SearchAgent", "action": "search", "params": {"query": f"Analyze: {goal}"}},
            {"agent": "ReasoningAgent", "action": "reason", "params": {"query": f"Summarize findings about: {goal}"}}
        ]

        return await self._execute_workflow_steps(workflow_steps, goal)

    async def _research_analysis_workflow(self, goal: str, max_iterations: int) -> Dict[str, Any]:
        """Research analysis workflow: search → reason → plan → validate"""
        workflow_steps = [
            {"agent": "SearchAgent", "action": "search", "params": {"query": goal}},
            {"agent": "ReasoningAgent", "action": "reason", "params": {"query": f"Analyze research on: {goal}"}},
            {"agent": "PlannerAgent", "action": "plan", "params": {"goal": f"Apply research findings to: {goal}"}},
            {"agent": "EthicsAgent", "action": "review", "params": {"action": f"Research application: {goal}"}}
        ]

        return await self._execute_workflow_steps(workflow_steps, goal)

    async def _generic_workflow(self, goal: str, max_iterations: int) -> Dict[str, Any]:
        """Generic workflow using plan-execute-feedback cycle"""
        return await self._plan_execute_feedback_cycle({"goal": goal, "max_cycles": max_iterations})

    async def _execute_workflow_steps(self, steps: List[Dict], goal: str) -> Dict[str, Any]:
        """Execute a sequence of workflow steps"""
        results = []

        for step in steps:
            agent_name = step["agent"]
            task = {
                "action": step["action"],
                "params": step["params"]
            }

            result = await self.agent_manager.execute_task(agent_name, task)
            results.append({
                "step": step,
                "result": result
            })

            # Stop on critical failure
            if result.get("status") == "error":
                break

        return {
            "status": "success",
            "workflow": "custom",
            "goal": goal,
            "steps_executed": len(results),
            "results": results
        }

    async def _ensure_active_session(self, user_id: str, goal: str) -> str:
        """Ensure there's an active session for the user"""
        if not self.agent_manager:
            return f"session_{user_id}_fallback"

        try:
            session_task = {
                "action": "create",
                "user_id": user_id,
                "metadata": {
                    "goal": goal,
                    "agent": "CopilotAgent",
                    "workflow_type": "plan_execute_feedback"
                }
            }

            result = await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                session_task,
                strategy="first"
            )

            if result.get("status") == "success":
                return result.get("session_id", f"session_{user_id}_fallback")
            else:
                self.logger.warning("Failed to create session, using fallback")
                return f"session_{user_id}_fallback"
        except Exception as e:
            self.logger.exception(f"Failed to ensure active session: {e}")
            return f"session_{user_id}_fallback"

    async def _update_session_context(self, session_id: str, context_update: Dict[str, Any]):
        """Update session context"""
        if not self.agent_manager or session_id.endswith("_fallback"):
            return

        try:
            update_task = {
                "action": "update",
                "session_id": session_id,
                "context_update": context_update
            }

            await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                update_task,
                strategy="first"
            )
        except Exception as e:
            self.logger.debug(f"Failed to update session context: {e}")

    async def _store_execution_event(self, session_id: str, goal: str, plan: List[Dict], execution_result: Dict[str, Any]):
        """Store execution event in episodic memory"""
        if not self.agent_manager:
            return

        try:
            event = {
                "type": "workflow_execution",
                "session_id": session_id,
                "goal": goal,
                "plan_steps": len(plan),
                "execution_status": execution_result.get("status"),
                "completed_steps": execution_result.get("completed_steps", 0),
                "summary": f"Executed {len(plan)} steps for goal: {goal[:100]}..."
            }

            memory_task = {
                "action": "store",
                "type": "episodic",
                "event": event
            }

            await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                memory_task,
                strategy="first"
            )
        except Exception as e:
            self.logger.debug(f"Failed to store execution event: {e}")

    async def _close_session(self, session_id: str, cycle_results: List[Dict]):
        """Close the session with final results"""
        if not self.agent_manager or session_id.endswith("_fallback"):
            return

        try:
            # Update with final results
            final_context = {
                "phase": "completed",
                "total_cycles": len(cycle_results),
                "final_status": cycle_results[-1]["execution"].get("status") if cycle_results else "unknown",
                "summary": f"Completed workflow with {len(cycle_results)} cycles"
            }

            await self._update_session_context(session_id, final_context)

            # Close the session
            close_task = {
                "action": "close",
                "session_id": session_id
            }

            await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                close_task,
                strategy="first"
            )
        except Exception as e:
            self.logger.debug(f"Failed to close session: {e}")

    # Creative Workflow Methods
    async def generate_creative_ideas(self, domain: str, count: int = 5) -> Dict[str, Any]:
        """Generate multiple creative ideas in a domain"""
        try:
            creative_agent = await self._get_agent("CreativeAgent")
            if not creative_agent:
                return {"status": "error", "error": "CreativeAgent not available"}

            ideas = []
            for i in range(count):
                result = await creative_agent.generate_idea(domain)
                if result["status"] == "success":
                    ideas.append(result["idea"])
                else:
                    self.logger.warning(f"Failed to generate idea {i+1}: {result.get('error')}")

            return {
                "status": "success",
                "ideas": ideas,
                "count": len(ideas),
                "domain": domain
            }
        except Exception as e:
            self.logger.exception("Failed to generate creative ideas")
            return {"status": "error", "error": str(e)}

    async def creative_exploration_workflow(self, theme: str) -> Dict[str, Any]:
        """Complete creative exploration: dream → analyze → fuse"""
        try:
            # Get agents
            dream_agent = await self._get_agent("DreamModeAgent")
            pattern_agent = await self._get_agent("PatternRecognitionAgent")
            fusion_agent = await self._get_agent("IdeaFusionAgent")

            if not all([dream_agent, pattern_agent, fusion_agent]):
                return {"status": "error", "error": "Required agents not available"}

            # Step 1: Create dream session and generate dreams
            session_result = await dream_agent.create_dream_session(theme=theme)
            if session_result["status"] != "success":
                return session_result

            session_id = session_result["session"]["session_id"]

            # Generate dreams in the session
            dreams = []
            for i in range(3):  # Generate 3 dreams
                dream_result = await dream_agent.generate_dream(session_id)
                if dream_result["status"] == "success":
                    dreams.append(dream_result["dream"])

            # Step 2: Analyze patterns in dreams
            pattern_result = await pattern_agent.analyze_patterns(
                dreams, analysis_type="creative_themes", domain="mixed"
            )

            # Step 3: Fuse top ideas
            if len(dreams) >= 2:
                fusion_result = await fusion_agent.fuse_ideas(dreams[:3])  # Fuse top 3
                fusion = fusion_result if fusion_result["status"] == "success" else None
            else:
                fusion = None

            return {
                "status": "success",
                "theme": theme,
                "dreams": dreams,
                "patterns": pattern_result.get("patterns", []) if pattern_result["status"] == "success" else [],
                "fusion": fusion,
                "session_id": session_id
            }
        except Exception as e:
            self.logger.exception("Failed in creative exploration workflow")
            return {"status": "error", "error": str(e)}

    async def idea_to_plan_workflow(self, idea_id: str) -> Dict[str, Any]:
        """Convert a creative idea into an actionable plan"""
        try:
            creative_agent = await self._get_agent("CreativeAgent")
            if not creative_agent:
                return {"status": "error", "error": "CreativeAgent not available"}

            # Convert idea to plan
            plan_result = await creative_agent.convert_to_plan(idea_id)
            return plan_result
        except Exception as e:
            self.logger.exception("Failed in idea-to-plan workflow")
            return {"status": "error", "error": str(e)}

    async def _get_agent(self, agent_name: str):
        """Helper to get an agent from the agent manager"""
        if not self.agent_manager:
            self.logger.error("Agent manager not available")
            return None

        try:
            return await self.agent_manager.get_agent(agent_name)
        except Exception as e:
            self.logger.error(f"Failed to get agent {agent_name}: {e}")
            return None

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True