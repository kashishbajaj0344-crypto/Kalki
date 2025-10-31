from typing import Dict, Any, List
import logging

from ..base_agent import BaseAgent, AgentCapability, AgentStatus


class PlannerAgent(BaseAgent):
    """Enhanced task planning and decomposition agent with multi-agent coordination"""

    def __init__(self, agent_manager=None, config: Dict[str, Any] = None):
        super().__init__(
            name="PlannerAgent",
            capabilities=[AgentCapability.PLANNING],
            description="Decomposes complex tasks into executable sub-tasks with multi-agent coordination",
            config=config or {}
        )
        self.agent_manager = agent_manager
        self.logger = logging.getLogger("kalki.agent.PlannerAgent")

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

        if action == "plan":
            return await self._create_plan(params)
        elif action == "plan_and_execute":
            return await self._plan_and_execute(params)
        elif action == "reasoning_guided_plan":
            return await self._reasoning_guided_plan(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _plan_and_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan and execute it using available agents"""
        try:
            goal = params.get("goal", "")
            max_steps = params.get("max_steps", 5)

            # First, create the plan
            plan_result = await self._create_plan({"goal": goal, "max_steps": max_steps})
            if plan_result["status"] != "success":
                return plan_result

            plan = plan_result["plan"]
            execution_results = []

            # Execute each step using appropriate agents
            for step in plan:
                step_result = await self._execute_plan_step(step, params)
                execution_results.append({
                    "step": step,
                    "result": step_result
                })

                # Stop if a critical step fails
                if step_result.get("status") == "error":
                    break

            return {
                "status": "success",
                "goal": goal,
                "plan": plan,
                "execution_results": execution_results,
                "completed_steps": len([r for r in execution_results if r["result"].get("status") == "success"])
            }

        except Exception as e:
            self.logger.exception(f"Plan and execute error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _reasoning_guided_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a plan guided by reasoning from the ReasoningAgent"""
        try:
            goal = params.get("goal", "")

            # First, recall relevant memories for this goal
            relevant_memories = await self._recall_relevant_memories(goal)
            
            # Get reasoning from ReasoningAgent
            reasoning_result = await self._get_reasoning_assistance(goal, relevant_memories)
            if reasoning_result["status"] != "success":
                self.logger.warning("Failed to get reasoning assistance, falling back to basic planning")
                return await self._create_plan(params)

            reasoning = reasoning_result.get("answer", "")

            # Store reasoning in memory for future reference
            await self._store_reasoning_context(goal, reasoning, relevant_memories)

            # Create plan informed by reasoning and memories
            enhanced_params = params.copy()
            enhanced_params["reasoning_context"] = reasoning
            enhanced_params["relevant_memories"] = relevant_memories

            plan_result = await self._create_plan(enhanced_params)

            # Add reasoning metadata to result
            if plan_result["status"] == "success":
                plan_result["reasoning_used"] = reasoning
                plan_result["memories_used"] = len(relevant_memories)
                plan_result["planning_method"] = "reasoning_guided"

            return plan_result

        except Exception as e:
            self.logger.exception(f"Reasoning guided plan error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _get_reasoning_assistance(self, goal: str) -> Dict[str, Any]:
        """Get reasoning assistance from ReasoningAgent"""
        if not self.agent_manager:
            return {"status": "error", "error": "No agent manager available"}

        reasoning_task = {
            "action": "reason",
            "params": {
                "query": f"Analyze this goal and provide reasoning for how to approach it: {goal}",
                "steps": 3
            }
        }

        return await self.agent_manager.execute_by_capability(
            AgentCapability.REASONING,
            reasoning_task,
            strategy="optimal"
        )

    async def _recall_relevant_memories(self, goal: str) -> List[Dict[str, Any]]:
        """Recall memories relevant to the current goal"""
        if not self.agent_manager:
            return []

        try:
            # Extract key concepts from the goal for semantic memory search
            key_concepts = self._extract_key_concepts(goal)
            
            relevant_memories = []
            
            # Recall episodic memories (recent planning sessions)
            episodic_task = {
                "action": "recall",
                "type": "episodic",
                "limit": 5
            }
            
            episodic_result = await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                episodic_task,
                strategy="first"
            )
            
            if episodic_result.get("status") == "success":
                episodic_memories = episodic_result.get("memories", [])
                # Filter for planning-related memories
                planning_memories = [m for m in episodic_memories if "planning" in str(m.get("event", {})).lower()]
                relevant_memories.extend(planning_memories)
            
            # Recall semantic memories for key concepts
            for concept in key_concepts:
                semantic_task = {
                    "action": "recall",
                    "type": "semantic",
                    "concept": concept
                }
                
                semantic_result = await self.agent_manager.execute_by_capability(
                    AgentCapability.MEMORY,
                    semantic_task,
                    strategy="first"
                )
                
                if semantic_result.get("status") == "success":
                    semantic_memories = semantic_result.get("memories", [])
                    relevant_memories.extend(semantic_memories)
            
            self.logger.debug(f"Recalled {len(relevant_memories)} relevant memories for goal: {goal}")
            return relevant_memories
            
        except Exception as e:
            self.logger.exception(f"Failed to recall relevant memories: {e}")
            return []

    async def _store_reasoning_context(self, goal: str, reasoning: str, relevant_memories: List[Dict[str, Any]] = None) -> None:
        """Store reasoning context in MemoryAgent for future reference"""
        if not self.agent_manager:
            return

        try:
            # Store as episodic memory
            episodic_event = {
                "type": "planning_session",
                "goal": goal,
                "reasoning": reasoning,
                "memories_used": len(relevant_memories) if relevant_memories else 0,
                "summary": f"Planning session for: {goal[:100]}..."
            }
            
            episodic_task = {
                "action": "store",
                "type": "episodic",
                "event": episodic_event
            }
            
            await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                episodic_task,
                strategy="first"
            )
            
            # Store key concepts as semantic memory
            key_concepts = self._extract_key_concepts(goal)
            for concept in key_concepts:
                semantic_knowledge = {
                    "goal": goal,
                    "reasoning": reasoning,
                    "last_used": self.last_active.isoformat(),
                    "success_patterns": self._extract_success_patterns(reasoning)
                }
                
                semantic_task = {
                    "action": "store",
                    "type": "semantic",
                    "concept": concept,
                    "knowledge": semantic_knowledge
                }
                
                await self.agent_manager.execute_by_capability(
                    AgentCapability.MEMORY,
                    semantic_task,
                    strategy="first"
                )
                
        except Exception as e:
            self.logger.debug(f"Failed to store reasoning context in memory: {e}")

    def _extract_key_concepts(self, goal: str) -> List[str]:
        """Extract key concepts from a goal for memory retrieval"""
        # Simple keyword extraction - could be enhanced with NLP
        words = goal.lower().split()
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "how", "what", "when", "where", "why"}
        concepts = [word for word in words if len(word) > 3 and word not in stop_words]
        return concepts[:5]  # Limit to top 5 concepts

    def _extract_success_patterns(self, reasoning: str) -> List[str]:
        """Extract successful patterns from reasoning text"""
        # Simple pattern extraction - could be enhanced
        patterns = []
        if "step" in reasoning.lower():
            patterns.append("step_by_step_approach")
        if "analyze" in reasoning.lower():
            patterns.append("analysis_first")
        if "validate" in reasoning.lower():
            patterns.append("validation_important")
        return patterns

    async def _enhance_plan_with_memories(self, base_steps: List[Dict[str, Any]], 
                                        memories: List[Dict[str, Any]], goal: str) -> List[Dict[str, Any]]:
        """Enhance plan steps based on recalled memories"""
        enhanced_steps = base_steps.copy()
        
        try:
            # Analyze memories for successful patterns
            successful_patterns = []
            failed_patterns = []
            
            for memory in memories:
                event = memory.get("event", {})
                if event.get("result_status") == "success":
                    successful_patterns.extend(self._extract_patterns_from_memory(memory))
                elif event.get("result_status") == "error":
                    failed_patterns.extend(self._extract_patterns_from_memory(memory))
            
            # Add steps based on successful patterns
            if "step_by_step_approach" in successful_patterns and len(enhanced_steps) < 6:
                enhanced_steps.insert(1, {
                    "action": "break_down_goal",
                    "description": "Break down goal into detailed sub-steps"
                })
            
            if "analysis_first" in successful_patterns:
                # Ensure analysis comes first
                analysis_step = next((s for s in enhanced_steps if s["action"] == "analyze_goal"), None)
                if analysis_step:
                    enhanced_steps.remove(analysis_step)
                    enhanced_steps.insert(0, analysis_step)
            
            if "validation_important" in successful_patterns and not any(s["action"] == "validate_results" for s in enhanced_steps):
                enhanced_steps.append({
                    "action": "validate_results",
                    "description": "Validate outcomes against requirements"
                })
            
            # Avoid failed patterns
            if "rushed_execution" in failed_patterns:
                # Add a review step
                enhanced_steps.insert(-1, {
                    "action": "review_plan",
                    "description": "Review plan before execution"
                })
                
        except Exception as e:
            self.logger.debug(f"Failed to enhance plan with memories: {e}")
            return base_steps  # Return original steps on error
        
        return enhanced_steps

    def _extract_patterns_from_memory(self, memory: Dict[str, Any]) -> List[str]:
        """Extract patterns from a memory entry"""
        patterns = []
        event = memory.get("event", {})
        
        # Pattern extraction based on event characteristics
        if event.get("complexity") == "high" and event.get("result_status") == "success":
            patterns.append("step_by_step_approach")
        
        if "analysis" in str(event).lower() and event.get("result_status") == "success":
            patterns.append("analysis_first")
        
        if "validate" in str(event).lower():
            patterns.append("validation_important")
        
        # Add more pattern recognition logic here
        
        return patterns

    async def _execute_plan_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single plan step using appropriate agents"""
        try:
            step_action = step.get("action", "")
            step_description = step.get("description", "")

            # Determine which capability is needed for this step
            capability = self._map_action_to_capability(step_action)

            if not capability:
                return {
                    "status": "error",
                    "error": f"No capability mapping for action: {step_action}"
                }

            # Create task for the step
            step_task = {
                "action": step_action,
                "params": {
                    "description": step_description,
                    "context": context,
                    "complexity": "medium"
                }
            }

            # Execute using optimal agent selection
            result = await self.agent_manager.execute_by_capability(
                capability,
                step_task,
                strategy="optimal"
            )

            return result

        except Exception as e:
            self.logger.exception(f"Step execution error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _map_action_to_capability(self, action: str) -> AgentCapability:
        """Map plan step actions to agent capabilities"""
        action_mapping = {
            "analyze_goal": AgentCapability.REASONING,
            "gather_resources": AgentCapability.SEARCH,
            "execute_plan": AgentCapability.PLANNING,
            "validate_results": AgentCapability.REASONING,
            "search": AgentCapability.SEARCH,
            "reason": AgentCapability.REASONING,
            "store": AgentCapability.MEMORY,
            "retrieve": AgentCapability.MEMORY
        }

        return action_mapping.get(action)

    async def _create_plan(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            goal = params.get("goal", "")
            constraints = params.get("constraints", {})
            max_steps = params.get("max_steps", 4)
            reasoning_context = params.get("reasoning_context", "")
            relevant_memories = params.get("relevant_memories", [])

            # Recall additional relevant memories for this goal if not already provided
            if not relevant_memories:
                relevant_memories = await self._recall_relevant_memories(goal)

            # Create basic plan structure
            steps = [
                {"step": 1, "action": "analyze_goal", "description": f"Analyze: {goal}"},
                {"step": 2, "action": "gather_resources", "description": "Gather required resources"},
                {"step": 3, "action": "execute_plan", "description": "Execute planned actions"},
                {"step": 4, "action": "validate_results", "description": "Validate outcomes"}
            ]

            # Enhance plan based on recalled memories
            enhanced_steps = await self._enhance_plan_with_memories(steps, relevant_memories, goal)

            # Limit steps if specified
            enhanced_steps = enhanced_steps[:max_steps]

            # Renumber steps
            for i, step in enumerate(enhanced_steps):
                step["step"] = i + 1

            # Enhance with reasoning context if available
            if reasoning_context:
                enhanced_steps.insert(0, {
                    "step": 0,
                    "action": "reason",
                    "description": f"Incorporate reasoning: {reasoning_context[:100]}..."
                })
                # Renumber steps again
                for i, step in enumerate(enhanced_steps):
                    step["step"] = i + 1

            return {
                "status": "success",
                "goal": goal,
                "plan": enhanced_steps,
                "constraints": constraints,
                "reasoning_enhanced": bool(reasoning_context),
                "memories_used": len(relevant_memories)
            }

        except Exception as e:
            self.logger.exception(f"Planning error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
