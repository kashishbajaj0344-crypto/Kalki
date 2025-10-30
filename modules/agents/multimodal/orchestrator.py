from typing import Dict, Any, List, Optional, Set
import asyncio
import logging
import time

from ..base_agent import BaseAgent, AgentStatus

logger = logging.getLogger("kalki.orchestrator")


class DependencyError(Exception):
    pass


class AgentOrchestrator:
    """Dependency-aware orchestrator for multimodal agents."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None, logger: Optional[logging.Logger] = None):
        self._agents: Dict[str, BaseAgent] = {}
        self._init_order: List[str] = []
        self._initialized: Set[str] = set()
        try:
            self._loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            self._loop = loop
        self._logger = logger or logging.getLogger("kalki.orchestrator")

    def register_agents(self, agents: List[BaseAgent]) -> None:
        for agent in agents:
            if agent.name in self._agents:
                self._logger.warning("[Orchestrator] Agent %s already registered, skipping", agent.name)
                continue
            self._agents[agent.name] = agent
        self._logger.debug("[Orchestrator] Registered agents: %s", list(self._agents.keys()))

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for name, agent in self._agents.items():
            deps = getattr(agent, "dependencies", []) or []
            graph[name] = deps
        return graph

    def _topological_sort(self) -> List[str]:
        graph = self._build_dependency_graph()
        visited: Set[str] = set()
        temp: Set[str] = set()
        order: List[str] = []

        def visit(node: str):
            if node in temp:
                raise DependencyError(f"Cyclic dependency detected at {node}")
            if node not in visited:
                temp.add(node)
                for dep in graph.get(node, []):
                    if dep not in self._agents:
                        raise DependencyError(f"Missing dependency '{dep}' required by '{node}'")
                    visit(dep)
                temp.remove(node)
                visited.add(node)
                order.append(node)

        for n in graph:
            if n not in visited:
                visit(n)

        return order

    async def initialize_all(self, parallel: bool = False) -> bool:
        try:
            order = self._topological_sort()
            self._init_order = order
            self._logger.info("[Orchestrator] Computed init order: %s", order)
        except DependencyError as e:
            self._logger.exception("[Orchestrator] Dependency resolution failed: %s", e)
            return False

        if not parallel:
            for name in self._init_order:
                if name in self._initialized:
                    self._logger.debug("[Orchestrator] Agent %s already initialized, skipping", name)
                    continue
                agent = self._agents[name]
                try:
                    ok = await agent.initialize()
                    if not ok:
                        self._logger.error("[Orchestrator] Agent %s failed to initialize", name)
                        return False
                    agent.update_status(AgentStatus.RUNNING)
                    self._initialized.add(name)
                    self._logger.info("[Orchestrator] Agent %s initialized", name)
                except Exception as e:
                    self._logger.exception("[Orchestrator] Initialization of agent %s failed: %s", name, e)
                    return False
            return True

        # Parallel initialization logic omitted here for brevity (kept in original __init__ file)
        return await self._initialize_parallel()

    async def _initialize_parallel(self) -> bool:
        pending = set(self._agents.keys()) - self._initialized
        while pending:
            batch = []
            for name in list(pending):
                deps = getattr(self._agents[name], "dependencies", []) or []
                if all(d in self._initialized for d in deps):
                    batch.append(name)
            if not batch:
                self._logger.error("[Orchestrator] Cannot progress initialization; unresolved dependencies remain: %s", pending)
                return False

            self._logger.debug("[Orchestrator] Initializing batch in parallel: %s", batch)
            tasks = [self._safe_initialize(self._agents[name], name) for name in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for name, res in zip(batch, results):
                if isinstance(res, Exception) or res is False:
                    self._logger.error("[Orchestrator] Agent %s failed to initialize in parallel batch: %s", name, res)
                    return False
                self._initialized.add(name)
                pending.discard(name)
                self._logger.info("[Orchestrator] Agent %s initialized (parallel)", name)

        return True

    async def _safe_initialize(self, agent: BaseAgent, name: str) -> bool:
        try:
            ok = await agent.initialize()
            if ok:
                agent.update_status(AgentStatus.RUNNING)
            return ok
        except Exception as e:
            self._logger.exception("[Orchestrator] Exception initializing agent %s: %s", name, e)
            return False

    async def execute_for_agent(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        if agent_name not in self._agents:
            msg = f"Agent '{agent_name}' not registered"
            self._logger.error("[Orchestrator] %s", msg)
            return {"status": "error", "error": msg}

        if not isinstance(task, dict) or "action" not in task:
            msg = "Invalid task: missing 'action' key"
            self._logger.error("[Orchestrator] %s", msg)
            return {"status": "error", "error": msg}

        try:
            await self._ensure_dependencies_initialized(agent_name)
        except DependencyError as e:
            self._logger.exception("[Orchestrator] Failed to ensure dependencies for %s: %s", agent_name, e)
            return {"status": "error", "error": str(e)}

        agent = self._agents[agent_name]
        start = time.perf_counter()
        try:
            result = await agent.execute(task)
        except Exception as e:
            self._logger.exception("[Orchestrator] Execution error in agent %s: %s", agent_name, e)
            return {"status": "error", "error": str(e)}
        finally:
            duration = time.perf_counter() - start

        if isinstance(result, dict):
            result.setdefault("meta", {})
            result["meta"]["duration_seconds"] = duration
            result["meta"]["agent"] = agent_name
        else:
            result = {"status": "success", "result": result, "meta": {"duration_seconds": duration, "agent": agent_name}}

        self._logger.debug("[Orchestrator] Executed %s on %s in %.4fs", task.get("action"), agent_name, duration)
        return result

    async def _ensure_dependencies_initialized(self, agent_name: str) -> None:
        agent = self._agents[agent_name]
        deps = getattr(agent, "dependencies", []) or []
        for dep in deps:
            if dep not in self._agents:
                raise DependencyError(f"Missing dependency '{dep}' required by '{agent_name}'")
            if dep not in self._initialized:
                await self._ensure_dependencies_initialized(dep)
                dep_agent = self._agents[dep]
                if dep not in self._initialized:
                    ok = await dep_agent.initialize()
                    if not ok:
                        raise DependencyError(f"Initialization failed for dependency '{dep}'")
                    dep_agent.update_status(AgentStatus.RUNNING)
                    self._initialized.add(dep)
                    self._logger.info("[Orchestrator] Dependency agent %s initialized for %s", dep, agent_name)

    async def broadcast_task(self, agent_names: List[str], task: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        missing = [n for n in agent_names if n not in self._agents]
        if missing:
            msg = f"Agents not registered: {missing}"
            self._logger.error("[Orchestrator] %s", msg)
            return {n: {"status": "error", "error": msg} for n in agent_names}

        coros = [self.execute_for_agent(n, task) for n in agent_names]
        results = await asyncio.gather(*coros, return_exceptions=False)
        return {name: res for name, res in zip(agent_names, results)}

    def agent_status(self, agent_name: str) -> Optional[AgentStatus]:
        agent = self._agents.get(agent_name)
        return getattr(agent, "status", None) if agent else None

    def health_check(self, agent_name: str) -> Dict[str, Any]:
        if agent_name not in self._agents:
            return {"status": "error", "error": "not_registered"}
        agent = self._agents[agent_name]
        status = getattr(agent, "status", None)
        health_callable = getattr(agent, "health", None)
        return {"status": status.value if status else "unknown", "has_health_callable": bool(health_callable)}
