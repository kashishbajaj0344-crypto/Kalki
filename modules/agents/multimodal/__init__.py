"""
Multi-modal agents (Phase 13, 17) - Vision, Audio, Sensor Fusion, AR/VR

Includes:
- VisionAgent
- AudioAgent
- SensorFusionAgent (depends on VisionAgent, AudioAgent)
- ARInsightAgent (depends on VisionAgent, SensorFusionAgent)

Also includes AgentOrchestrator: a dependency-aware orchestrator with:
- dependency resolution with cycle detection
- sequential and optional parallel initialization
- avoidance of redundant initialization
- contextual logging
- task validation and execution timing
- broadcast_task for parallel execution across agents
- agent_status / health_check helpers
"""
from typing import Dict, Any, List, Optional, Set
import asyncio
import logging
import time

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = logging.getLogger(__name__)


class VisionAgent(BaseAgent):
    """Visual processing and analysis agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="VisionAgent",
            capabilities=[AgentCapability.VISION],
            description="Processes and analyzes visual information",
            config=config or {}
        )

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute vision processing task

        Task format:
        {
            "action": "analyze|detect|classify",
            "params": {
                "image_path": str,
                "mode": str
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})

        if action == "analyze":
            return await self._analyze_image(params)
        elif action == "detect":
            return await self._detect_objects(params)
        elif action == "classify":
            return await self._classify_image(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _analyze_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image content"""
        try:
            image_path = params.get("image_path", "")

            analysis = {
                "description": "Visual content analysis",
                "dominant_colors": ["blue", "green", "white"],
                "composition": "balanced",
                "features_detected": ["faces", "objects", "text"]
            }

            return {
                "status": "success",
                "image_path": image_path,
                "analysis": analysis
            }

        except Exception as e:
            self.logger.exception(f"Image analysis error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _detect_objects(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect objects in image"""
        try:
            image_path = params.get("image_path", "")

            detections = [
                {"class": "person", "confidence": 0.95, "bbox": [100, 100, 200, 300]},
                {"class": "car", "confidence": 0.88, "bbox": [300, 150, 450, 280]}
            ]

            return {
                "status": "success",
                "image_path": image_path,
                "detections": detections,
                "count": len(detections)
            }

        except Exception as e:
            self.logger.exception(f"Object detection error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _classify_image(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Classify image content"""
        try:
            image_path = params.get("image_path", "")

            classification = {
                "primary_class": "landscape",
                "confidence": 0.92,
                "top_5": [
                    {"class": "landscape", "confidence": 0.92},
                    {"class": "nature", "confidence": 0.85},
                    {"class": "outdoor", "confidence": 0.78}
                ]
            }

            return {
                "status": "success",
                "image_path": image_path,
                "classification": classification
            }

        except Exception as e:
            self.logger.exception(f"Image classification error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True


class AudioAgent(BaseAgent):
    """Audio processing and analysis agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="AudioAgent",
            capabilities=[AgentCapability.AUDIO],
            description="Processes and analyzes audio information",
            config=config or {}
        )

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute audio processing task

        Task format:
        {
            "action": "transcribe|analyze|classify",
            "params": {
                "audio_path": str,
                "language": str
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})

        if action == "transcribe":
            return await self._transcribe_audio(params)
        elif action == "analyze":
            return await self._analyze_audio(params)
        elif action == "classify":
            return await self._classify_audio(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _transcribe_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio to text"""
        try:
            audio_path = params.get("audio_path", "")
            language = params.get("language", "en")

            transcription = {
                "text": "This is a sample transcription of the audio content.",
                "confidence": 0.94,
                "language": language,
                "duration_seconds": 15.5
            }

            return {
                "status": "success",
                "audio_path": audio_path,
                "transcription": transcription
            }

        except Exception as e:
            self.logger.exception(f"Audio transcription error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _analyze_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio characteristics"""
        try:
            audio_path = params.get("audio_path", "")

            analysis = {
                "sample_rate": 44100,
                "duration": 15.5,
                "channels": 2,
                "detected_features": ["speech", "music", "background_noise"],
                "loudness_db": -12.5,
                "tempo_bpm": 120
            }

            return {
                "status": "success",
                "audio_path": audio_path,
                "analysis": analysis
            }

        except Exception as e:
            self.logger.exception(f"Audio analysis error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _classify_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Classify audio content"""
        try:
            audio_path = params.get("audio_path", "")

            classification = {
                "primary_class": "speech",
                "confidence": 0.89,
                "top_3": [
                    {"class": "speech", "confidence": 0.89},
                    {"class": "conversation", "confidence": 0.76},
                    {"class": "indoor", "confidence": 0.65}
                ]
            }

            return {
                "status": "success",
                "audio_path": audio_path,
                "classification": classification
            }

        except Exception as e:
            self.logger.exception(f"Audio classification error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True


class SensorFusionAgent(BaseAgent):
    """Multi-sensor data fusion agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="SensorFusionAgent",
            capabilities=[AgentCapability.SENSOR_FUSION],
            description="Fuses data from multiple sensor modalities",
            dependencies=["VisionAgent", "AudioAgent"],
            config=config or {}
        )

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sensor fusion task

        Task format:
        {
            "action": "fuse|correlate|integrate",
            "params": {
                "sensor_data": dict,
                "modalities": list
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})

        if action == "fuse":
            return await self._fuse_sensors(params)
        elif action == "correlate":
            return await self._correlate_data(params)
        elif action == "integrate":
            return await self._integrate_modalities(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _fuse_sensors(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse multi-sensor data"""
        try:
            sensor_data = params.get("sensor_data", {})
            modalities = params.get("modalities", [])

            fusion_result = {
                "modalities_fused": modalities,
                "combined_confidence": 0.93,
                "insights": [
                    "Visual and audio data correlate",
                    "Scene understanding enhanced through fusion"
                ],
                "fused_representation": {
                    "scene": "office_meeting",
                    "participants": 3,
                    "activity": "discussion"
                }
            }

            return {
                "status": "success",
                "fusion": fusion_result
            }

        except Exception as e:
            self.logger.exception(f"Sensor fusion error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _correlate_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate data across modalities"""
        try:
            correlations = {
                "temporal_alignment": "synchronized",
                "spatial_alignment": "calibrated",
                "correlation_score": 0.87,
                "matched_events": [
                    {"timestamp": "2025-01-01T10:00:00", "modalities": ["vision", "audio"]}
                ]
            }

            return {
                "status": "success",
                "correlations": correlations
            }

        except Exception as e:
            self.logger.exception(f"Correlation error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _integrate_modalities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate multiple modalities"""
        try:
            modalities = params.get("modalities", [])

            integration = {
                "modalities": modalities,
                "integration_quality": "high",
                "enhanced_perception": True,
                "unified_model": {
                    "scene_understanding": 0.92,
                    "context_awareness": 0.88
                }
            }

            return {
                "status": "success",
                "integration": integration
            }

        except Exception as e:
            self.logger.exception(f"Integration error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True


class ARInsightAgent(BaseAgent):
    """Augmented Reality insights agent"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="ARInsightAgent",
            capabilities=[AgentCapability.AR_INSIGHTS],
            description="Provides augmented reality insights and overlays",
            dependencies=["VisionAgent", "SensorFusionAgent"],
            config=config or {}
        )

    async def initialize(self) -> bool:
        try:
            self.logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute AR task

        Task format:
        {
            "action": "generate_overlay|annotate|enhance",
            "params": {
                "scene_data": dict,
                "context": dict
            }
        }
        """
        action = task.get("action")
        params = task.get("params", {})

        if action == "generate_overlay":
            return await self._generate_overlay(params)
        elif action == "annotate":
            return await self._annotate_scene(params)
        elif action == "enhance":
            return await self._enhance_reality(params)
        else:
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    async def _generate_overlay(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AR overlay"""
        try:
            scene_data = params.get("scene_data", {})

            overlay = {
                "type": "information_overlay",
                "elements": [
                    {"type": "label", "text": "Object A", "position": [100, 200]},
                    {"type": "annotation", "text": "Interactive element", "position": [300, 150]}
                ],
                "render_mode": "3d",
                "interactive": True
            }

            return {
                "status": "success",
                "overlay": overlay
            }

        except Exception as e:
            self.logger.exception(f"Overlay generation error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _annotate_scene(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Annotate AR scene"""
        try:
            annotations = [
                {"object": "chair", "info": "Ergonomic design", "priority": "low"},
                {"object": "screen", "info": "Display active", "priority": "high"}
            ]

            return {
                "status": "success",
                "annotations": annotations
            }

        except Exception as e:
            self.logger.exception(f"Annotation error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _enhance_reality(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance reality with additional information"""
        try:
            enhancements = {
                "visual": ["highlight_objects", "add_dimensions"],
                "informational": ["display_metrics", "show_relationships"],
                "interactive": ["enable_selection", "contextual_menus"]
            }

            return {
                "status": "success",
                "enhancements": enhancements,
                "quality_improvement": "35%"
            }

        except Exception as e:
            self.logger.exception(f"Reality enhancement error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def shutdown(self) -> bool:
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True


# ---------------------------
# Agent Orchestration Helpers
# ---------------------------
class DependencyError(Exception):
    pass


class AgentOrchestrator:
    """
    Improved orchestrator addressing suggestions:
      - Avoid redundant initialization of already-initialized dependencies
      - Optional parallel initialization of independent agents
      - Unified and contextual logging
      - Task validation (ensure 'action' present)
      - Broadcast tasks to multiple agents in parallel
      - agent_status and health_check helpers
      - Execution timing metrics
      - Safer asyncio loop handling (uses get_running_loop when possible)
    """

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None, logger: Optional[logging.Logger] = None):
        self._agents: Dict[str, BaseAgent] = {}
        self._init_order: List[str] = []
        self._initialized: Set[str] = set()
        # prefer running loop when available, otherwise fall back to provided loop or None
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
        """
        Initialize all registered agents in dependency order.

        If parallel=True, independent agents (whose dependencies are already satisfied)
        will be initialized concurrently in batches.
        """
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

        # Parallel initialization: batch agents whose deps are satisfied
        pending = set(self._agents.keys()) - self._initialized
        while pending:
            # collect batch of agents whose dependencies are all satisfied
            batch = []
            for name in list(pending):
                deps = getattr(self._agents[name], "dependencies", []) or []
                if all(d in self._initialized for d in deps):
                    batch.append(name)
            if not batch:
                # circular or missing deps
                self._logger.error("[Orchestrator] Cannot progress initialization; unresolved dependencies remain: %s", pending)
                return False

            self._logger.debug("[Orchestrator] Initializing batch in parallel: %s", batch)
            tasks = []
            for name in batch:
                agent = self._agents[name]
                # skip if already initialized (just in case)
                if name in self._initialized:
                    pending.discard(name)
                    continue
                tasks.append(self._safe_initialize(agent, name))

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
        """
        Validate task, ensure dependencies are initialized, then execute the task.
        Returns the agent's result augmented with 'duration_seconds' for profiling.
        """
        if agent_name not in self._agents:
            msg = f"Agent '{agent_name}' not registered"
            self._logger.error("[Orchestrator] %s", msg)
            return {"status": "error", "error": msg}

        # simple task validation
        if not isinstance(task, dict) or "action" not in task:
            msg = "Invalid task: missing 'action' key"
            self._logger.error("[Orchestrator] %s", msg)
            return {"status": "error", "error": msg}

        # Ensure dependencies are initialized
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

        # attach duration meta for profiling
        if isinstance(result, dict):
            result.setdefault("meta", {})
            result["meta"]["duration_seconds"] = duration
            result["meta"]["agent"] = agent_name
        else:
            result = {"status": "success", "result": result, "meta": {"duration_seconds": duration, "agent": agent_name}}

        self._logger.debug("[Orchestrator] Executed %s on %s in %.4fs", task.get("action"), agent_name, duration)
        return result

    async def _ensure_dependencies_initialized(self, agent_name: str) -> None:
        """
        Recursively initialize dependencies for the given agent if not already initialized.
        Avoid re-initializing dependencies that are already in self._initialized.
        """
        agent = self._agents[agent_name]
        deps = getattr(agent, "dependencies", []) or []
        for dep in deps:
            if dep not in self._agents:
                raise DependencyError(f"Missing dependency '{dep}' required by '{agent_name}'")
            if dep not in self._initialized:
                # initialize dependencies of dependency first
                await self._ensure_dependencies_initialized(dep)
                dep_agent = self._agents[dep]
                # double-check: only call initialize once per agent
                if dep not in self._initialized:
                    ok = await dep_agent.initialize()
                    if not ok:
                        raise DependencyError(f"Initialization failed for dependency '{dep}'")
                    dep_agent.update_status(AgentStatus.RUNNING)
                    self._initialized.add(dep)
                    self._logger.info("[Orchestrator] Dependency agent %s initialized for %s", dep, agent_name)

    async def broadcast_task(self, agent_names: List[str], task: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Execute the same task on multiple agents in parallel.
        Returns a mapping agent_name -> result dict.
        """
        # filter and validate
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
        """
        Lightweight health check: returns registration and status info.
        Agents can implement richer checks if needed (by convention: .health() coroutine).
        """
        if agent_name not in self._agents:
            return {"status": "error", "error": "not_registered"}
        agent = self._agents[agent_name]
        status = getattr(agent, "status", None)
        # if agent exposes a coroutine health method, call it (non-blocking caller must await)
        health_callable = getattr(agent, "health", None)
        info = {"registered": True, "status": status}
        if callable(health_callable):
            info["health_method"] = "available (async) - call await orchestrator._call_agent_health(agent_name)"
        else:
            info["health_method"] = "not_available"
        return info

    async def _call_agent_health(self, agent_name: str) -> Dict[str, Any]:
        """
        Internal helper to call an agent's async health() method if present.
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            return {"status": "error", "error": "not_registered"}
        health_callable = getattr(agent, "health", None)
        if callable(health_callable):
            try:
                h = await health_callable()
                return {"status": "success", "health": h}
            except Exception as e:
                self._logger.exception("[Orchestrator] health() for %s raised: %s", agent_name, e)
                return {"status": "error", "error": str(e)}
        return {"status": "error", "error": "no_health_method"}

    async def shutdown_all(self) -> bool:
        """
        Shutdown all initialized agents in reverse initialization order.
        """
        if not self._init_order:
            try:
                self._init_order = self._topological_sort()
            except DependencyError as e:
                self._logger.warning("[Orchestrator] Could not compute init order during shutdown: %s", e)
                self._init_order = list(self._agents.keys())

        for name in reversed(self._init_order):
            agent = self._agents.get(name)
            if agent is None:
                continue
            try:
                ok = await agent.shutdown()
                if ok:
                    self._logger.info("[Orchestrator] Agent %s shut down", name)
                else:
                    self._logger.warning("[Orchestrator] Agent %s shutdown returned False", name)
            except Exception as e:
                self._logger.exception("[Orchestrator] Shutdown error for agent %s: %s", name, e)
        self._initialized.clear()
        return True