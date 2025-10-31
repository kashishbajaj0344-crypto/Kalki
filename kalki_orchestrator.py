#!/usr/bin/env python3
"""
kalki_orchestrator.py — Canonical Kalki v2.4 Production Entrypoint
====================================================================
Production-ready orchestrator that provides:
- Centralized configuration management
- Event-driven architecture initialization
- Multi-agent system orchestration
- Health monitoring and observability
- Graceful startup/shutdown handling
- CLI interface for different run modes

Usage:
    python kalki_orchestrator.py --help
    python kalki_orchestrator.py --start-all
    python kalki_orchestrator.py --ingest-only
    python kalki_orchestrator.py --agents-only
    python kalki_orchestrator.py --gui-only
"""

import argparse
import asyncio
import signal
import sys
from typing import Optional, Dict, Any
import os

from modules.config import CONFIG, get_config, DIRS
from modules.logging_config import setup_logging, get_logger
from modules.eventbus import EventBus
from modules.agents.agent_manager import AgentManager
from modules.llm import get_llm_engine
from modules.vectordb import VectorDBManager
from modules.metrics.collector import MetricsCollector
from modules.robustness import RobustnessManager
from modules.session import Session

# Kalki v2.4 — kalki_orchestrator.py v2.4.0

logger = get_logger("Kalki.Orchestrator")

class KalkiOrchestrator:
    """Main orchestrator for Kalki v2.4 production deployment"""

    def __init__(self):
        self.config = get_config()
        self.eventbus: Optional[EventBus] = None
        self.agent_manager: Optional[AgentManager] = None
        self.vector_db: Optional[VectorDBManager] = None
        self.metrics: Optional[MetricsCollector] = None
        self.robustness: Optional[RobustnessManager] = None
        self.session_manager: Optional[Session] = None
        self.running = False

    async def initialize_core_systems(self) -> bool:
        """Initialize core Kalki systems in dependency order"""
        try:
            logger.info("🔧 Initializing Kalki core systems...")

            # 1. EventBus (foundation for all communication)
            self.eventbus = EventBus()
            logger.info("✅ EventBus initialized")

            # 2. Metrics collection
            self.metrics = MetricsCollector()
            await self.metrics.initialize()
            logger.info("✅ Metrics collector initialized")

            # 3. Session management
            self.session_manager = Session.load_or_create()
            await self.session_manager.initialize()
            logger.info("✅ Session manager initialized")

            # 4. Vector database
            self.vector_db = VectorDBManager()
            logger.info("✅ Vector database initialized")

            # 5. LLM engine
            self.llm_engine = get_llm_engine()
            if not self.llm_engine:
                logger.warning("⚠️ LLM initialization failed, continuing with limited functionality")
            else:
                logger.info("✅ LLM engine initialized")

            # 6. Agent manager
            self.agent_manager = AgentManager(self.eventbus)
            await self.agent_manager.initialize()
            logger.info("✅ Agent manager initialized")

            # 7. Robustness monitoring
            self.robustness = RobustnessManager(self.eventbus, self.metrics)
            await self.robustness.initialize()
            logger.info("✅ Robustness monitoring initialized")

            # Publish system ready event
            await self.eventbus.publish_async("system.orchestrator.ready", {
                "version": "2.4.0",
                "components": ["eventbus", "metrics", "vectordb", "llm", "agents", "robustness"]
            })

            logger.info("🎉 All core systems initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize core systems: {e}")
            await self.shutdown()
            return False

    async def start_ingestion_pipeline(self) -> bool:
        """Start document ingestion and processing pipeline"""
        try:
            if not self.agent_manager:
                logger.error("Agent manager not initialized")
                return False

            # Register ingestion agents
            from modules.agents.core.ingestion_agent import IngestionAgent
            from modules.agents.core.chunking_agent import ChunkingAgent

            ingestion_agent = IngestionAgent(self.vector_db, self.eventbus)
            chunking_agent = ChunkingAgent(self.vector_db, self.eventbus)

            await self.agent_manager.register_agent(ingestion_agent)
            await self.agent_manager.register_agent(chunking_agent)

            logger.info("✅ Ingestion pipeline started")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start ingestion pipeline: {e}")
            return False

    async def start_ai_agents(self) -> bool:
        """Start AI agent orchestration"""
        try:
            if not self.agent_manager:
                logger.error("Agent manager not initialized")
                return False

            # Register core AI agents
            from modules.agents.core.llm_agent import LLMAgent
            from modules.agents.core.reasoning_agent import ReasoningAgent
            from modules.agents.core.memory_agent import MemoryAgent

            llm_agent = LLMAgent(self.eventbus)
            reasoning_agent = ReasoningAgent(self.eventbus)
            memory_agent = MemoryAgent(self.eventbus, self.vector_db)

            await self.agent_manager.register_agent(llm_agent)
            await self.agent_manager.register_agent(reasoning_agent)
            await self.agent_manager.register_agent(memory_agent)

            logger.info("✅ AI agents started")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start AI agents: {e}")
            return False

    async def start_gui_interface(self) -> bool:
        """Start graphical user interface"""
        try:
            from modules.gui import KalkiGUI
            gui = KalkiGUI(self.eventbus, self.agent_manager)
            await gui.initialize()

            logger.info("✅ GUI interface started")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start GUI: {e}")
            return False

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run comprehensive system health checks"""
        health_status = {
            "overall": "unknown",
            "components": {},
            "timestamp": asyncio.get_event_loop().time()
        }

        checks = [
            ("eventbus", lambda: self.eventbus is not None),
            ("agent_manager", lambda: self.agent_manager is not None),
            ("vector_db", lambda: self.vector_db is not None),
            ("metrics", lambda: self.metrics is not None),
            ("robustness", lambda: self.robustness is not None),
        ]

        all_healthy = True
        for component_name, check_func in checks:
            try:
                healthy = check_func()
                health_status["components"][component_name] = "healthy" if healthy else "unhealthy"
                if not healthy:
                    all_healthy = False
            except Exception as e:
                health_status["components"][component_name] = f"error: {e}"
                all_healthy = False

        health_status["overall"] = "healthy" if all_healthy else "degraded"
        return health_status

    async def run_main_loop(self):
        """Main orchestration loop"""
        self.running = True
        logger.info("🚀 Kalki orchestrator entering main loop")

        while self.running:
            try:
                # Periodic health checks
                if int(asyncio.get_event_loop().time()) % 60 == 0:  # Every minute
                    health = await self.run_health_checks()
                    await self.eventbus.publish_async("system.health_check", health)

                # Process agent tasks
                if self.agent_manager:
                    await self.agent_manager.process_pending_tasks()

                # Small delay to prevent busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Back off on errors

    async def shutdown(self):
        """Graceful shutdown of all systems"""
        logger.info("🛑 Initiating Kalki orchestrator shutdown...")

        shutdown_order = [
            ("robustness", self.robustness),
            ("agent_manager", self.agent_manager),
            ("metrics", self.metrics),
            ("vector_db", self.vector_db),
            ("eventbus", self.eventbus),
        ]

        for component_name, component in shutdown_order:
            if component:
                try:
                    if hasattr(component, 'shutdown'):
                        await component.shutdown()
                    elif hasattr(component, 'cleanup'):
                        await component.cleanup()
                    logger.info(f"✅ {component_name} shut down")
                except Exception as e:
                    logger.error(f"❌ Error shutting down {component_name}: {e}")

        self.running = False
        logger.info("🎯 Kalki orchestrator shutdown complete")

    def handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())


async def main():
    """Main entrypoint with CLI argument parsing"""
    parser = argparse.ArgumentParser(description="Kalki v2.4 Production Orchestrator")
    parser.add_argument("--start-all", action="store_true",
                       help="Start all systems (ingestion, agents, GUI)")
    parser.add_argument("--ingest-only", action="store_true",
                       help="Start only ingestion pipeline")
    parser.add_argument("--agents-only", action="store_true",
                       help="Start only AI agents")
    parser.add_argument("--gui-only", action="store_true",
                       help="Start only GUI interface")
    parser.add_argument("--health-check", action="store_true",
                       help="Run health check and exit")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level=log_level)

    orchestrator = KalkiOrchestrator()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, orchestrator.handle_shutdown_signal)
    signal.signal(signal.SIGTERM, orchestrator.handle_shutdown_signal)

    try:
        # Initialize core systems
        if not await orchestrator.initialize_core_systems():
            logger.error("Failed to initialize core systems")
            return 1

        # Handle different run modes
        if args.health_check:
            health = await orchestrator.run_health_checks()
            print(f"System Health: {health['overall']}")
            for component, status in health['components'].items():
                print(f"  {component}: {status}")
            return 0

        elif args.ingest_only:
            success = await orchestrator.start_ingestion_pipeline()
            if success:
                logger.info("Ingestion pipeline running. Press Ctrl+C to stop.")
                await orchestrator.run_main_loop()
            else:
                return 1

        elif args.agents_only:
            success = await orchestrator.start_ai_agents()
            if success:
                logger.info("AI agents running. Press Ctrl+C to stop.")
                await orchestrator.run_main_loop()
            else:
                return 1

        elif args.gui_only:
            success = await orchestrator.start_gui_interface()
            if success:
                logger.info("GUI interface running. Press Ctrl+C to stop.")
                await orchestrator.run_main_loop()
            else:
                return 1

        elif args.start_all:
            # Start all components
            ingest_ok = await orchestrator.start_ingestion_pipeline()
            agents_ok = await orchestrator.start_ai_agents()
            gui_ok = await orchestrator.start_gui_interface()

            if ingest_ok and agents_ok and gui_ok:
                logger.info("🎉 All Kalki systems running! Press Ctrl+C to stop.")
                await orchestrator.run_main_loop()
            else:
                logger.error("Some systems failed to start")
                return 1

        else:
            parser.print_help()
            return 0

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    finally:
        await orchestrator.shutdown()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)