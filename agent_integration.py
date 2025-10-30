#!/usr/bin/env python3
"""
Kalki Agent Integration
Integrates Phase 4-12 agents with the existing Kalki RAG system
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from modules.agents.phase4_memory import SessionAgent, MemoryAgent
from modules.agents.phase5_reasoning import PlannerAgent, OrchestratorAgent, ComputeOptimizerAgent
from modules.agents.phase12_safety import EthicsAgent, RiskAssessmentAgent
from modules.llm import ask_kalki
from modules.ingest import ingest_pdf_file
from modules.config import ROOT_DIR

logger = logging.getLogger("kalki.integration")


class KalkiAgentSystem:
    """
    Integrated agent system that combines RAG with multi-agent architecture
    """
    
    def __init__(self):
        """Initialize the agent system"""
        self.session_agent = SessionAgent()
        self.memory_agent = MemoryAgent()
        self.planner = PlannerAgent()
        self.orchestrator = OrchestratorAgent()
        self.compute_optimizer = ComputeOptimizerAgent()
        self.ethics_agent = EthicsAgent()
        self.risk_agent = RiskAssessmentAgent()
        
        # Initialize agents
        self.session_agent.initialize()
        self.memory_agent.initialize()
        
        logger.info("Kalki Agent System initialized")
    
    def create_user_session(self, user_id: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a new user session"""
        result = self.session_agent.execute({
            "action": "create",
            "user_id": user_id,
            "metadata": context or {}
        })
        return result["session_id"]
    
    def intelligent_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Process query with full agent orchestration:
        1. Plan the query execution
        2. Check ethics and safety
        3. Execute RAG query
        4. Store in memory
        5. Update session
        """
        try:
            # Step 1: Create execution plan
            plan_result = self.planner.execute({
                "action": "create_plan",
                "task": f"Answer query: {query}",
                "context": {"session_id": session_id}
            })
            plan = plan_result["plan"]
            
            # Step 2: Ethics check
            ethics_result = self.ethics_agent.execute({
                "action": "evaluate",
                "action_to_evaluate": {
                    "type": "query_processing",
                    "query": query,
                    "accesses_data": True
                },
                "context": {"session_id": session_id}
            })
            
            if ethics_result["evaluation"]["recommendation"] != "approve":
                logger.warning(f"Query blocked by ethics check: {query}")
                return {
                    "status": "blocked",
                    "reason": "Ethics validation failed",
                    "issues": ethics_result["evaluation"]["issues"]
                }
            
            # Step 3: Risk assessment
            risk_result = self.risk_agent.execute({
                "action": "assess",
                "action_to_assess": {
                    "type": "query_processing",
                    "query": query
                }
            })
            
            # Step 4: Execute RAG query
            answer = ask_kalki(query)
            
            # Step 5: Store in memory
            self.memory_agent.execute({
                "action": "store",
                "type": "episodic",
                "event": {
                    "summary": f"Query: {query[:100]}",
                    "details": f"Answer: {answer[:200]}",
                    "session_id": session_id
                }
            })
            
            # Step 6: Update session
            self.session_agent.execute({
                "action": "update",
                "session_id": session_id,
                "context_update": {
                    "query": query,
                    "answer_preview": answer[:100]
                }
            })
            
            return {
                "status": "success",
                "answer": answer,
                "plan": plan,
                "ethics_score": ethics_result["evaluation"]["ethical_score"],
                "risk_level": risk_result["assessment"]["severity"]
            }
            
        except Exception as e:
            logger.exception(f"Intelligent query failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def intelligent_ingest(self, pdf_path: Path, session_id: str, domain: str = "general") -> Dict[str, Any]:
        """
        Ingest PDF with full agent orchestration:
        1. Plan ingestion
        2. Ethics and safety check
        3. Allocate compute resources
        4. Execute ingestion
        5. Store in memory
        """
        try:
            # Step 1: Create execution plan
            plan_result = self.planner.execute({
                "action": "create_plan",
                "task": f"Ingest PDF: {pdf_path.name}",
                "context": {"domain": domain}
            })
            
            # Step 2: Ethics check
            ethics_result = self.ethics_agent.execute({
                "action": "evaluate",
                "action_to_evaluate": {
                    "type": "data_ingestion",
                    "file": str(pdf_path),
                    "domain": domain
                }
            })
            
            if ethics_result["evaluation"]["recommendation"] != "approve":
                return {
                    "status": "blocked",
                    "reason": "Ethics validation failed"
                }
            
            # Step 3: Allocate resources
            compute_result = self.compute_optimizer.execute({
                "action": "allocate",
                "task_id": f"ingest_{pdf_path.stem}",
                "requirements": {
                    "cpu_cores": 2,
                    "memory_gb": 2,
                    "priority": "normal"
                }
            })
            
            # Step 4: Execute ingestion
            success = ingest_pdf_file(pdf_path, domain=domain)
            
            # Step 5: Release resources
            self.compute_optimizer.execute({
                "action": "release",
                "task_id": f"ingest_{pdf_path.stem}"
            })
            
            # Step 6: Store in memory
            if success:
                self.memory_agent.execute({
                    "action": "store",
                    "type": "semantic",
                    "concept": f"ingested_{domain}",
                    "knowledge": {
                        "file": str(pdf_path),
                        "domain": domain,
                        "status": "success"
                    }
                })
            
            return {
                "status": "success" if success else "failed",
                "file": str(pdf_path),
                "domain": domain,
                "compute_allocation": compute_result["allocation"]
            }
            
        except Exception as e:
            logger.exception(f"Intelligent ingest failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        session = self.session_agent.execute({
            "action": "get",
            "session_id": session_id
        })
        
        # Get recent memories for this session
        memories = self.memory_agent.execute({
            "action": "recall",
            "type": "episodic",
            "limit": 10
        })
        
        session_memories = [
            m for m in memories.get("memories", [])
            if m.get("event", {}).get("session_id") == session_id
        ]
        
        return {
            "session": session.get("data"),
            "memories_count": len(session_memories),
            "recent_activities": session_memories[:5]
        }


def demo_integration():
    """Demonstrate integrated agent system"""
    logger.info("=== Kalki Agent Integration Demo ===")
    
    # Initialize system
    system = KalkiAgentSystem()
    
    # Create session
    session_id = system.create_user_session(
        "demo_user",
        context={"project": "AI Research", "clearance": "standard"}
    )
    logger.info(f"Created session: {session_id}")
    
    # Intelligent query with full orchestration
    result = system.intelligent_query(
        "What are the key principles of AI safety?",
        session_id
    )
    
    if result["status"] == "success":
        logger.info(f"Query successful!")
        logger.info(f"  Ethics score: {result['ethics_score']:.2f}")
        logger.info(f"  Risk level: {result['risk_level']}")
        logger.info(f"  Answer preview: {result['answer'][:150]}...")
    else:
        logger.error(f"Query failed: {result.get('reason')}")
    
    # Get session summary
    summary = system.get_session_summary(session_id)
    logger.info(f"Session summary:")
    logger.info(f"  Activities: {summary['memories_count']}")
    logger.info(f"  Status: {summary['session']['state']}")
    
    logger.info("=== Demo Complete ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo_integration()
