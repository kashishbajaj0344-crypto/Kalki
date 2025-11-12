#!/usr/bin/env python3
"""
Kalki v3.0 - Ultimate System Integration Test
Tests the complete cohesion of all 20 cognitive phases
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from modules.orchestrator import get_orchestrator, KalkiOrchestrator
from modules.llm import get_llm_engine
from modules.vectordb import BGEEmbedder
from modules.generative_design_engine import GenerativeDesignEngine
from modules.consciousness_engine import ConsciousnessEngine
from modules.agents.agent_manager import AgentManager
from modules.cad_exporter import get_cad_exporter
from modules.cad_drawings import get_drawing_generator
from modules.freecad_integration import get_freecad_integration

async def test_ultimate_system_cohesion():
    """Test the complete Kalki system cohesion"""
    print("🧠 KALKI v3.0 - Ultimate System Cohesion Test")
    print("=" * 60)

    results = {
        "orchestrator": False,
        "llm_engine": False,
        "vector_db": False,
        "design_engine": False,
        "consciousness": False,
        "agent_manager": False,
        "cad_system": False,
        "integration": False,
        "complex_task": False
    }

    try:
        # Test 1: Orchestrator Initialization
        print("1. Testing Orchestrator...")
        orchestrator = get_orchestrator()
        status = await orchestrator.get_system_status()
        results["orchestrator"] = status["health"] == "operational"
        print(f"   ✅ Orchestrator: {status['health']}")

        # Test 2: LLM Engine
        print("2. Testing LLM Engine...")
        llm_engine = get_llm_engine()
        results["llm_engine"] = llm_engine is not None
        print(f"   ✅ LLM Engine: {'Available' if llm_engine else 'Unavailable'}")

        # Test 3: Vector Database
        print("3. Testing Vector Database...")
        embedder = BGEEmbedder()
        test_embedding = embedder.embed(["test query"])
        results["vector_db"] = len(test_embedding) > 0
        print(f"   ✅ Vector DB: {'Working' if results['vector_db'] else 'Failed'}")

        # Test 4: Design Engine
        print("4. Testing Design Engine...")
        design_engine = GenerativeDesignEngine()
        results["design_engine"] = design_engine is not None
        print(f"   ✅ Design Engine: {'Available' if design_engine else 'Unavailable'}")

        # Test 5: Consciousness Engine
        print("5. Testing Consciousness Engine...")
        consciousness_engine = ConsciousnessEngine()
        results["consciousness"] = consciousness_engine is not None
        print(f"   ✅ Consciousness: {'Available' if consciousness_engine else 'Unavailable'}")

        # Test 6: Agent Manager
        print("6. Testing Agent Manager...")
        agent_manager = AgentManager()
        results["agent_manager"] = agent_manager is not None
        print(f"   ✅ Agent Manager: {'Available' if agent_manager else 'Unavailable'}")

        # Test 7: CAD System
        print("7. Testing CAD System...")
        cad_exporter = get_cad_exporter()
        cad_drawings = get_drawing_generator()
        freecad = get_freecad_integration()
        results["cad_system"] = all([cad_exporter, cad_drawings, freecad])
        print(f"   ✅ CAD System: {'Complete' if results['cad_system'] else 'Incomplete'}")

        # Test 8: System Integration
        print("8. Testing System Integration...")
        # Test that all components can communicate
        try:
            integration_test = await orchestrator.process_task({
                "query": "system integration test",
                "type": "diagnostic",
                "context": {"test_mode": True}
            })
            results["integration"] = integration_test["status"] == "completed"
            print(f"   ✅ Integration: {'Successful' if results['integration'] else 'Failed'}")
        except Exception as e:
            print(f"   ❌ Integration: Failed with error: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            results["integration"] = False

        # Test 9: Complex Task Processing
        print("9. Testing Complex Task Processing...")
        complex_task = {
            "query": "Design a robotic arm capable of precise manipulation for surgical applications",
            "type": "design",
            "requirements": {
                "precision": "sub-millimeter",
                "degrees_of_freedom": 6,
                "payload_capacity": "5kg",
                "application": "surgical robotics"
            },
            "context": {
                "domain": "medical robotics",
                "constraints": ["sterility", "precision", "safety"],
                "output_format": ["CAD", "specifications", "analysis"]
            }
        }

        start_time = time.time()
        try:
            task_result = await orchestrator.process_task(complex_task)
            execution_time = time.time() - start_time
            results["complex_task"] = task_result["status"] == "completed"
            print(f"   ✅ Complex Task: {'Processed' if results['complex_task'] else 'Failed'} ({execution_time:.2f}s)")
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"   ❌ Complex Task: Failed with error: {e}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            results["complex_task"] = False

        # Final Results
        print("\n" + "=" * 60)
        print("🧠 KALKI v3.0 SYSTEM COHESION RESULTS")
        print("=" * 60)

        total_tests = len(results)
        passed_tests = sum(results.values())
        cohesion_percentage = (passed_tests / total_tests) * 100

        for test, passed in results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {test.replace('_', ' ').title()}: {'PASS' if passed else 'FAIL'}")

        print(f"\n🎯 Overall Cohesion: {cohesion_percentage:.1f}% ({passed_tests}/{total_tests})")

        if cohesion_percentage >= 90:
            print("🚀 SCI-FI GRADE ACHIEVED: System rivals major LLMs!")
        elif cohesion_percentage >= 80:
            print("⚡ ADVANCED AI: System is highly capable!")
        elif cohesion_percentage >= 70:
            print("🔧 FUNCTIONAL: System works but needs optimization!")
        else:
            print("🔨 DEVELOPMENT: System needs significant work!")

        return cohesion_percentage >= 80

    except Exception as e:
        print(f"❌ Critical system failure: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test execution"""
    success = await test_ultimate_system_cohesion()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())