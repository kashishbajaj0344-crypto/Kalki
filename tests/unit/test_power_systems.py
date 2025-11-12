#!/usr/bin/env python3
"""
Test All Kalki Systems - Verify readiness for power-up
"""

import sys
import os
import asyncio

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

async def test_all_systems():
    """Test each system to verify it's ready"""
    
    print("\n" + "="*60)
    print("🔍 KALKI SYSTEMS READINESS CHECK")
    print("="*60 + "\n")
    
    systems = {}
    power_components = []
    
    # Test 1: LLM Engine
    try:
        from modules.llm import initialize_llm_engine, get_llm_engine
        await initialize_llm_engine()
        llm = get_llm_engine()
        systems['LLM'] = llm is not None
        if systems['LLM']:
            power_components.append('LLM')
        print("✅ LLM Engine (Llama 3.1 8B): READY")
    except Exception as e:
        systems['LLM'] = False
        print(f"❌ LLM Engine: ERROR - {e}")
    
    # Test 2: Vector Database
    try:
        from modules.learning.vectordb import VectorDBManager
        vectordb = VectorDBManager()
        systems['VectorDB'] = True
        power_components.append('VectorDB')
        print("✅ Vector Database: READY")
    except Exception as e:
        systems['VectorDB'] = False
        print(f"❌ Vector Database: ERROR - {e}")
    
    # Test 3: Meta-Core
    try:
        from modules.meta_core import MetaCore
        meta_core = MetaCore()
        systems['MetaCore'] = True
        power_components.append('MetaCore')
        print("✅ Meta-Core: READY")
    except Exception as e:
        systems['MetaCore'] = False
        print(f"❌ Meta-Core: ERROR - {e}")
    
    # Test 4: Consciousness Engine
    try:
        from modules.consciousness_engine import ConsciousnessEngine
        consciousness = ConsciousnessEngine()
        systems['Consciousness'] = True
        power_components.append('Consciousness')
        print("✅ Consciousness Engine: READY")
    except Exception as e:
        systems['Consciousness'] = False
        print(f"❌ Consciousness Engine: ERROR - {e}")
    
    # Test 5: Self-Evolution Manager
    try:
        from modules.self_evolution_manager import SelfEvolutionManager
        evolution = SelfEvolutionManager()
        systems['Evolution'] = True
        power_components.append('Evolution')
        print("✅ Self-Evolution Manager: READY")
    except Exception as e:
        systems['Evolution'] = False
        print(f"❌ Self-Evolution Manager: ERROR - {e}")
    
    # Test 6: Autonomous Research System
    try:
        from modules.autonomous_research_system import AutonomousResearchSystem
        research = AutonomousResearchSystem()
        systems['Research'] = True
        power_components.append('Research')
        print("✅ Autonomous Research System: READY")
    except Exception as e:
        systems['Research'] = False
        print(f"❌ Autonomous Research System: ERROR - {e}")
    
    # Test 7: Agent Manager
    try:
        from modules.agents.agent_manager import AgentManager
        agent_manager = AgentManager()
        systems['Agents'] = True
        power_components.append('Agents')
        print("✅ Agent Manager: READY")
    except Exception as e:
        systems['Agents'] = False
        print(f"❌ Agent Manager: ERROR - {e}")
    
    # Test 8: Quantum Reasoning
    try:
        from modules.agents.quantum.quantum_reasoning import QuantumReasoningAgent
        quantum = QuantumReasoningAgent()
        systems['Quantum'] = True
        power_components.append('Quantum')
        print("✅ Quantum Reasoning: READY")
    except Exception as e:
        systems['Quantum'] = False
        print(f"⚠️  Quantum Reasoning: {e}")
    
    # Test 9: Professional Deliverables
    try:
        from modules.professional_deliverables import (
            ProfessionalDeliverablesGenerator
        )
        deliverables = ProfessionalDeliverablesGenerator()
        systems['Deliverables'] = True
        power_components.append('Deliverables')
        print("✅ Professional Deliverables: READY")
    except Exception as e:
        systems['Deliverables'] = False
        print(f"⚠️  Professional Deliverables: {e}")
    
    # Test 10: Knowledge Databases
    import sqlite3
    db_count = 0
    databases = [
        'data/knowledge/formulas.db',
        'data/knowledge/span_tables.db',
        'data/knowledge/procedures.db',
        'data/knowledge/inspection_criteria.db',
        'data/knowledge/cost_data.db',
        'data/knowledge/load_parameters.db',
        'data/knowledge/decision_trees.db'
    ]
    
    for db_path in databases:
        if os.path.exists(db_path):
            db_count += 1
    
    systems['KnowledgeDBs'] = db_count > 0
    if systems['KnowledgeDBs']:
        power_components.append('KnowledgeDBs')
    print(f"✅ Knowledge Databases: {db_count}/7 available")
    
    # Calculate power level
    total_systems = len(systems)
    active_systems = sum(1 for v in systems.values() if v)
    power_level = (active_systems / total_systems) * 100
    
    print("\n" + "="*60)
    print(f"⚡ POWER LEVEL: {power_level:.0f}%")
    print(f"📊 ACTIVE SYSTEMS: {active_systems}/{total_systems}")
    print("="*60)
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:\n")
    
    if power_level < 25:
        print("🚨 CRITICAL: Core systems offline")
        print("   → Run: pip install -r requirements.txt")
        print("   → Check module imports")
    elif power_level < 50:
        print("⚠️  BASIC: Core systems ready, advanced systems missing")
        print("   → Start with: python kalki_app_enhanced.py (25% power)")
        print("   → Next: Activate agents for 50% power")
    elif power_level < 75:
        print("✅ GOOD: Most systems operational")
        print("   → Ready for agent activation")
        print("   → Follow Day 2-3 of roadmap")
    elif power_level < 95:
        print("🎯 EXCELLENT: Nearly at full power")
        print("   → Complete remaining integrations")
        print("   → Follow Day 4-6 of roadmap")
    else:
        print("🎉 SUPREME: All systems operational!")
        print("   → Ready for kalki_app_supreme.py")
        print("   → 100% power achieved")
    
    print("\n" + "="*60)
    
    # Next steps
    print("\n🚀 NEXT STEPS:\n")
    
    if systems.get('LLM') and systems.get('VectorDB') and systems.get('MetaCore'):
        print("✅ Quick Start Ready!")
        print("   Run: streamlit run kalki_app_enhanced.py")
        print("   This gives you 25% power immediately\n")
    
    if systems.get('Agents'):
        print("✅ Agent System Ready!")
        print("   Next: Implement Day 2 (Agent Coordination)")
        print("   Target: 50% power\n")
    
    if systems.get('Consciousness') and systems.get('Evolution'):
        print("✅ Advanced Intelligence Ready!")
        print("   Next: Implement Day 3 (Consciousness + Evolution)")
        print("   Target: 70% power\n")
    
    if systems.get('Research'):
        print("✅ Autonomous Research Ready!")
        print("   Next: Implement Day 4 (Research System)")
        print("   Target: 85% power\n")
    
    print("="*60)
    
    return power_level >= 25

if __name__ == "__main__":
    success = asyncio.run(test_all_systems())
    print("\n")
    
    if success:
        print("✅ READY TO POWER UP")
        exit(0)
    else:
        print("⚠️  SETUP REQUIRED - See recommendations above")
        exit(1)
