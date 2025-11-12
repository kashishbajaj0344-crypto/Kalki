"""
Quick Test for TIER 1 Integration
Simplified test to verify all components are working
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')


async def main():
    print("\n" + "="*80)
    print("🚀 KALKI TIER 1 QUICK TEST")
    print("="*80 + "\n")
    
    # Test 1: Supreme Control Hub
    print("1️⃣ Supreme Control Hub...")
    from modules.supreme_control_hub import get_supreme_control_hub
    hub = get_supreme_control_hub()
    result = await hub.process_supreme_task(
        task="Design a simple bracket",
        mode="supreme"
    )
    print(f"   ✅ Quality: {result.design_quality:.1%}, Knowledge: {result.knowledge_integrated}")
    
    # Test 2: Multi-Modal Validator
    print("\n2️⃣ Multi-Modal Validator...")
    from modules.multimodal_validator import get_multimodal_validator
    from modules.design_brain import DesignBlueprint
    validator = get_multimodal_validator()
    
    blueprint = DesignBlueprint(
        id="test",
        description="test design",
        domain="mechanical",
        requirements=[],
        components={},
        materials=[],
        specifications={},
        constraints=[],
        ethical_considerations=[],
        environmental_impact={},
        generated_artifacts={}
    )
    
    report = await validator.validate_design(blueprint, validation_types=['structural'])
    print(f"   ✅ Validation Score: {report.overall_score:.1%}")
    
    # Test 3: Conscious Decision Engine  
    print("\n3️⃣ Conscious Decision Engine...")
    from modules.conscious_decision_engine import get_conscious_decision_engine, DecisionOption
    engine = get_conscious_decision_engine()
    
    options = [
        DecisionOption(id='a', description='Option A', base_score=0.7),
        DecisionOption(id='b', description='Option B', base_score=0.9)
    ]
    
    decision = await engine.make_decision(options, context={})
    print(f"   ✅ Selected: {decision.selected_option.description}, Score: {decision.final_score:.1%}")
    
    # Test 4: Quantum Optimizer
    print("\n4️⃣ Quantum Design Optimizer...")
    from modules.quantum_design_optimizer import get_quantum_optimizer
    optimizer = get_quantum_optimizer()
    
    result = await optimizer.optimize_design(
        parameters={'x': 1.0, 'y': 2.0},
        parameter_ranges={'x': (0, 10), 'y': (0, 10)},
        constraints={},
        max_iterations=10
    )
    print(f"   ✅ Optimized: Score {result['final_score']:.1%}")
    
    # Test 5: Evolution Loop
    print("\n5️⃣ Autonomous Evolution...")
    from modules.autonomous_evolution_loop import get_evolution_loop
    evolution = get_evolution_loop()
    await evolution.initialize()
    status = evolution.get_evolution_status()
    print(f"   ✅ Performance: {status['current_performance']:.1%}")
    
    # Test 6: Telemetry
    print("\n6️⃣ Real-World Telemetry...")
    from modules.realworld_telemetry_integration import get_telemetry_integration
    telemetry = get_telemetry_integration()
    await telemetry.initialize()
    status = telemetry.get_telemetry_status()
    print(f"   ✅ Deployed Designs: {status['deployed_designs']}")
    
    print("\n" + "="*80)
    print("🎉 ALL TIER 1 COMPONENTS OPERATIONAL!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
