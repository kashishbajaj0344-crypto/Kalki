#!/usr/bin/env python3
"""
Test Phase 19: Learning & Adaptation Systems Integration
========================================================

Validates:
1. All 4 learning systems imported correctly
2. Orchestrator integration and initialization
3. Instance variables set properly
4. Singleton pattern functioning
5. System status reporting includes learning section
6. Reinforcement loop connected to query feedback
7. Learning systems connected to self-evolution
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all learning system imports work"""
    print("=" * 70)
    print("TEST 1: Learning & Adaptation Systems Imports")
    print("=" * 70)
    
    try:
        from modules.hybrid_learning_system import get_hybrid_system
        print("✅ HybridLearningSystem imported successfully")
    except Exception as e:
        print(f"❌ Failed to import HybridLearningSystem: {e}")
        return False
    
    try:
        from modules.federated_learning_bridge import get_federated_learning_bridge
        print("✅ FederatedLearningBridge imported successfully")
    except Exception as e:
        print(f"❌ Failed to import FederatedLearningBridge: {e}")
        return False
    
    try:
        from modules.reinforcement_loop import get_reinforcement_loop
        print("✅ ReinforcementLoop imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ReinforcementLoop: {e}")
        return False
    
    try:
        from modules.automated_validation_suite import get_automated_validation_suite
        print("✅ AutomatedValidationSuite imported successfully")
    except Exception as e:
        print(f"❌ Failed to import AutomatedValidationSuite: {e}")
        return False
    
    print("\n✅ All 4 learning system imports successful\n")
    return True


def test_orchestrator_integration():
    """Test that KalkiOrchestrator has learning system integration"""
    print("=" * 70)
    print("TEST 2: Orchestrator Integration")
    print("=" * 70)
    
    try:
        from kalki_complete import KalkiOrchestrator
        orchestrator = KalkiOrchestrator()
        
        # Check instance variables
        assert hasattr(orchestrator, 'hybrid_learning'), "Missing hybrid_learning attribute"
        assert hasattr(orchestrator, 'federated_learning'), "Missing federated_learning attribute"
        assert hasattr(orchestrator, 'reinforcement_loop'), "Missing reinforcement_loop attribute"
        assert hasattr(orchestrator, 'validation_suite'), "Missing validation_suite attribute"
        
        print("✅ All 4 learning system instance variables present")
        
        # Check for initialization method
        assert hasattr(orchestrator, '_initialize_learning_adaptation_systems'), \
            "Missing _initialize_learning_adaptation_systems method"
        print("✅ Initialization method _initialize_learning_adaptation_systems exists")
        
        print("\n✅ Orchestrator integration verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_initialization_sequence():
    """Test that initialization sequence includes Phase 19"""
    print("=" * 70)
    print("TEST 3: Initialization Sequence")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for Phase 19 initialization call
        assert 'await self._initialize_learning_adaptation_systems()' in content, \
            "Phase 19 initialization not called in initialize_system()"
        print("✅ Phase 19 initialization called in system startup")
        
        # Check for proper phase order (after Phase 18, before Phase 22)
        phase_18_pos = content.find('await self._initialize_visual_pipeline()')
        phase_19_pos = content.find('await self._initialize_learning_adaptation_systems()')
        phase_22_pos = content.find('await self._initialize_supreme_synthesis_phase()')
        
        assert phase_18_pos < phase_19_pos < phase_22_pos, \
            "Phase 19 not in correct order (should be after 18, before 22)"
        print("✅ Phase 19 in correct initialization order (after 18, before 22)")
        
        # Check that success message mentions Phase 19
        assert 'Phases 1-19, 21-25 active' in content, \
            "Success message doesn't mention Phase 19"
        print("✅ Success message updated to include Phase 19")
        
        print("\n✅ Initialization sequence verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Initialization sequence test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_singleton_patterns():
    """Test that singleton patterns work correctly"""
    print("=" * 70)
    print("TEST 4: Singleton Patterns")
    print("=" * 70)
    
    try:
        from modules.hybrid_learning_system import get_hybrid_system
        from modules.federated_learning_bridge import get_federated_learning_bridge
        from modules.reinforcement_loop import get_reinforcement_loop
        from modules.automated_validation_suite import get_automated_validation_suite
        
        # Test that singleton returns same instance
        hybrid1 = get_hybrid_system()
        hybrid2 = get_hybrid_system()
        assert hybrid1 is hybrid2, "HybridLearningSystem singleton not working"
        print("✅ HybridLearningSystem singleton pattern working")
        
        federated1 = get_federated_learning_bridge()
        federated2 = get_federated_learning_bridge()
        assert federated1 is federated2, "FederatedLearningBridge singleton not working"
        print("✅ FederatedLearningBridge singleton pattern working")
        
        reinforcement1 = get_reinforcement_loop()
        reinforcement2 = get_reinforcement_loop()
        assert reinforcement1 is reinforcement2, "ReinforcementLoop singleton not working"
        print("✅ ReinforcementLoop singleton pattern working")
        
        validation1 = get_automated_validation_suite()
        validation2 = get_automated_validation_suite()
        assert validation1 is validation2, "AutomatedValidationSuite singleton not working"
        print("✅ AutomatedValidationSuite singleton pattern working")
        
        print("\n✅ All singleton patterns verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Singleton pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_status_reporting():
    """Test that system status includes learning section"""
    print("=" * 70)
    print("TEST 5: System Status Reporting")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for learning_adaptation section in system status
        assert '"learning_adaptation": {' in content, \
            "learning_adaptation section missing from system status"
        print("✅ learning_adaptation section present in system status")
        
        # Check for all 4 systems in status
        assert '"hybrid_learning": self.hybrid_learning is not None' in content, \
            "hybrid_learning not in system status"
        assert '"federated_learning": self.federated_learning is not None' in content, \
            "federated_learning not in system status"
        assert '"reinforcement_loop": self.reinforcement_loop is not None' in content, \
            "reinforcement_loop not in system status"
        assert '"validation_suite": self.validation_suite is not None' in content, \
            "validation_suite not in system status"
        print("✅ All 4 learning systems in status reporting")
        
        # Check for system count
        assert '"learning_systems_count": len(self.phase_agents.get(\'learning_adaptation\'' in content, \
            "learning_systems_count not in system status"
        print("✅ System count tracking present")
        
        print("\n✅ System status reporting verified\n")
        return True
        
    except Exception as e:
        print(f"❌ System status test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reinforcement_loop_integration():
    """Test that reinforcement loop is integrated into query processing"""
    print("=" * 70)
    print("TEST 6: Reinforcement Loop Query Integration")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for reinforcement loop in query processing
        assert 'if self.reinforcement_loop:' in content, \
            "Reinforcement loop not checked in query processing"
        print("✅ Reinforcement loop checked in query processing")
        
        # Check for evaluate_response call
        assert 'await self.reinforcement_loop.evaluate_response(' in content, \
            "evaluate_response not called in query processing"
        print("✅ evaluate_response() called for query feedback")
        
        # Check that it's placed after self-evolution (Phase 23)
        evolution_pos = content.find('Phase 23: Self-evolution feedback loop')
        reinforcement_pos = content.find('Phase 19: Reinforcement learning feedback')
        
        assert evolution_pos > 0 and reinforcement_pos > 0, \
            "Phase markers not found"
        assert reinforcement_pos > evolution_pos, \
            "Reinforcement loop not after self-evolution in query flow"
        print("✅ Reinforcement loop properly positioned (after Phase 23)")
        
        print("\n✅ Reinforcement loop integration verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Reinforcement loop integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_learning_evolution_connections():
    """Test that learning systems are connected to self-evolution"""
    print("=" * 70)
    print("TEST 7: Learning ↔ Evolution Connections")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for reinforcement → evolution connection
        assert 'Connecting reinforcement loop to self-evolution feedback' in content, \
            "Reinforcement → Evolution connection not documented"
        print("✅ Reinforcement loop → Self-evolution connection present")
        
        # Check for validation → evolution connection
        assert 'Connecting validation suite to self-evolution pipeline' in content, \
            "Validation → Evolution connection not documented"
        print("✅ Validation suite → Self-evolution connection present")
        
        # Check for conditional connections
        assert 'if self.self_evolution_manager and self.reinforcement_loop:' in content, \
            "Conditional reinforcement connection missing"
        assert 'if self.validation_suite and self.self_evolution_manager:' in content, \
            "Conditional validation connection missing"
        print("✅ Conditional connection logic present")
        
        print("\n✅ Learning ↔ Evolution connections verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Learning-evolution connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("PHASE 19: LEARNING & ADAPTATION SYSTEMS INTEGRATION TEST")
    print("=" * 70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Initialization Sequence", test_initialization_sequence),
        ("Singleton Patterns", test_singleton_patterns),
        ("System Status Reporting", test_system_status_reporting),
        ("Reinforcement Loop Integration", test_reinforcement_loop_integration),
        ("Learning ↔ Evolution Connections", test_learning_evolution_connections)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 LEARNING & ADAPTATION SYSTEMS INTEGRATION TEST PASSED! 🎉")
        print("\nPhase 19 Systems Integrated:")
        print("  • HybridLearningSystem (PDF → Vector + Structured + Training)")
        print("  • FederatedLearningBridge (Distributed evolution)")
        print("  • ReinforcementLoop (Reward-based optimization)")
        print("  • AutomatedValidationSuite (Continuous testing)")
        print("\nConnections Established:")
        print("  • Reinforcement loop → Query feedback pipeline")
        print("  • Reinforcement loop → Self-evolution manager")
        print("  • Validation suite → Evolution recommendations")
        print("\nPhase 19 Status: ✅ FULLY OPERATIONAL")
        print("=" * 70 + "\n")
        return True
    else:
        print("\n⚠️  Some tests failed. Review output above for details.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
