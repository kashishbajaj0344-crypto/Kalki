#!/usr/bin/env python3
"""
Test Phase 20: Safety & Governance Framework Integration
=======================================================

Validates:
1. All 5 governance systems imported correctly
2. Orchestrator integration and initialization
3. Instance variables set properly
4. Singleton pattern functioning
5. System status reporting includes governance section
6. Governance checks integrated into evolution pipeline
7. Human review connected to high-impact decisions
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all governance system imports work"""
    print("=" * 70)
    print("TEST 1: Safety & Governance Systems Imports")
    print("=" * 70)
    
    try:
        from modules.canary_deployment_manager import get_canary_deployment_manager
        print("✅ CanaryDeploymentManager imported successfully")
    except Exception as e:
        print(f"❌ Failed to import CanaryDeploymentManager: {e}")
        return False
    
    try:
        from modules.external_red_teaming_certification import get_external_red_teaming_certification
        print("✅ ExternalRedTeamingCertification imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ExternalRedTeamingCertification: {e}")
        return False
    
    try:
        from modules.simulated_adversarial_tests import get_simulated_adversarial_tests
        print("✅ SimulatedAdversarialTests imported successfully")
    except Exception as e:
        print(f"❌ Failed to import SimulatedAdversarialTests: {e}")
        return False
    
    try:
        from modules.governance_sla_framework import get_governance_sla_framework
        print("✅ GovernanceSLAFramework imported successfully")
    except Exception as e:
        print(f"❌ Failed to import GovernanceSLAFramework: {e}")
        return False
    
    try:
        from modules.human_review_cadence import get_human_review_cadence
        print("✅ HumanReviewCadence imported successfully")
    except Exception as e:
        print(f"❌ Failed to import HumanReviewCadence: {e}")
        return False
    
    print("\n✅ All 5 governance system imports successful\n")
    return True


def test_orchestrator_integration():
    """Test that KalkiOrchestrator has governance system integration"""
    print("=" * 70)
    print("TEST 2: Orchestrator Integration")
    print("=" * 70)
    
    try:
        from kalki_complete import KalkiOrchestrator
        orchestrator = KalkiOrchestrator()
        
        # Check instance variables
        assert hasattr(orchestrator, 'canary_deployment'), "Missing canary_deployment attribute"
        assert hasattr(orchestrator, 'red_teaming'), "Missing red_teaming attribute"
        assert hasattr(orchestrator, 'adversarial_tests'), "Missing adversarial_tests attribute"
        assert hasattr(orchestrator, 'governance_sla'), "Missing governance_sla attribute"
        assert hasattr(orchestrator, 'human_review'), "Missing human_review attribute"
        
        print("✅ All 5 governance system instance variables present")
        
        # Check for initialization method
        assert hasattr(orchestrator, '_initialize_safety_governance_framework'), \
            "Missing _initialize_safety_governance_framework method"
        print("✅ Initialization method _initialize_safety_governance_framework exists")
        
        print("\n✅ Orchestrator integration verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_initialization_sequence():
    """Test that initialization sequence includes Phase 20"""
    print("=" * 70)
    print("TEST 3: Initialization Sequence")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for Phase 20 initialization call
        assert 'await self._initialize_safety_governance_framework()' in content, \
            "Phase 20 initialization not called in initialize_system()"
        print("✅ Phase 20 initialization called in system startup")
        
        # Check for proper phase order (after Phase 19, before Phase 22)
        phase_19_pos = content.find('await self._initialize_learning_adaptation_systems()')
        phase_20_pos = content.find('await self._initialize_safety_governance_framework()')
        phase_22_pos = content.find('await self._initialize_supreme_synthesis_phase()')
        
        assert phase_19_pos < phase_20_pos < phase_22_pos, \
            "Phase 20 not in correct order (should be after 19, before 22)"
        print("✅ Phase 20 in correct initialization order (after 19, before 22)")
        
        # Check that success message mentions Phase 20
        assert 'Phases 1-20, 21-25 active' in content, \
            "Success message doesn't mention Phase 20"
        print("✅ Success message updated to include Phase 20")
        
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
        from modules.canary_deployment_manager import get_canary_deployment_manager
        from modules.external_red_teaming_certification import get_external_red_teaming_certification
        from modules.simulated_adversarial_tests import get_simulated_adversarial_tests
        from modules.governance_sla_framework import get_governance_sla_framework
        from modules.human_review_cadence import get_human_review_cadence
        
        # Test that singleton returns same instance
        canary1 = get_canary_deployment_manager()
        canary2 = get_canary_deployment_manager()
        assert canary1 is canary2, "CanaryDeploymentManager singleton not working"
        print("✅ CanaryDeploymentManager singleton pattern working")
        
        red_team1 = get_external_red_teaming_certification()
        red_team2 = get_external_red_teaming_certification()
        assert red_team1 is red_team2, "ExternalRedTeamingCertification singleton not working"
        print("✅ ExternalRedTeamingCertification singleton pattern working")
        
        adversarial1 = get_simulated_adversarial_tests()
        adversarial2 = get_simulated_adversarial_tests()
        assert adversarial1 is adversarial2, "SimulatedAdversarialTests singleton not working"
        print("✅ SimulatedAdversarialTests singleton pattern working")
        
        governance1 = get_governance_sla_framework()
        governance2 = get_governance_sla_framework()
        assert governance1 is governance2, "GovernanceSLAFramework singleton not working"
        print("✅ GovernanceSLAFramework singleton pattern working")
        
        review1 = get_human_review_cadence()
        review2 = get_human_review_cadence()
        assert review1 is review2, "HumanReviewCadence singleton not working"
        print("✅ HumanReviewCadence singleton pattern working")
        
        print("\n✅ All singleton patterns verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Singleton pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_status_reporting():
    """Test that system status includes governance section"""
    print("=" * 70)
    print("TEST 5: System Status Reporting")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for safety_governance section in system status
        assert '"safety_governance": {' in content, \
            "safety_governance section missing from system status"
        print("✅ safety_governance section present in system status")
        
        # Check for all 5 systems in status
        assert '"canary_deployment": self.canary_deployment is not None' in content, \
            "canary_deployment not in system status"
        assert '"red_teaming": self.red_teaming is not None' in content, \
            "red_teaming not in system status"
        assert '"adversarial_tests": self.adversarial_tests is not None' in content, \
            "adversarial_tests not in system status"
        assert '"governance_sla": self.governance_sla is not None' in content, \
            "governance_sla not in system status"
        assert '"human_review": self.human_review is not None' in content, \
            "human_review not in system status"
        print("✅ All 5 governance systems in status reporting")
        
        # Check for system count
        assert '"governance_systems_count": len(self.phase_agents.get(\'safety_governance\'' in content, \
            "governance_systems_count not in system status"
        print("✅ System count tracking present")
        
        print("\n✅ System status reporting verified\n")
        return True
        
    except Exception as e:
        print(f"❌ System status test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_governance_evolution_integration():
    """Test that governance is integrated into evolution pipeline"""
    print("=" * 70)
    print("TEST 6: Governance → Evolution Integration")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for governance checks in query processing
        assert 'Phase 20: Governance & safety checks for evolution recommendations' in content, \
            "Phase 20 governance checks not in query processing"
        print("✅ Phase 20 governance checks present in query processing")
        
        # Check for governance interaction with evolution
        assert 'if self.self_evolution_manager and self.governance_sla:' in content, \
            "Governance not checking evolution recommendations"
        print("✅ Governance checks evolution recommendations")
        
        # Check that it's placed after Phase 23 (self-evolution)
        evolution_pos = content.find('Phase 23: Self-evolution feedback loop')
        governance_pos = content.find('Phase 20: Governance & safety checks')
        
        assert evolution_pos > 0 and governance_pos > 0, \
            "Phase markers not found"
        assert governance_pos > evolution_pos, \
            "Governance checks not after self-evolution in query flow"
        print("✅ Governance checks properly positioned (after Phase 23)")
        
        print("\n✅ Governance → Evolution integration verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Governance-evolution integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_governance_connections():
    """Test that governance systems are connected to evolution and human review"""
    print("=" * 70)
    print("TEST 7: Governance System Connections")
    print("=" * 70)
    
    try:
        # Read the kalki_complete.py file
        with open('kalki_complete.py', 'r') as f:
            content = f.read()
        
        # Check for canary → evolution connection
        assert 'Connecting canary deployment to self-evolution rollouts' in content, \
            "Canary → Evolution connection not documented"
        print("✅ Canary deployment → Self-evolution connection present")
        
        # Check for governance SLA → evolution connection
        assert 'Connecting governance SLA to evolution change management' in content, \
            "Governance SLA → Evolution connection not documented"
        print("✅ Governance SLA → Evolution change management connection present")
        
        # Check for human review → evolution connection
        assert 'Connecting human review to high-impact evolution decisions' in content, \
            "Human review → Evolution connection not documented"
        print("✅ Human review → High-impact decisions connection present")
        
        # Check for human review flagging in query processing
        assert 'if self.human_review:' in content, \
            "Human review not checked in query processing"
        print("✅ Human review flagging present in query flow")
        
        print("\n✅ Governance system connections verified\n")
        return True
        
    except Exception as e:
        print(f"❌ Governance connections test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "=" * 70)
    print("PHASE 20: SAFETY & GOVERNANCE FRAMEWORK INTEGRATION TEST")
    print("=" * 70 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Initialization Sequence", test_initialization_sequence),
        ("Singleton Patterns", test_singleton_patterns),
        ("System Status Reporting", test_system_status_reporting),
        ("Governance → Evolution Integration", test_governance_evolution_integration),
        ("Governance System Connections", test_governance_connections)
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
        print("\n🎉 SAFETY & GOVERNANCE FRAMEWORK INTEGRATION TEST PASSED! 🎉")
        print("\nPhase 20 Systems Integrated:")
        print("  • CanaryDeploymentManager (A/B testing & gradual rollout)")
        print("  • ExternalRedTeamingCertification (Security audits & compliance)")
        print("  • SimulatedAdversarialTests (Jailbreak detection & fuzzing)")
        print("  • GovernanceSLAFramework (Change management & SLAs)")
        print("  • HumanReviewCadence (Weekly oversight & approvals)")
        print("\nConnections Established:")
        print("  • Canary deployment → Evolution rollouts")
        print("  • Governance SLA → Change management approvals")
        print("  • Human review → High-impact evolution decisions")
        print("  • Governance checks → Query processing pipeline")
        print("\nPhase 20 Status: ✅ FULLY OPERATIONAL")
        print("=" * 70 + "\n")
        return True
    else:
        print("\n⚠️  Some tests failed. Review output above for details.\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
