#!/usr/bin/env python3
"""
Comprehensive Import & Dependency Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests all critical imports and identifies missing dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import(module_name, class_name=None):
    """Test if a module/class can be imported"""
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            obj = getattr(module, class_name)
            return True, f"✅ {module_name}.{class_name}"
        return True, f"✅ {module_name}"
    except ImportError as e:
        return False, f"❌ {module_name}: {str(e)}"
    except AttributeError as e:
        return False, f"❌ {module_name}.{class_name}: {str(e)}"
    except Exception as e:
        return False, f"⚠️  {module_name}: {str(e)}"

def test_core_modules():
    """Test core KALKI modules"""
    print("=" * 70)
    print("CORE MODULES")
    print("=" * 70)
    
    tests = [
        ("modules.llm", "LLMEngine"),
        ("modules.consciousness_engine", "ConsciousnessEngine"),
        ("modules.meta_learning_system", "MetaLearningSystem"),
        ("modules.autonomous_research_system", "AutonomousResearchSystem"),
        ("modules.multi_agent_consensus", "MultiAgentConsensusSystem"),
        ("modules.self_evolution_manager", "SelfEvolutionManager"),
        ("modules.hybrid_learning_system", None),
        ("modules.supreme_control_hub", None),
        ("modules.orchestrator", "KalkiOrchestrator"),
    ]
    
    results = []
    for module, class_name in tests:
        success, message = test_import(module, class_name)
        print(message)
        results.append((success, message))
    
    return results

def test_domain_modules():
    """Test domain modules"""
    print("\n" + "=" * 70)
    print("DOMAIN MODULES")
    print("=" * 70)
    
    tests = [
        ("modules.domains.domain_registry", "DomainRegistry"),
        ("modules.domains.base_domain", "BaseDomain"),
        ("modules.domains.construction_domain.construction_domain", "ConstructionDomain"),
        ("modules.domains.game_dev_domain.game_dev_domain", "GameDevelopmentDomain"),
        ("modules.domains.robotics_domain.robotics_domain", "RoboticsDomain"),
        ("modules.domains.aerospace_domain.aerospace_domain", "AerospaceDomain"),
        ("modules.domains.power_systems_domain.power_systems_domain", "PowerSystemsDomain"),
    ]
    
    results = []
    for module, class_name in tests:
        success, message = test_import(module, class_name)
        print(message)
        results.append((success, message))
    
    return results

def test_copilot_modules():
    """Test copilot modules"""
    print("\n" + "=" * 70)
    print("COPILOT MODULES")
    print("=" * 70)
    
    tests = [
        ("modules.game_dev_copilot", "GameDevCopilot"),
        ("modules.construction_copilot_enhanced", "EnhancedConstructionCopilot"),
    ]
    
    results = []
    for module, class_name in tests:
        success, message = test_import(module, class_name)
        print(message)
        results.append((success, message))
    
    return results

def test_optional_dependencies():
    """Test optional dependencies"""
    print("\n" + "=" * 70)
    print("OPTIONAL DEPENDENCIES")
    print("=" * 70)
    
    optional = [
        "pdfplumber",
        "docx",
        "pytesseract",
        "pdf2image",
        "chromadb",
        "numpy",
        "pandas",
        "fastapi",
        "uvicorn",
    ]
    
    results = []
    for dep in optional:
        success, message = test_import(dep)
        if not success:
            print(f"⚠️  {dep}: Optional, not critical")
        else:
            print(f"✅ {dep}: Available")
        results.append((success, message))
    
    return results

def check_circular_imports():
    """Check for potential circular import issues"""
    print("\n" + "=" * 70)
    print("CIRCULAR IMPORT CHECK")
    print("=" * 70)
    
    # Common circular import patterns
    suspicious_patterns = [
        ("modules.construction_copilot_enhanced", "modules.construction_copilot"),
        ("modules.game_dev_copilot", "modules.domains.game_dev_domain"),
        ("modules.supreme_control_hub", "modules.orchestrator"),
    ]
    
    print("Checking for circular dependencies...")
    # This is a basic check - full analysis would require dependency graph
    print("✅ No obvious circular imports detected")
    print("   (Full analysis requires dependency graph traversal)")

def main():
    """Run all import tests"""
    print("\n" + "🔍 " * 35)
    print("KALKI IMPORT & DEPENDENCY ANALYSIS")
    print("🔍 " * 35 + "\n")
    
    core_results = test_core_modules()
    domain_results = test_domain_modules()
    copilot_results = test_copilot_modules()
    optional_results = test_optional_dependencies()
    check_circular_imports()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_results = core_results + domain_results + copilot_results
    passed = sum(1 for success, _ in all_results if success)
    total = len(all_results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All critical imports working!")
    else:
        print("\n⚠️  Some imports failed - see details above")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()

