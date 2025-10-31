#!/usr/bin/env python3
"""
Kalki v2.4 — Production Readiness Validation Script
Validates that all production infrastructure components are properly configured and functional.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def validate_imports():
    """Validate that all required modules can be imported."""
    print("🔍 Validating imports...")

    try:
        # Core orchestrator
        from kalki_orchestrator import KalkiOrchestrator
        print("✅ Orchestrator import successful")

        # Core modules
        from modules.config import CONFIG, get_config, DIRS
        print("✅ Config module import successful")

        from modules.eventbus import EventBus
        print("✅ EventBus import successful")

        from modules.agents.agent_manager import AgentManager
        print("✅ AgentManager import successful")

        from modules.llm import get_llm_engine
        print("✅ LLM module import successful")

        from modules.vectordb import VectorDBManager
        print("✅ VectorDB import successful")

        from modules.metrics.collector import MetricsCollector
        print("✅ MetricsCollector import successful")

        from modules.robustness import RobustnessManager
        print("✅ RobustnessManager import successful")

        from modules.session import Session
        print("✅ Session import successful")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

async def validate_configuration():
    """Validate configuration loading."""
    print("\n🔍 Validating configuration...")

    try:
        from modules.config import CONFIG, DIRS

        # Check required directories exist
        required_dirs = ['data', 'logs', 'memory', 'pdfs', 'sessions']
        for dir_name in required_dirs:
            if dir_name in DIRS and Path(DIRS[dir_name]).exists():
                print(f"✅ Directory {dir_name} exists")
            else:
                print(f"⚠️ Directory {dir_name} missing or not configured")

        # Check configuration sections
        required_config_sections = ['llm', 'vectordb', 'agents']
        for section in required_config_sections:
            if section in CONFIG:
                print(f"✅ Config section '{section}' exists")
            else:
                print(f"⚠️ Config section '{section}' missing")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

async def validate_orchestrator():
    """Validate orchestrator can be instantiated."""
    print("\n🔍 Validating orchestrator instantiation...")

    try:
        from kalki_orchestrator import KalkiOrchestrator

        # Test orchestrator creation (without full initialization)
        orchestrator = KalkiOrchestrator()
        print("✅ Orchestrator instantiation successful")

        # Check required attributes exist
        required_attrs = ['config', 'eventbus', 'logger']
        for attr in required_attrs:
            if hasattr(orchestrator, attr):
                print(f"✅ Orchestrator has {attr} attribute")
            else:
                print(f"⚠️ Orchestrator missing {attr} attribute")

        return True

    except Exception as e:
        print(f"❌ Orchestrator validation failed: {e}")
        return False

async def validate_docker_setup():
    """Validate Docker configuration exists."""
    print("\n🔍 Validating Docker setup...")

    dockerfile_path = Path("Dockerfile")
    docker_compose_path = Path("docker-compose.yml")

    if dockerfile_path.exists():
        print("✅ Dockerfile exists")
    else:
        print("❌ Dockerfile missing")

    if docker_compose_path.exists():
        print("✅ docker-compose.yml exists")
    else:
        print("❌ docker-compose.yml missing")

    return dockerfile_path.exists() and docker_compose_path.exists()

async def validate_environment_template():
    """Validate environment template exists."""
    print("\n🔍 Validating environment configuration...")

    env_example_path = Path(".env.example")

    if env_example_path.exists():
        print("✅ .env.example exists")

        # Check for required API keys
        with open(env_example_path, 'r') as f:
            content = f.read()

        required_keys = ['OPENAI_API_KEY', 'HUGGINGFACE_API_KEY']
        for key in required_keys:
            if key in content:
                print(f"✅ {key} template present")
            else:
                print(f"⚠️ {key} template missing")

        return True
    else:
        print("❌ .env.example missing")
        return False

async def validate_documentation():
    """Validate documentation files exist."""
    print("\n🔍 Validating documentation...")

    docs = [
        "README.md",
        "PRODUCTION_DEPLOYMENT.md",
        "SECURITY_POLICY.md",
        "MONITORING_OBSERVABILITY.md",
        "TESTING_STRATEGY.md",
        "PRODUCTION_READINESS_SUMMARY.md"
    ]

    all_present = True
    for doc in docs:
        if Path(doc).exists():
            print(f"✅ {doc} exists")
        else:
            print(f"❌ {doc} missing")
            all_present = False

    return all_present

async def validate_ci_cd():
    """Validate CI/CD configuration."""
    print("\n🔍 Validating CI/CD setup...")

    github_workflows = Path(".github/workflows")
    if github_workflows.exists():
        ci_file = github_workflows / "ci.yml"
        if ci_file.exists():
            print("✅ CI workflow exists")
            return True
        else:
            print("❌ CI workflow missing")
            return False
    else:
        print("❌ .github/workflows directory missing")
        return False

async def main():
    """Run all validation checks."""
    print("🚀 Kalki v2.4 — Production Readiness Validation")
    print("=" * 50)

    validations = [
        ("Imports", validate_imports),
        ("Configuration", validate_configuration),
        ("Orchestrator", validate_orchestrator),
        ("Docker Setup", validate_docker_setup),
        ("Environment Template", validate_environment_template),
        ("Documentation", validate_documentation),
        ("CI/CD", validate_ci_cd),
    ]

    results = []
    for name, validator in validations:
        try:
            result = await validator()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if result:
            passed += 1

    print(f"\nOverall Score: {passed}/{total} validations passed")

    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED! Kalki v2.4 is production-ready.")
        return 0
    else:
        print("⚠️ Some validations failed. Review output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)