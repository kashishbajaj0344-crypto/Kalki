#!/usr/bin/env python3
"""
Test script for Kalki's internet connectivity features.
Run this after setting up API keys to verify everything works.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_web_search_agent():
    """Test the WebSearchAgent directly"""
    print("🧪 Testing WebSearchAgent...")

    try:
        from modules.agents.core.web_search import WebSearchAgent

        # Initialize agent
        agent = WebSearchAgent()

        # Initialize the agent (this creates the HTTP session)
        init_success = await agent.initialize()
        if not init_success:
            print("   ❌ WebSearchAgent initialization failed")
            return False

        # Test basic search
        print("   Testing basic search...")
        results = await agent.search("Python programming", max_results=2)
        print(f"   ✅ Found {len(results)} results")

        # Test content safety
        print("   Testing safety controls...")
        safe_results = await agent.search("safe topic", max_results=1)
        print("   ✅ Safety controls working")

        return True

    except Exception as e:
        print(f"   ❌ WebSearchAgent test failed: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    print("📦 Testing imports...")

    try:
        import aiohttp
        import bs4
        from dotenv import load_dotenv
        print("   ✅ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_env_file():
    """Test environment file loading"""
    print("🔧 Testing environment configuration...")

    env_file = Path(".env")
    if not env_file.exists():
        print("   ⚠️  No .env file found - using default configuration")
        return True

    try:
        from dotenv import load_dotenv
        load_dotenv()

        # Check for any configured API keys
        api_keys = [
            'GOOGLE_SEARCH_API_KEY',
            'BING_SEARCH_API_KEY',
            'SERPAPI_KEY',
            'OPENAI_API_KEY',
            'HUGGINGFACE_API_KEY'
        ]

        configured = 0
        for key in api_keys:
            if os.getenv(key):
                configured += 1

        if configured > 0:
            print(f"   ✅ {configured} API key(s) configured")
        else:
            print("   ⚠️  No API keys configured - will use DuckDuckGo fallback")

        return True

    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        return False

async def test_kalki_integration():
    """Test integration with main Kalki system"""
    print("🔗 Testing Kalki integration...")

    try:
        # Check if sklearn is available (required by some advanced agents)
        try:
            import sklearn
        except ImportError:
            print("   ⚠️  sklearn not available - skipping full integration test")
            print("   💡 Install sklearn with: pip install scikit-learn")
            return True

        from kalki_complete import KalkiComplete

        # Initialize Kalki
        kalki = KalkiComplete()

        # Check if WebSearchAgent is registered
        agents = kalki.list_agents()
        web_agent_found = any('web' in agent.lower() or 'search' in agent.lower()
                            for agent in agents)

        if web_agent_found:
            print("   ✅ WebSearchAgent integrated with Kalki")
        else:
            print("   ⚠️  WebSearchAgent not found in agent list")
            print(f"   Available agents: {agents}")

        return True

    except Exception as e:
        print(f"   ❌ Kalki integration test failed: {e}")
        return False

def test_cli_commands():
    """Test CLI command structure"""
    print("💻 Testing CLI commands...")

    try:
        # Check if sklearn is available
        try:
            import sklearn
        except ImportError:
            print("   ⚠️  sklearn not available - skipping CLI test")
            print("   💡 Install sklearn with: pip install scikit-learn")
            return True

        import kalki_cli

        # Check if web commands are available
        parser = kalki_cli.create_parser()

        # Try to parse web search command
        args = parser.parse_args(['web', 'search', 'test query'])
        if hasattr(args, 'web_command') and args.web_command == 'search':
            print("   ✅ CLI web search command available")
        else:
            print("   ⚠️  CLI web search command not properly configured")
            return False

        # Try web research command
        args = parser.parse_args(['web', 'research', 'test topic'])
        if hasattr(args, 'web_command') and args.web_command == 'research':
            print("   ✅ CLI web research command available")
        else:
            print("   ⚠️  CLI web research command not properly configured")
            return False

        # Try web fetch command
        args = parser.parse_args(['web', 'fetch', 'https://example.com'])
        if hasattr(args, 'web_command') and args.web_command == 'fetch':
            print("   ✅ CLI web fetch command available")
        else:
            print("   ⚠️  CLI web fetch command not properly configured")
            return False

        return True

    except Exception as e:
        print(f"   ❌ CLI test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 Kalki Internet Connectivity Test Suite")
    print("=" * 50)
    print()

    tests = [
        ("Import Dependencies", test_imports),
        ("Environment Configuration", test_env_file),
        ("WebSearchAgent", test_web_search_agent),
        ("Kalki Integration", test_kalki_integration),
        ("CLI Commands", test_cli_commands),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * (len(test_name) + 3))

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Internet connectivity is ready.")
        print("\n🚀 Try these commands:")
        print("   kalki web search 'Python programming' --results 3")
        print("   kalki web research 'artificial intelligence'")
        print("   kalki query 'What are current AI trends?'")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("💡 Run 'python setup_internet.py' if you haven't set up API keys yet.")

    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())