#!/usr/bin/env python3
"""
Full System Integration Test
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests the complete KALKI system flow:
1. Domain Registry
2. Copilot Loading
3. Supreme Control Hub
4. Unified Chat Interface
5. Game Dev Copilot
6. Construction Copilot
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.domains.domain_registry import DomainRegistry
from modules.supreme_control_hub import SupremeControlHub
from apps.kalki_unified_chat import UnifiedKalkiChat

async def test_domain_registry():
    """Test 1: Domain Registry"""
    print("\n" + "="*70)
    print("TEST 1: Domain Registry")
    print("="*70)
    
    try:
        dr = DomainRegistry()
        domains = dr.list_domains()
        print(f"✅ Domain Registry loaded")
        print(f"✅ Found {len(domains)} domains: {', '.join(domains)}")
        
        # Test copilot detection
        has_game_dev = dr.has_copilot("game_development")
        has_construction = dr.has_copilot("construction")
        
        print(f"✅ Game Dev Copilot: {'Available' if has_game_dev else 'Not Available'}")
        print(f"✅ Construction Copilot: {'Available' if has_construction else 'Not Available'}")
        
        # Test domain inference
        test_queries = [
            "make me a solitaire game",
            "build a house",
            "design a robot arm"
        ]
        
        for query in test_queries:
            inferred = await dr.infer_domain(query)
            print(f"✅ Query: '{query}' → Domain: {inferred[0] if inferred else 'None'}")
        
        return True
    except Exception as e:
        print(f"❌ Domain Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_supreme_control_hub():
    """Test 2: Supreme Control Hub"""
    print("\n" + "="*70)
    print("TEST 2: Supreme Control Hub")
    print("="*70)
    
    try:
        hub = SupremeControlHub()
        print("✅ Supreme Control Hub initialized")
        
        # Test domain-aware query
        test_query = "make me a solitaire style game"
        print(f"\n📝 Testing query: '{test_query}'")
        
        result = await hub.process_domain_aware_query(
            query=test_query,
            context={},
            project_id=None
        )
        
        if result.get("success"):
            print(f"✅ Query processed successfully")
            print(f"   Domain: {result.get('domain', {}).get('name', 'Unknown')}")
            print(f"   Copilot Used: {result.get('domain', {}).get('copilot_used', False)}")
            print(f"   Response: {result.get('answer', '')[:100]}...")
            return True
        else:
            print(f"⚠️  Query processing returned success=False")
            print(f"   Error: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Supreme Control Hub test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_game_dev_copilot():
    """Test 3: Game Dev Copilot Direct"""
    print("\n" + "="*70)
    print("TEST 3: Game Dev Copilot (Direct)")
    print("="*70)
    
    try:
        from modules.game_dev_copilot import GameDevCopilot
        
        copilot = GameDevCopilot()
        print("✅ Game Dev Copilot loaded")
        
        # Test starting a new project
        result = await copilot.start_new_game_project("make me a solitaire game")
        
        if result.get("status") == "question_asked":
            print(f"✅ Project creation initiated")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Next Question: {result.get('next_question', {}).get('question', 'N/A')[:80]}...")
            return True
        elif result.get("status") == "project_created":
            print(f"✅ Project created directly")
            print(f"   Project ID: {result.get('project_id')}")
            return True
        else:
            print(f"⚠️  Unexpected status: {result.get('status')}")
            return False
            
    except Exception as e:
        print(f"❌ Game Dev Copilot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_construction_copilot():
    """Test 4: Construction Copilot Direct"""
    print("\n" + "="*70)
    print("TEST 4: Construction Copilot (Direct)")
    print("="*70)
    
    try:
        from modules.construction_copilot_enhanced import EnhancedConstructionCopilot
        
        copilot = EnhancedConstructionCopilot()
        print("✅ Construction Copilot loaded")
        print(f"✅ Copilot has {len(copilot.active_projects)} active projects")
        return True
            
    except Exception as e:
        print(f"❌ Construction Copilot test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_unified_chat():
    """Test 5: Unified Chat Interface"""
    print("\n" + "="*70)
    print("TEST 5: Unified Chat Interface")
    print("="*70)
    
    try:
        chat = UnifiedKalkiChat()
        print("✅ Unified Chat initialized")
        
        # Test domain detection
        test_query = "what size joists for a 16 foot span"
        print(f"\n📝 Testing query: '{test_query}'")
        
        result = await chat.process_message(test_query)
        
        if result.get("response"):
            print(f"✅ Message processed")
            print(f"   Domain: {result.get('domain', 'None')}")
            print(f"   Response length: {len(result.get('response', ''))} chars")
            return True
        else:
            print(f"⚠️  No response received")
            return False
            
    except Exception as e:
        print(f"❌ Unified Chat test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_end_to_end_flow():
    """Test 6: Complete End-to-End Flow"""
    print("\n" + "="*70)
    print("TEST 6: End-to-End Flow (Game Dev)")
    print("="*70)
    
    try:
        # Simulate full flow: User → Unified Chat → Supreme Hub → Copilot
        chat = UnifiedKalkiChat()
        
        # Step 1: User asks for a game
        query1 = "make me a solitaire style game"
        print(f"\n📝 Step 1: User query: '{query1}'")
        
        result1 = await chat.process_message(query1)
        
        if result1.get("response") or result1.get("next_question"):
            print(f"✅ Step 1 passed - Query routed correctly")
            print(f"   Domain detected: {result1.get('domain', 'None')}")
            
            # Check if we got a question (expected for game dev)
            if result1.get("next_question"):
                print(f"   ✅ Smart question flow initiated")
                return True
            elif "project" in result1.get("response", "").lower():
                print(f"   ✅ Project creation response received")
                return True
            else:
                print(f"   ⚠️  Unexpected response format")
                return False
        else:
            print(f"❌ Step 1 failed - No response")
            return False
            
    except Exception as e:
        print(f"❌ End-to-End test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("\n" + "🔍 " * 35)
    print("KALKI FULL SYSTEM INTEGRATION TEST")
    print("🔍 " * 35)
    
    results = []
    
    # Run tests
    results.append(("Domain Registry", await test_domain_registry()))
    results.append(("Supreme Control Hub", await test_supreme_control_hub()))
    results.append(("Game Dev Copilot", await test_game_dev_copilot()))
    results.append(("Construction Copilot", await test_construction_copilot()))
    results.append(("Unified Chat", await test_unified_chat()))
    results.append(("End-to-End Flow", await test_end_to_end_flow()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({passed*100//total}%)")
    print(f"{'='*70}\n")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - System is fully integrated!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed - System is mostly working")
    else:
        print("⚠️  Some tests failed - System needs attention")

if __name__ == "__main__":
    asyncio.run(main())

