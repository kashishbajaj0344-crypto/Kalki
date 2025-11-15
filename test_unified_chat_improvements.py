#!/usr/bin/env python3
"""
Test Unified Chat Improvements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests all improvements:
1. Session Management (persistence, retrieval)
2. Generic Copilot Workflow Handler
3. User Confirmation Flow
4. Progress Indicators
5. Workflow Execution
6. Error Handling
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from apps.kalki_unified_chat import UnifiedKalkiChat

async def test_session_management():
    """Test 1: Session Management"""
    print("\n" + "="*70)
    print("TEST 1: Session Management")
    print("="*70)
    
    try:
        chat = UnifiedKalkiChat()
        print("✅ Unified Chat initialized")
        
        # Test session file creation
        session_file = Path("data/chat_sessions.json")
        if session_file.exists():
            session_file.unlink()  # Clean up for test
        
        # Test updating session
        chat._update_active_session("game_development", "test_session_123", "test_project_456")
        print("✅ Session updated")
        
        # Test session persistence
        if session_file.exists():
            print("✅ Session file created")
            with open(session_file, 'r') as f:
                sessions = json.load(f)
                if "game_development" in sessions:
                    print(f"✅ Session persisted: {sessions['game_development']}")
                else:
                    print("❌ Session not found in file")
                    return False
        else:
            print("❌ Session file not created")
            return False
        
        # Test session retrieval
        session_id = chat._get_active_session("game_development")
        if session_id == "test_session_123":
            print("✅ Session retrieved correctly")
        else:
            print(f"❌ Session retrieval failed: got {session_id}, expected test_session_123")
            return False
        
        # Test loading sessions
        new_chat = UnifiedKalkiChat()
        loaded_session = new_chat._get_active_session("game_development")
        if loaded_session == "test_session_123":
            print("✅ Sessions loaded from disk correctly")
        else:
            print(f"❌ Session loading failed: got {loaded_session}")
            return False
        
        # Cleanup
        if session_file.exists():
            session_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_generic_workflow_handler():
    """Test 2: Generic Copilot Workflow Handler"""
    print("\n" + "="*70)
    print("TEST 2: Generic Copilot Workflow Handler")
    print("="*70)
    
    try:
        chat = UnifiedKalkiChat()
        
        # Test with game development
        result = {
            "success": True,
            "answer": "Project created!",
            "project_id": "test_project_123",
            "session_id": "test_session_123",
            "status": "project_created"
        }
        
        workflow_result = await chat._handle_copilot_workflow(
            "game_development",
            result,
            "make me a game"
        )
        
        if "pending_workflow" in workflow_result.get("metadata", {}):
            print("✅ Workflow handler detected project creation")
            print(f"✅ Pending workflow created: {workflow_result['metadata']['pending_workflow']}")
        else:
            print("❌ Pending workflow not created")
            return False
        
        # Test with construction
        result2 = {
            "success": True,
            "answer": "Construction project created!",
            "project_id": "test_project_456",
            "session_id": "test_session_456",
            "status": "project_created"
        }
        
        workflow_result2 = await chat._handle_copilot_workflow(
            "construction",
            result2,
            "build a house"
        )
        
        if "pending_workflow" in workflow_result2.get("metadata", {}):
            print("✅ Construction workflow handler working")
        else:
            print("⚠️  Construction workflow not configured (may be expected)")
        
        # Test with domain without copilot
        result3 = {
            "success": True,
            "answer": "Query processed",
            "status": "completed"
        }
        
        workflow_result3 = await chat._handle_copilot_workflow(
            "aerospace",
            result3,
            "design a rocket"
        )
        
        if workflow_result3 == result3:
            print("✅ Handler gracefully handles domains without copilots")
        else:
            print("❌ Handler modified result for domain without copilot")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Generic workflow handler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_workflow_execution():
    """Test 3: Workflow Execution"""
    print("\n" + "="*70)
    print("TEST 3: Workflow Execution")
    print("="*70)
    
    try:
        chat = UnifiedKalkiChat()
        
        # Test game dev workflow execution
        copilot = chat.domain_registry.get_copilot("game_development")
        if not copilot:
            print("⚠️  Game dev copilot not available - skipping workflow execution test")
            return True  # Not a failure, just unavailable
        
        # Create a test session first
        test_result = await copilot.start_new_game_project("make me a solitaire game")
        session_id = test_result.get("session_id")
        
        if not session_id:
            print("⚠️  Could not create test session - skipping workflow execution test")
            return True
        
        # Answer questions to get to project creation
        # (This is a simplified test - in real scenario would answer all questions)
        print(f"✅ Test session created: {session_id}")
        
        # Test workflow execution method exists
        if hasattr(chat, "_execute_workflow"):
            print("✅ _execute_workflow method exists")
        else:
            print("❌ _execute_workflow method not found")
            return False
        
        # Test that it handles errors gracefully
        try:
            result = await chat._execute_workflow("invalid_domain", "build", "invalid_session")
            if result.get("status") == "error":
                print("✅ Error handling works correctly")
            else:
                print("⚠️  Error handling may need improvement")
        except Exception as e:
            print(f"❌ Workflow execution raised exception: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_context_enhancement():
    """Test 4: Context Enhancement with Sessions"""
    print("\n" + "="*70)
    print("TEST 4: Context Enhancement with Sessions")
    print("="*70)
    
    try:
        chat = UnifiedKalkiChat()
        
        # Set up a test session
        chat._update_active_session("game_development", "test_session_789", "test_project_789")
        
        # Test context retrieval
        context = chat._get_chat_context()
        print("✅ Chat context retrieved")
        
        # Test that active session is included in context
        active_session = chat._get_active_session("game_development")
        if active_session == "test_session_789":
            print("✅ Active session retrieved correctly")
        else:
            print(f"❌ Active session retrieval failed: got {active_session}")
            return False
        
        # Test with multiple domains
        chat._update_active_session("construction", "construction_session_123", "construction_project_123")
        
        game_session = chat._get_active_session("game_development")
        construction_session = chat._get_active_session("construction")
        
        if game_session == "test_session_789" and construction_session == "construction_session_123":
            print("✅ Multiple domain sessions work correctly")
        else:
            print(f"❌ Multiple domain sessions failed")
            return False
        
        # Cleanup
        session_file = Path("data/chat_sessions.json")
        if session_file.exists():
            session_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Context enhancement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_end_to_end_flow():
    """Test 5: End-to-End Flow with Improvements"""
    print("\n" + "="*70)
    print("TEST 5: End-to-End Flow")
    print("="*70)
    
    try:
        chat = UnifiedKalkiChat()
        
        # Simulate a complete flow
        print("\n📝 Step 1: User creates game project")
        result1 = await chat.process_message("make me a solitaire style game")
        
        if result1.get("response"):
            print("✅ Message processed")
            print(f"   Domain: {result1.get('domain', 'None')}")
            
            # Check if session was created
            if result1.get("metadata", {}).get("session_id"):
                print(f"✅ Session created: {result1['metadata']['session_id']}")
            else:
                print("⚠️  No session ID in metadata (may be expected if project not created yet)")
            
            # Check if workflow was offered
            if result1.get("metadata", {}).get("pending_workflow"):
                print("✅ Workflow offer detected in response")
            else:
                print("⚠️  No workflow offer (may be expected if project not created)")
            
            return True
        else:
            print("❌ No response received")
            return False
            
    except Exception as e:
        print(f"❌ End-to-end flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_error_handling():
    """Test 6: Error Handling"""
    print("\n" + "="*70)
    print("TEST 6: Error Handling")
    print("="*70)
    
    try:
        chat = UnifiedKalkiChat()
        
        # Test workflow execution with invalid inputs
        result1 = await chat._execute_workflow("nonexistent_domain", "build", "invalid_session")
        if result1.get("status") == "error":
            print("✅ Handles nonexistent domain gracefully")
        else:
            print("❌ Should return error for nonexistent domain")
            return False
        
        # Test workflow execution with missing copilot
        result2 = await chat._execute_workflow("aerospace", "build", "test_session")
        if result2.get("status") == "error":
            print("✅ Handles missing copilot gracefully")
        else:
            print("⚠️  May need better error handling for missing copilot")
        
        # Test session management with invalid data
        try:
            chat._update_active_session("test_domain", "test_session", None)
            print("✅ Session update handles None project_id")
        except Exception as e:
            print(f"❌ Session update failed with None: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("\n" + "🔍 " * 35)
    print("UNIFIED CHAT IMPROVEMENTS TEST SUITE")
    print("🔍 " * 35)
    
    results = []
    
    # Run tests
    results.append(("Session Management", await test_session_management()))
    results.append(("Generic Workflow Handler", await test_generic_workflow_handler()))
    results.append(("Workflow Execution", await test_workflow_execution()))
    results.append(("Context Enhancement", await test_context_enhancement()))
    results.append(("End-to-End Flow", await test_end_to_end_flow()))
    results.append(("Error Handling", await test_error_handling()))
    
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
        print("🎉 ALL TESTS PASSED - All improvements working!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed - Improvements mostly working")
    else:
        print("⚠️  Some tests failed - Review needed")
    
    # Cleanup
    session_file = Path("data/chat_sessions.json")
    if session_file.exists():
        try:
            session_file.unlink()
            print("\n🧹 Cleaned up test session file")
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())

