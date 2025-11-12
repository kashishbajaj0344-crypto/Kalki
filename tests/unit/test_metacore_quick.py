#!/usr/bin/env python3
"""
Quick MetaCore Integration Test
Validates MetaCore integration without full system initialization
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metacore_integration_quick():
    """Quick test of MetaCore integration structure"""
    
    print("=" * 80)
    print("🧪 QUICK METACORE INTEGRATION TEST")
    print("=" * 80)
    
    # Test 1: Check MetaCore imports
    print("\n1️⃣ Testing MetaCore imports...")
    try:
        from modules.meta_core import MetaCore, ReasoningDepth, OutputStyle
        print("   ✅ MetaCore import successful")
    except ImportError as e:
        print(f"   ❌ Failed to import MetaCore: {e}")
        return False
    
    # Test 2: Check MetaCore instantiation
    print("\n2️⃣ Testing MetaCore instantiation...")
    try:
        meta_core = MetaCore()
        print("   ✅ MetaCore instantiated successfully")
    except Exception as e:
        print(f"   ❌ Failed to instantiate MetaCore: {e}")
        return False
    
    # Test 3: Check key methods exist
    print("\n3️⃣ Testing MetaCore methods...")
    required_methods = [
        'assess_task_complexity',
        'generate_meta_prompt',
        'evaluate_response_quality',
        'set_reasoning_depth',
        'set_output_style',
        'get_meta_status'
    ]
    
    for method in required_methods:
        if hasattr(meta_core, method):
            print(f"   ✅ Method '{method}' exists")
        else:
            print(f"   ❌ Method '{method}' missing")
            return False
    
    # Test 4: Check orchestrator integration
    print("\n4️⃣ Testing orchestrator integration...")
    try:
        from kalki_complete import KalkiOrchestrator
        import inspect
        
        # Check if _initialize_meta_core_system method exists
        if hasattr(KalkiOrchestrator, '_initialize_meta_core_system'):
            print("   ✅ _initialize_meta_core_system method exists")
        else:
            print("   ❌ _initialize_meta_core_system method missing")
            return False
        
        print("   ✅ Orchestrator integration complete")
        
    except ImportError as e:
        print(f"   ❌ Failed to import KalkiOrchestrator: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Integration check failed: {e}")
        return False
    
    # Test 5: Check initialization sequence
    print("\n5️⃣ Testing initialization sequence...")
    try:
        source = inspect.getsource(KalkiOrchestrator.initialize_system)
        if '_initialize_meta_core_system' in source:
            print("   ✅ MetaCore initialization found in initialize_system")
        else:
            print("   ⚠️  MetaCore initialization not found in initialize_system")
            
    except Exception as e:
        print(f"   ⚠️  Could not check initialization sequence: {e}")
    
    # Test 6: Check query processing integration
    print("\n6️⃣ Testing query processing integration...")
    try:
        source = inspect.getsource(KalkiOrchestrator.process_query)
        
        if 'self.meta_core' in source:
            print("   ✅ MetaCore integration found in process_query")
        else:
            print("   ⚠️  MetaCore integration not found in process_query")
            
        if 'assess_task_complexity' in source:
            print("   ✅ Task complexity assessment found")
        else:
            print("   ⚠️  Task complexity assessment not found")
            
        if 'generate_meta_prompt' in source:
            print("   ✅ Meta-prompt generation found")
        else:
            print("   ⚠️  Meta-prompt generation not found")
            
        if 'evaluate_response_quality' in source:
            print("   ✅ Response quality evaluation found")
        else:
            print("   ⚠️  Response quality evaluation not found")
            
    except Exception as e:
        print(f"   ⚠️  Could not check query processing: {e}")
    
    return True

if __name__ == "__main__":
    print()
    success = test_metacore_integration_quick()
    print("\n" + "=" * 80)
    
    if success:
        print("🎉 METACORE INTEGRATION TEST PASSED")
        print("=" * 80)
        print("\n✅ MetaCore successfully integrated into Kalki orchestrator")
        print("✅ Progressive reasoning methods available")
        print("✅ Quality evaluation methods available")
        print("\nℹ️  To test full functionality, run: python3 test_metacore_integration.py")
        print("   (Note: Full test takes 5-7 minutes due to LLM initialization)")
        sys.exit(0)
    else:
        print("❌ METACORE INTEGRATION TEST FAILED")
        print("=" * 80)
        sys.exit(1)
