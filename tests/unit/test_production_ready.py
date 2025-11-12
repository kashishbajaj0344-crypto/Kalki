#!/usr/bin/env python3
"""
Production-Ready Test - Verify all critical requirements are asked
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.game_dev_copilot import GameDevCopilot
import logging

logging.basicConfig(level=logging.WARNING)

async def test_production_requirements():
    """Test that all critical requirements are enforced"""
    print("=" * 70)
    print("PRODUCTION-READY TEST: Critical Requirements Enforcement")
    print("=" * 70)
    print()
    
    copilot = GameDevCopilot()
    
    # Start project
    result = await copilot.start_new_game_project("make me a solitaire game")
    session_id = result.get('session_id')
    
    print(f"✅ Session created: {session_id}")
    print(f"📊 Initial completeness: {result.get('completeness', 0):.0%}")
    print(f"❓ Missing critical: {result.get('missing_critical', [])}")
    print()
    
    # Answer 1: Platforms
    if result.get('status') == 'needs_input':
        next_q = result.get('next_question')
        print("Question 1:", next_q.question if next_q else 'N/A')
        print("Answer: Both Android and iOS")
        print()
        
        result = await copilot.answer_question(session_id, "Both Android and iOS")
        print(f"📊 Completeness after Q1: {result.get('completeness', 0):.0%}")
        print(f"❓ Missing critical: {result.get('missing_critical', [])}")
        print(f"Status: {result.get('status')}")
        print()
        
        if result.get('status') == 'needs_input':
            print("✅ CORRECT: Still asking questions (engine/monetization needed)")
            next_q = result.get('next_question')
            print("Question 2:", next_q.question if next_q else 'N/A')
            print()
            
            # Answer 2: Engine
            print("Answer: Unity")
            result = await copilot.answer_question(session_id, "Unity")
            print(f"📊 Completeness after Q2: {result.get('completeness', 0):.0%}")
            print(f"❓ Missing critical: {result.get('missing_critical', [])}")
            print(f"Status: {result.get('status')}")
            print()
            
            if result.get('status') == 'needs_input':
                print("✅ CORRECT: Still asking questions (monetization needed)")
                next_q = result.get('next_question')
                print("Question 3:", next_q.question if next_q else 'N/A')
                print()
                
                # Answer 3: Monetization
                print("Answer: Freemium")
                result = await copilot.answer_question(session_id, "Freemium")
                print(f"📊 Completeness after Q3: {result.get('completeness', 0):.0%}")
                print(f"Status: {result.get('status')}")
                print()
                
                if result.get('status') == 'project_created':
                    print("✅ SUCCESS: Project created after all critical requirements!")
                    print(f"Project ID: {result.get('project_id')}")
                else:
                    print(f"❌ ERROR: Expected project_created, got {result.get('status')}")
            else:
                print(f"❌ ERROR: Expected needs_input after engine, got {result.get('status')}")
        else:
            print(f"❌ ERROR: Expected needs_input after platforms, got {result.get('status')}")
            if result.get('status') == 'project_created':
                print("   Project was created too early - missing engine and monetization!")
    else:
        print(f"❌ ERROR: Expected needs_input initially, got {result.get('status')}")

if __name__ == "__main__":
    asyncio.run(test_production_requirements())

