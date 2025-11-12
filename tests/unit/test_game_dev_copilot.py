#!/usr/bin/env python3
"""
Test Game Development Copilot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests the GameDevCopilot with minimal input like "make me a carjam style game"
and demonstrates the smart question-asking flow.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.game_dev_copilot import GameDevCopilot
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_minimal_input():
    """Test with minimal input - just game reference"""
    print("=" * 70)
    print("TEST 1: Minimal Input - 'make me a carjam style game'")
    print("=" * 70)
    print()
    
    try:
        copilot = GameDevCopilot()
        
        # User gives minimal input
        user_input = "make me a carjam style game"
        print(f"👤 User: {user_input}")
        print()
        
        result = await copilot.start_new_game_project(user_input)
        
        print("🤖 KALKI Response:")
        print(result.get('message', 'No message'))
        print()
        
        if result.get('status') == 'needs_input':
            print(f"📊 Requirements Completeness: {result.get('completeness', 0):.0%}")
            print(f"❓ Questions to ask: {len(result.get('questions', []))}")
            print()
            
            # Show next question
            next_q = result.get('next_question')
            if next_q:
                print(f"Next Question Category: {next_q.category}")
                print(f"Importance: {next_q.importance}")
                print()
            
            return result
        else:
            print("✅ Project created successfully!")
            print(f"Project ID: {result.get('project_id')}")
            return result
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_question_answering():
    """Test the full question-answering flow"""
    print("\n" + "=" * 70)
    print("TEST 2: Full Question-Answering Flow")
    print("=" * 70)
    print()
    
    try:
        copilot = GameDevCopilot()
        
        # Start with minimal input
        user_input = "make me a carjam style game"
        print(f"👤 User: {user_input}")
        print()
        
        result = await copilot.start_new_game_project(user_input)
        session_id = result.get('session_id')
        
        if not session_id:
            print("❌ No session ID returned")
            return
        
        print("🤖 KALKI: [Researches carjam...]")
        print(result.get('message', ''))
        print()
        
        # Simulate user answers
        answers = [
            ("Android and iOS", "platform"),
            ("Unity", "engine"),
            ("Freemium with ads", "monetization"),
        ]
        
        question_count = 0
        max_questions = 5  # Limit to prevent infinite loop
        
        while result.get('status') == 'needs_input' and question_count < max_questions:
            next_q = result.get('next_question')
            if not next_q:
                break
            
            print(f"❓ Question {question_count + 1}: {next_q.question}")
            print()
            
            # Find matching answer
            answer = None
            for ans, category in answers:
                if category in next_q.category:
                    answer = ans
                    break
            
            if not answer:
                # Default answers
                if 'platform' in next_q.category:
                    answer = "Both Android and iOS"
                elif 'engine' in next_q.category:
                    answer = "Unity"
                elif 'monetization' in next_q.category:
                    answer = "Freemium"
                elif 'genre' in next_q.category:
                    answer = "Racing"
                else:
                    answer = "I don't know"
            
            print(f"👤 User: {answer}")
            print()
            
            result = await copilot.answer_question(session_id, answer)
            
            if result.get('status') == 'needs_input':
                print(f"📊 Completeness: {result.get('completeness', 0):.0%}")
                print()
            else:
                print("✅ Project Created!")
                print(result.get('message', ''))
                print()
                break
            
            question_count += 1
        
        if result.get('status') == 'project_created':
            print("🎉 SUCCESS: Project created from minimal input!")
            print(f"Project ID: {result.get('project_id')}")
            print(f"Requirements: {result.get('requirements', {})}")
            print()
            
            # Show roadmap
            roadmap = result.get('roadmap', {})
            if roadmap:
                print("📋 Development Roadmap:")
                for step in roadmap.get('immediate_next_steps', []):
                    print(f"  • {step}")
                print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_different_game_references():
    """Test with different game references"""
    print("\n" + "=" * 70)
    print("TEST 3: Different Game References")
    print("=" * 70)
    print()
    
    test_cases = [
        "make me a flappy bird style game",
        "build me a puzzle game like candy crush",
        "create a racing game similar to temple run",
    ]
    
    copilot = GameDevCopilot()
    
    for user_input in test_cases:
        print(f"👤 User: {user_input}")
        print()
        
        try:
            result = await copilot.start_new_game_project(user_input)
            
            if result.get('status') == 'needs_input':
                print("✅ KALKI understood the reference and is asking questions")
                print(f"📊 Completeness: {result.get('completeness', 0):.0%}")
            else:
                print("✅ Project created")
            
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


async def main():
    """Run all tests"""
    print("\n" + "🎮 " * 35)
    print("GAME DEVELOPMENT COPILOT TEST SUITE")
    print("🎮 " * 35)
    print()
    
    # Test 1: Minimal input
    await test_minimal_input()
    
    # Test 2: Full flow
    await test_question_answering()
    
    # Test 3: Different references
    await test_different_game_references()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

