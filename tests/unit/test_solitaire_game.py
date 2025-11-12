#!/usr/bin/env python3
"""
Test Solitaire Game Development
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests the GameDevCopilot with "solitaire style game" input
and shows the full question-answering flow.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.game_dev_copilot import GameDevCopilot
import logging

# Set up logging (less verbose)
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings/errors
    format='%(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_solitaire_game():
    """Test solitaire game development with interactive flow"""
    print("=" * 70)
    print("🎮 SOLITAIRE GAME DEVELOPMENT TEST")
    print("=" * 70)
    print()
    
    try:
        copilot = GameDevCopilot()
        
        # User gives minimal input
        user_input = "make me a solitaire style game"
        print(f"👤 User: {user_input}")
        print()
        print("🤖 KALKI: [Processing your request...]")
        print()
        
        result = await copilot.start_new_game_project(user_input)
        
        # Show initial response
        if result.get('message'):
            print(result['message'])
            print()
        
        session_id = result.get('session_id')
        if not session_id:
            print("❌ No session created")
            return
        
        # Interactive question-answering
        question_count = 0
        max_questions = 6
        
        # Pre-defined answers for solitaire game
        answers = {
            'platform': 'Both Android and iOS',
            'engine': 'Unity',
            'monetization': 'Freemium with ads',
            'genre': 'Puzzle',
            'art_style': '2D classic card style'
        }
        
        while result.get('status') == 'needs_input' and question_count < max_questions:
            next_q = result.get('next_question')
            if not next_q:
                break
            
            question_count += 1
            print(f"❓ Question {question_count}: {next_q.question}")
            print()
            
            # Get answer based on category
            answer = None
            category = next_q.category
            
            if 'platform' in category:
                answer = answers.get('platform', 'Both Android and iOS')
            elif 'engine' in category:
                answer = answers.get('engine', 'Unity')
            elif 'monetization' in category:
                answer = answers.get('monetization', 'Freemium')
            elif 'genre' in category:
                answer = answers.get('genre', 'Puzzle')
            elif 'art' in category:
                answer = answers.get('art_style', '2D')
            else:
                answer = "I'll let you decide"
            
            print(f"👤 User: {answer}")
            print()
            print("🤖 KALKI: [Processing your answer...]")
            print()
            
            result = await copilot.answer_question(session_id, answer)
            
            if result.get('status') == 'needs_input':
                completeness = result.get('completeness', 0)
                print(f"📊 Requirements Progress: {completeness:.0%} complete")
                print()
            else:
                # Project created!
                print("=" * 70)
                print("✅ PROJECT CREATED SUCCESSFULLY!")
                print("=" * 70)
                print()
                
                # Print message if available
                message = result.get('message', '')
                if message:
                    print(message)
                    print()
                else:
                    print("🎮 Game Project Created!")
                    print()
                
                # Show project details
                project_id = result.get('project_id')
                requirements = result.get('requirements')
                
                if project_id:
                    print(f"📋 Project ID: {project_id}")
                    print()
                
                if requirements:
                    print("📝 Project Requirements:")
                    print(f"  • Concept: {requirements.game_concept}")
                    if requirements.genre:
                        print(f"  • Genre: {requirements.genre.value}")
                    if requirements.target_platforms:
                        print(f"  • Platforms: {', '.join(requirements.target_platforms)}")
                    if requirements.game_engine:
                        print(f"  • Engine: {requirements.game_engine}")
                    if requirements.monetization_model:
                        print(f"  • Monetization: {requirements.monetization_model}")
                    if requirements.art_style:
                        print(f"  • Art Style: {requirements.art_style}")
                    print()
                
                # Show roadmap
                roadmap = result.get('roadmap', {})
                if roadmap:
                    print("🗺️  Development Roadmap:")
                    steps = roadmap.get('immediate_next_steps', [])
                    if steps:
                        for i, step in enumerate(steps, 1):
                            print(f"  {i}. {step}")
                    else:
                        print("  • Set up development environment")
                        print("  • Create project structure")
                        print("  • Implement card game mechanics")
                        print("  • Design UI/UX")
                        print("  • Add game logic")
                    print()
                
                print("=" * 70)
                print("🎉 Ready to start building your solitaire game!")
                print("=" * 70)
                break
        
        if result.get('status') == 'needs_input':
            print("⚠️  Still need more information to create project")
            print(f"   Completeness: {result.get('completeness', 0):.0%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_solitaire_game())

