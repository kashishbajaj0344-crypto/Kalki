#!/usr/bin/env python3
"""
Test the unified chatbot with local models
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kalki_unified_chat import UnifiedKalkiChat

async def test_chatbot():
    """Test the chatbot with a few queries"""
    print("=" * 70)
    print("Testing Kalki Unified Chatbot")
    print("=" * 70)
    
    chat = UnifiedKalkiChat()
    
    # Test queries
    test_queries = [
        "What size joists for a 16 foot span?",  # Construction
        "How do I create a Unity character controller?",  # Game Dev
        "Design a PID controller for a robot arm",  # Robotics
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {query}")
        print('='*70)
        
        try:
            result = await chat.process_message(query)
            
            print(f"\n✅ Response received!")
            print(f"Domain: {result.get('domain', 'None')}")
            print(f"Confidence: {result.get('confidence', 0.0):.2f}")
            print(f"\nResponse preview:")
            response_text = result.get('response', 'No response')
            print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Chat Statistics:")
    print('='*70)
    chat.show_stats()
    
    print(f"\n{'='*70}")
    print("✅ Test complete!")
    print('='*70)

if __name__ == "__main__":
    asyncio.run(test_chatbot())

