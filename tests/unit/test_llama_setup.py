#!/usr/bin/env python3
"""
Test Llama 3.1 8B Setup with HuggingFace
Verify the model loads and generates responses
"""

import asyncio
import sys
from modules.llm import get_llm_engine, initialize_llm_engine

async def test_llama():
    print("🧪 Testing Llama 3.1 8B Setup")
    print("=" * 60)
    
    # Initialize LLM engine
    print("\n1️⃣ Initializing LLM engine...")
    success = await initialize_llm_engine()
    
    if not success:
        print("❌ Failed to initialize LLM engine")
        return False
    
    print("✅ LLM engine initialized")
    
    # Get engine instance
    engine = get_llm_engine()
    print(f"\n2️⃣ Model info:")
    print(f"   Model: {engine.llama_engine.model_name if hasattr(engine, 'llama_engine') else 'Unknown'}")
    print(f"   Device: {engine.llama_engine.device if hasattr(engine, 'llama_engine') else 'Unknown'}")
    
    # Test generation
    print("\n3️⃣ Testing text generation...")
    test_prompt = "What are the key steps in building a house foundation?"
    
    print(f"\n   Prompt: {test_prompt}")
    print(f"\n   Generating response...")
    
    response = await engine.generate(test_prompt, max_new_tokens=150)
    
    print(f"\n   Response:")
    print(f"   {'-' * 55}")
    print(f"   {response[:500]}...")
    print(f"   {'-' * 55}")
    
    if len(response) > 20 and "error" not in response.lower():
        print("\n✅ Generation successful!")
        return True
    else:
        print("\n❌ Generation failed or returned error")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_llama())
        if success:
            print("\n" + "=" * 60)
            print("🎉 Llama 3.1 8B is working perfectly!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("❌ Llama 3.1 8B test failed")
            print("=" * 60)
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
