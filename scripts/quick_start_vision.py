#!/usr/bin/env python3
"""
Quick Start: Kalki Dual-Model Intelligence
===============================================
Get started with text + vision AI in 5 minutes
"""

import asyncio
from pathlib import Path

# Step 1: Initialize the dual-model system
async def quick_start():
    """Minimal example to get started"""
    
    from modules.llm import initialize_llm_engine, get_llm_engine
    
    print("🚀 Initializing Kalki Dual-Model Intelligence...")
    print("   • Llama 3.1 8B Instruct (text)")
    print("   • Llama 3.2 11B Vision (multimodal)")
    print()
    
    # Initialize (loads both models)
    success = await initialize_llm_engine()
    
    if not success:
        print("❌ Failed to initialize. Check model paths.")
        return
    
    llm = get_llm_engine()
    
    # Check what's loaded
    text_ok = llm.llama_engine and llm.llama_engine.pipe is not None
    vision_ok = llm.vision_engine and llm.vision_engine.is_initialized
    
    print(f"✅ Text Model:   {'Loaded' if text_ok else 'Failed'}")
    print(f"✅ Vision Model: {'Loaded' if vision_ok else 'Failed'}")
    print()
    
    if not text_ok:
        print("⚠️  No models loaded. System will use rule-based fallback.")
        return
    
    # Example 1: Simple text query
    print("="*60)
    print("EXAMPLE 1: Text Query (Fast)")
    print("="*60)
    
    query = "What is the formula for calculating maximum bending moment in a simply supported beam with uniform load?"
    print(f"Q: {query}")
    print()
    
    response = await llm.generate(query, max_new_tokens=256)
    print(f"A: {response[:500]}...")
    print()
    
    # Example 2: Vision analysis (if available)
    if vision_ok:
        print("="*60)
        print("EXAMPLE 2: Image Analysis (Vision Model)")
        print("="*60)
        
        # Create a simple test image if doesn't exist
        test_img = "data/temp_images/quick_start_diagram.png"
        if not Path(test_img).exists():
            from PIL import Image, ImageDraw
            
            Path("data/temp_images").mkdir(parents=True, exist_ok=True)
            
            img = Image.new('RGB', (600, 400), 'white')
            draw = ImageDraw.Draw(img)
            
            # Simple structural element
            draw.rectangle([100, 150, 500, 200], outline='black', width=3)
            draw.text((250, 250), "L = 15 ft", fill='black')
            draw.text((80, 160), "A", fill='black')
            draw.text((510, 160), "B", fill='black')
            
            img.save(test_img)
            print(f"Created test diagram: {test_img}")
        
        print(f"\nAnalyzing: {test_img}")
        
        response = await llm.analyze_image(
            test_img,
            "Describe this structural diagram. What are the key elements?"
        )
        
        print(f"A: {response[:400]}...")
        print()
    
    else:
        print("⚠️  Vision model not available. Install or check path:")
        print("    /Users/kashish/Desktop/Kalki/models/llama_3.2_11b_vision/")
        print()
    
    # Example 3: Extract structured data from diagram
    if vision_ok:
        print("="*60)
        print("EXAMPLE 3: Diagram Data Extraction")
        print("="*60)
        
        diagram_data = await llm.extract_diagram(test_img)
        
        print(f"Dimensions found: {len(diagram_data.get('dimensions', []))}")
        for dim in diagram_data.get('dimensions', [])[:3]:
            print(f"  • {dim}")
        
        print(f"\nMaterials mentioned: {len(diagram_data.get('materials', []))}")
        for mat in diagram_data.get('materials', [])[:3]:
            print(f"  • {mat}")
        
        print(f"\nFormulas extracted: {len(diagram_data.get('formulas', []))}")
        for formula in diagram_data.get('formulas', [])[:3]:
            print(f"  • {formula}")
        print()
    
    # Cleanup
    print("="*60)
    print("Cleaning up...")
    await llm.cleanup()
    print("✅ Done! Kalki is ready to use.")
    print()
    print("Next steps:")
    print("  1. Run full test suite: python tests/unit/test_vision_intelligence.py")
    print("  2. Process PDFs: python scripts/batch_ingest_pdfs.py --extract-images")
    print("  3. Start Kalki CLI: python src/kalki_cli.py")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  KALKI - Dual-Model Vision Intelligence                ║
║  Quick Start Demo                                            ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    asyncio.run(quick_start())
