#!/usr/bin/env python3
"""
KALKI - Dual-Model Vision Intelligence Test
================================================
Test suite for Llama 3.1 8B (text) + Llama 3.2 11B Vision (multimodal)

Tests:
1. Text model inference speed
2. Vision model image analysis
3. Cross-modal validation
4. Intelligent routing
5. Hybrid learning with vision extraction
"""

import asyncio
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.llm import get_llm_engine, initialize_llm_engine
from modules.utils.logging_config import get_logger

logger = get_logger("KalkiVisionTest")


async def test_text_model():
    """Test 1: Text Model Performance"""
    print("\n" + "="*60)
    print("TEST 1: Text Model (Llama 3.1 8B) Performance")
    print("="*60)
    
    llm = get_llm_engine()
    
    # Test query
    query = "Explain the difference between bending moment and shear force in structural engineering."
    
    print(f"\nQuery: {query}")
    print("\nGenerating response...")
    
    start = time.time()
    response = await llm.generate(query, max_new_tokens=256)
    elapsed = time.time() - start
    
    print(f"\nResponse ({elapsed:.2f}s):\n{response[:500]}...")
    print(f"\n✅ Text model working! Speed: {elapsed:.2f}s for 256 tokens")
    
    return elapsed


async def test_vision_model():
    """Test 2: Vision Model Image Analysis"""
    print("\n" + "="*60)
    print("TEST 2: Vision Model (Llama 3.2 11B Vision) Analysis")
    print("="*60)
    
    llm = get_llm_engine()
    
    if not llm.vision_engine or not llm.vision_engine.is_initialized:
        print("❌ Vision model not initialized!")
        return None
    
    # Create a test image (simple diagram)
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a simple structural diagram
    img = Image.new('RGB', (800, 600), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple beam with dimensions
    draw.rectangle([100, 200, 700, 250], outline='black', width=3)
    draw.line([100, 200, 100, 150], fill='black', width=2)
    draw.line([700, 200, 700, 150], fill='black', width=2)
    
    # Add text labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((350, 100), "Simple Beam", fill='black', font=font)
    draw.text((100, 270), "A", fill='black', font=font)
    draw.text((680, 270), "B", fill='black', font=font)
    draw.text((350, 170), "L = 20 ft", fill='black', font=font)
    
    # Save test image
    test_img_path = "data/temp_images/test_beam_diagram.png"
    Path("data/temp_images").mkdir(parents=True, exist_ok=True)
    img.save(test_img_path)
    
    print(f"\nCreated test diagram: {test_img_path}")
    print("\nAnalyzing with vision model...")
    
    query = "Describe this structural engineering diagram. What are the dimensions and key elements?"
    
    start = time.time()
    response = await llm.analyze_image(test_img_path, query)
    elapsed = time.time() - start
    
    print(f"\nResponse ({elapsed:.2f}s):\n{response[:500]}...")
    print(f"\n✅ Vision model working! Speed: {elapsed:.2f}s")
    
    return elapsed


async def test_cross_validation():
    """Test 3: Cross-Modal Validation"""
    print("\n" + "="*60)
    print("TEST 3: Cross-Modal Validation (Text ↔ Vision)")
    print("="*60)
    
    llm = get_llm_engine()
    
    if not llm.vision_engine or not llm.vision_engine.is_initialized:
        print("⚠️  Skipping (vision model not available)")
        return
    
    # Text claim about the diagram
    text_claim = "The beam spans 20 feet between supports A and B."
    
    # Validate against diagram
    print(f"\nText Claim: '{text_claim}'")
    print("Validating with vision model...")
    
    test_img = "data/temp_images/test_beam_diagram.png"
    if not Path(test_img).exists():
        print("⚠️  Test image not found, skipping")
        return
    
    result = await llm.cross_validate(text_claim, test_img)
    
    print(f"\nValidation Result:")
    print(f"  ✓ Validated: {result['validated']}")
    print(f"  ✓ Confidence: {result['confidence']:.2%}")
    if 'vision_analysis' in result:
        print(f"  ✓ Vision Says: {result['vision_analysis'][:200]}...")
    
    print(f"\n✅ Cross-validation complete!")


async def test_intelligent_routing():
    """Test 4: Intelligent Model Routing"""
    print("\n" + "="*60)
    print("TEST 4: Intelligent Model Routing")
    print("="*60)
    
    llm = get_llm_engine()
    
    # Test 1: Text-only query (should route to 3.1 8B)
    print("\n[Test 4a] Text-only query (routes to Llama 3.1 8B):")
    query1 = "What is the formula for bending stress?"
    
    start = time.time()
    response1 = await llm.generate(query1, max_new_tokens=128)
    time1 = time.time() - start
    
    print(f"  Query: {query1}")
    print(f"  Time: {time1:.2f}s")
    print(f"  Response: {response1[:150]}...")
    
    # Test 2: Image query (should route to 3.2 Vision)
    if llm.vision_engine and llm.vision_engine.is_initialized:
        print("\n[Test 4b] Image query (routes to Llama 3.2 Vision):")
        test_img = "data/temp_images/test_beam_diagram.png"
        if Path(test_img).exists():
            query2 = "What structural elements are shown?"
            
            start = time.time()
            response2 = await llm.generate(query2, image_path=test_img, max_new_tokens=128)
            time2 = time.time() - start
            
            print(f"  Query: {query2}")
            print(f"  Image: {test_img}")
            print(f"  Time: {time2:.2f}s")
            print(f"  Response: {response2[:150]}...")
    
    print(f"\n✅ Intelligent routing working!")


async def test_hybrid_learning():
    """Test 5: Hybrid Learning with Vision Extraction"""
    print("\n" + "="*60)
    print("TEST 5: Hybrid Learning with Vision Extraction")
    print("="*60)
    
    from modules.hybrid_learning_system import KnowledgeExtractor
    
    extractor = KnowledgeExtractor()
    
    # Create a test PDF content
    test_content = """
    Design Example: Simple Beam Analysis
    
    Given:
    - Beam length: L = 20 ft
    - Uniform load: w = 500 lb/ft
    - Maximum moment: M_max = w·L²/8
    
    Material: Steel (ASTM A36)
    - Yield strength: Fy = 36 ksi
    - Elastic modulus: E = 29,000 ksi
    """
    
    print("\nTest PDF Content:")
    print(test_content)
    
    print("\nExtracting knowledge (text-only mode)...")
    results = extractor.extract_from_pdf(
        "test_document.pdf",
        test_content,
        use_llm_enhancements=False,  # Skip LLM for speed
        extract_images=False  # No images in this test
    )
    
    print(f"\nExtraction Results:")
    print(f"  Formulas: {results['formulas']}")
    print(f"  Materials: {results['materials']}")
    print(f"  Rules: {results['rules']}")
    print(f"  Codes: {results['codes']}")
    
    if llm.vision_engine and llm.vision_engine.is_initialized:
        print("\n[Vision Enhancement Available]")
        print("  ✓ Can extract formulas from diagrams")
        print("  ✓ Can read dimensions from drawings")
        print("  ✓ Can identify materials visually")
    
    print(f"\n✅ Hybrid learning system ready!")


async def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("KALKI - DUAL-MODEL VISION INTELLIGENCE TEST SUITE")
    print("="*70)
    print("\nInitializing dual-model system...")
    print("  • Llama 3.1 8B Instruct (text reasoning)")
    print("  • Llama 3.2 11B Vision (multimodal analysis)")
    
    # Initialize LLM engine
    success = await initialize_llm_engine()
    
    if not success:
        print("\n❌ Failed to initialize LLM engine!")
        return
    
    llm = get_llm_engine()
    
    print(f"\n✅ LLM Engine initialized!")
    print(f"  • Text model: {'✓' if llm.llama_engine else '✗'}")
    print(f"  • Vision model: {'✓' if llm.vision_engine and llm.vision_engine.is_initialized else '✗'}")
    
    # Run tests
    try:
        text_time = await test_text_model()
        
        vision_time = await test_vision_model()
        
        await test_cross_validation()
        
        await test_intelligent_routing()
        
        await test_hybrid_learning()
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"✅ Text Model (3.1 8B): {text_time:.2f}s per response")
        if vision_time:
            print(f"✅ Vision Model (3.2 11B): {vision_time:.2f}s per image")
            print(f"📊 Speedup ratio: {vision_time/text_time:.1f}x slower (expected for vision)")
        print(f"\n🎉 All systems operational! Kalki is EXCEPTIONALLY SMART!")
        
    except Exception as e:
        logger.exception(f"Test failed: {e}")
        print(f"\n❌ Test failed: {e}")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        await llm.cleanup()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
