"""
Test Enhanced Domain Inference Heuristics
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.domains.domain_registry import DomainRegistry


async def test_domain_inference():
    """Test improved domain inference with various queries"""
    
    print("=" * 80)
    print("Testing Enhanced Domain Inference Heuristics")
    print("=" * 80)
    
    registry = DomainRegistry()
    
    # Test cases: (query, expected_domain, category)
    test_cases = [
        # Construction - Building Types
        ("Build a garage in my backyard", "construction", "Building Types"),
        ("Renovate my kitchen", "construction", "Building Types"),
        ("Design a workshop addition", "construction", "Building Types"),
        ("How to build a deck", "construction", "Building Types"),
        
        # Construction - Structural
        ("What size joists for 16 foot span?", "construction", "Structural"),
        ("How to frame a bearing wall", "construction", "Structural"),
        ("Calculate beam size for roof", "construction", "Structural"),
        ("Foundation requirements for 2-story house", "construction", "Structural"),
        
        # Construction - Materials
        ("How much concrete for slab?", "construction", "Materials"),
        ("Best lumber for outdoor deck", "construction", "Materials"),
        ("Insulation R-value for BC climate", "construction", "Materials"),
        
        # Construction - Codes
        ("BC building code stair requirements", "construction", "Codes"),
        ("Do I need a permit for shed?", "construction", "Codes"),
        ("Inspection checklist for framing", "construction", "Codes"),
        
        # Construction - Cost/Budget
        ("How much to build 2000 sq ft home?", "construction", "Cost"),
        ("Estimate cost for garage addition", "construction", "Cost"),
        
        # Game Development
        ("Create a 2D platformer in Unity", "game_development", "Game Dev"),
        ("How to implement health bar", "game_development", "Game Dev"),
        ("Procedural dungeon generation algorithm", "game_development", "Game Dev"),
        ("Player movement in Godot", "game_development", "Game Dev"),
        
        # Robotics
        ("Build autonomous robot with Arduino", "robotics", "Robotics"),
        ("How to control servo motors", "robotics", "Robotics"),
        ("Implement SLAM for navigation", "robotics", "Robotics"),
        ("Inverse kinematics for robotic arm", "robotics", "Robotics"),
        
        # Aerospace
        ("Design a quadcopter drone", "aerospace", "Aerospace"),
        ("Calculate thrust-to-weight ratio", "aerospace", "Aerospace"),
        ("VTOL aircraft propulsion system", "aerospace", "Aerospace"),
        ("Wing design for small aircraft", "aerospace", "Aerospace"),
        
        # Power Systems
        ("Battery sizing for electric vehicle", "power_systems", "Power"),
        ("Solar panel system design", "power_systems", "Power"),
        ("Fuel cell efficiency", "power_systems", "Power"),
        
        # Multi-domain (should pick strongest)
        ("Build flying robot drone", "aerospace", "Multi-domain"),  # aerospace + robotics, aerospace wins
        ("Game with realistic building physics", "game_development", "Multi-domain"),  # game + construction
    ]
    
    print(f"\nTesting {len(test_cases)} inference scenarios...\n")
    
    results = {
        "correct": 0,
        "incorrect": 0,
        "no_match": 0
    }
    
    failures = []
    
    for i, (query, expected, category) in enumerate(test_cases, 1):
        inferred = await registry.infer_domain(query)
        
        if not inferred:
            status = "❌ NO MATCH"
            results["no_match"] += 1
            failures.append((query, expected, "none", category))
        elif inferred[0] == expected:
            status = "✅ CORRECT"
            results["correct"] += 1
        else:
            status = f"❌ WRONG ({inferred[0]})"
            results["incorrect"] += 1
            failures.append((query, expected, inferred[0], category))
        
        print(f"{i:2d}. [{category:15s}] {status}")
        print(f"    Query: {query}")
        print(f"    Expected: {expected}, Got: {inferred[0] if inferred else 'none'}")
        if len(inferred) > 1:
            print(f"    Also matched: {', '.join(inferred[1:])}")
        print()
    
    # Summary
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(test_cases)}")
    print(f"✅ Correct: {results['correct']} ({results['correct']/len(test_cases)*100:.1f}%)")
    print(f"❌ Incorrect: {results['incorrect']} ({results['incorrect']/len(test_cases)*100:.1f}%)")
    print(f"❌ No Match: {results['no_match']} ({results['no_match']/len(test_cases)*100:.1f}%)")
    
    if failures:
        print(f"\n{'=' * 80}")
        print("FAILURES")
        print("=" * 80)
        for query, expected, got, category in failures:
            print(f"[{category}] {query}")
            print(f"  Expected: {expected}, Got: {got}\n")
    
    # Category breakdown
    print(f"{'=' * 80}")
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    
    categories = {}
    for query, expected, category in test_cases:
        if category not in categories:
            categories[category] = {"total": 0, "correct": 0}
        categories[category]["total"] += 1
        
        inferred = await registry.infer_domain(query)
        if inferred and inferred[0] == expected:
            categories[category]["correct"] += 1
    
    for category, stats in sorted(categories.items()):
        accuracy = stats["correct"] / stats["total"] * 100
        print(f"{category:20s}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
    
    print(f"\n{'=' * 80}")
    if results['correct'] >= len(test_cases) * 0.9:  # 90% threshold
        print("✅ Domain inference working excellently!")
    elif results['correct'] >= len(test_cases) * 0.75:  # 75% threshold
        print("✅ Domain inference working well")
    else:
        print("⚠️ Domain inference needs improvement")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_domain_inference())
