"""
Test Domain System

Validates the multi-domain architecture:
- Domain auto-discovery
- Domain registry loading
- Construction domain functionality
- Domain inference from queries
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import asyncio
from modules.domains.domain_registry import DomainRegistry


async def test_domain_registry():
    """Test domain registry initialization"""
    print("=" * 60)
    print("KALKI Multi-Domain System Test")
    print("=" * 60)
    print()
    
    # Initialize registry
    print("📋 Initializing Domain Registry...")
    registry = DomainRegistry()
    print()
    
    # List discovered domains
    print("🔍 Discovered Domains:")
    domains = registry.list_domains()
    for domain_name in domains:
        info = registry.get_domain_info(domain_name)
        if info:
            print(f"  ✅ {domain_name}")
            print(f"     Description: {info['description']}")
            print(f"     Knowledge Items: {info['knowledge_total']}")
            if info['knowledge_stats']:
                for k, v in info['knowledge_stats'].items():
                    print(f"       - {k}: {v}")
            print(f"     Deliverables: {', '.join(info['deliverables'][:3])}...")
    print()
    
    # Test domain inference
    print("🧠 Testing Domain Inference:")
    test_queries = [
        "Design me a 3-story house in BC",
        "Create a 2D platformer game",
        "Build a robot that can navigate autonomously",
        "Design a hydrogen fuel cell powered flying suit",
        "How do I calculate structural loads for a beam?"
    ]
    
    for query in test_queries:
        inferred = await registry.infer_domain(query)
        print(f"  Query: '{query}'")
        print(f"  → Inferred domains: {inferred if inferred else 'No domains matched'}")
        print()
    
    # Get overall statistics
    print("📊 Registry Statistics:")
    stats = registry.get_statistics()
    print(f"  Total Domains: {stats['total_domains']}")
    print(f"  Loaded Domains: {stats['loaded_domains']}")
    print(f"  Total Knowledge Items: {stats['total_knowledge_items']}")
    print()
    
    # Test construction domain specifically
    print("🏗️ Testing Construction Domain:")
    construction = registry.get_domain("construction")
    if construction:
        print(f"  Name: {construction.name}")
        print(f"  Description: {construction.description}")
        
        # Get knowledge extractors
        extractors = construction.get_knowledge_extractors()
        print(f"  Knowledge Extractors: {len(extractors)}")
        for ext in extractors:
            print(f"    - {ext.name}: {ext.description}")
        
        # Get deliverables
        deliverables = construction.get_deliverable_types()
        print(f"  Deliverables: {len(deliverables)}")
        for deliv in deliverables:
            print(f"    - {deliv.name}: {deliv.description}")
        
        # Test project creation
        print("\n  Creating test project...")
        project = await construction.create_project(
            "3-story home in Sechelt, BC",
            requirements={
                "location": "Sechelt, BC",
                "building_type": "single_family_residential",
                "size_sqft": 2500,
                "stories": 3
            }
        )
        print(f"    ✅ Project created: {project.project_id}")
        print(f"    Current Phase: {project.current_phase}")
        print(f"    Location: {project.location}")
        print(f"    Building Type: {project.building_type}")
        
        # Test requirement validation
        print("\n  Validating requirements...")
        validation = await construction.validate_requirements({
            "location": "Sechelt, BC",
            "building_type": "single_family_residential",
            "size_sqft": 2500
        })
        print(f"    Valid: {validation.valid}")
        if validation.errors:
            print(f"    Errors: {validation.errors}")
        if validation.warnings:
            print(f"    Warnings: {validation.warnings}")
        
        # Test complexity estimation
        print("\n  Estimating complexity...")
        complexity = await construction.estimate_complexity(project)
        print(f"    Overall Score: {complexity.overall_score:.1f}/100")
        print(f"    Time Estimate: {complexity.time_estimate_days} days")
        print(f"    Cost Estimate: ${complexity.cost_estimate_usd:,.0f}")
        print(f"    Risk Level: {complexity.risk_level}")
    else:
        print("  ❌ Construction domain not loaded")
    
    print()
    print("=" * 60)
    print("✅ Domain System Test Complete")
    print("=" * 60)


async def test_multi_domain_query():
    """Test querying across multiple domains"""
    print("\n" + "=" * 60)
    print("Multi-Domain Query Test")
    print("=" * 60)
    print()
    
    registry = DomainRegistry()
    
    # Test query that could span multiple domains
    query = "I need a flying device powered by alternative energy"
    print(f"Query: '{query}'")
    print()
    
    inferred_domains = await registry.infer_domain(query)
    print(f"Inferred domains: {inferred_domains}")
    print()
    
    if inferred_domains:
        print("Domain capabilities for this query:")
        for domain_name in inferred_domains:
            info = registry.get_domain_info(domain_name)
            if info:
                print(f"\n  {domain_name.upper()}:")
                print(f"    {info['description']}")
                print(f"    Can deliver: {', '.join(info['deliverables'][:3])}...")
    
    print()


if __name__ == "__main__":
    asyncio.run(test_domain_registry())
    asyncio.run(test_multi_domain_query())
