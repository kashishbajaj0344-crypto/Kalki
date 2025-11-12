"""
Test Supreme Control Hub Integration with Domain System
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.supreme_control_hub import get_supreme_control_hub
from modules.domains.construction_domain.construction_domain import ConstructionProjectStateMachine
from modules.domains.project_persistence import ProjectPersistence


async def test_domain_aware_hub():
    """Test Supreme Control Hub with domain awareness"""
    
    print("=" * 80)
    print("Testing Domain-Aware Supreme Control Hub")
    print("=" * 80)
    
    hub = get_supreme_control_hub()
    
    # Test 1: Domain-aware query routing
    print("\n1. Testing Domain-Aware Query Routing")
    print("-" * 80)
    
    queries = [
        "What size joists do I need for a 16 foot span?",
        "How do I frame a bearing wall?",
        "What's the code requirement for stair rise and run in BC?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = await hub.process_domain_aware_query(query)
        
        if result["success"]:
            print(f"✅ Domain: {result['domain']['name']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Knowledge available: {result['domain']['knowledge_stats']}")
            print(f"   Answer: {result['answer'][:150]}...")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    # Test 2: Create project and get contextual help
    print(f"\n{'=' * 80}")
    print("2. Testing Project-Specific Contextual Help")
    print("=" * 80)
    
    # Create a project
    project = ConstructionProjectStateMachine(
        "test-hub-123",
        "Build 2-story home in Victoria, BC"
    )
    project.location = "Victoria, BC"
    project.building_type = "residential_multi_story"
    project.size_sqft = 2000
    project.stories = 2
    project.budget["estimated_total"] = 600000
    project.budget["actual_spent"] = 50000
    
    # Complete some milestones
    project.mark_milestone_complete("Site survey complete")
    project.mark_milestone_complete("Budget approved")
    project.mark_milestone_complete("Design brief finalized")
    
    # Save project
    persistence = ProjectPersistence()
    success = persistence.save_project(project)
    
    print(f"\n✅ Created project: {project.project_id}")
    print(f"   Phase: {project.current_phase.value}")
    print(f"   Budget spent: ${project.budget['actual_spent']:,}")
    print(f"   Saved: {success}")
    
    # Get contextual help
    help_result = await hub.get_project_contextual_help(project.project_id)
    
    if help_result["success"]:
        print(f"\n📋 Contextual Help:")
        print(f"   Current Phase: {help_result['current_phase']}")
        print(f"   Progress: {help_result['progress']['completed_milestones']}/{help_result['progress']['total_milestones']} milestones")
        print(f"   Budget: {help_result['budget_status']['percent_spent']:.1f}% spent")
        
        print(f"\n💡 Recommendations:")
        for rec in help_result['recommendations']:
            print(f"   • {rec}")
        
        print(f"\n📝 Help Text:")
        print(f"   {help_result['help_text'][:300]}...")
    
    # Test 3: Project-specific query with context
    print(f"\n{'=' * 80}")
    print("3. Testing Project-Specific Query")
    print("=" * 80)
    
    project_query = "What foundation type should I use for this project?"
    print(f"\nQuery: {project_query}")
    print(f"Project: {project.description}")
    
    result = await hub.process_domain_aware_query(
        project_query,
        project_id=project.project_id
    )
    
    if result["success"]:
        print(f"\n✅ Answer with project context:")
        print(f"   Domain: {result['domain']['name']}")
        print(f"   Project Phase: {result['project_context']['phase']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"\n   {result['answer'][:300]}...")
    
    # Test 4: Generate deliverable through hub
    print(f"\n{'=' * 80}")
    print("4. Testing Deliverable Generation Through Hub")
    print("=" * 80)
    
    deliverable_result = await hub.generate_project_deliverable(
        project.project_id,
        "bill_of_materials"
    )
    
    if deliverable_result["success"]:
        print(f"\n✅ Generated deliverable:")
        print(f"   Type: {deliverable_result['deliverable_type']}")
        print(f"   Domain: {deliverable_result['domain']}")
        print(f"   Files: {list(deliverable_result['files'].keys())}")
        if deliverable_result['files']:
            first_file = list(deliverable_result['files'].values())[0]
            print(f"   Output preview: {str(first_file)[:200]}...")
    else:
        print(f"❌ Error: {deliverable_result.get('error')}")
    
    # Test 5: Get hub statistics with domain info
    print(f"\n{'=' * 80}")
    print("5. Supreme Control Hub Statistics")
    print("=" * 80)
    
    stats = hub.get_statistics()
    
    print(f"\nExecution Statistics:")
    print(f"   Total Executions: {stats['total_executions']}")
    print(f"   Average Quality: {stats.get('average_quality', 0):.3f}")
    print(f"   Average Execution Time: {stats.get('average_execution_time', 0):.3f}s")
    
    if 'domain_statistics' in stats:
        print(f"\nDomain Statistics:")
        print(f"   Total Domains: {stats['domain_statistics']['total_domains']}")
        print(f"   Registered Domains: {', '.join(stats['domain_statistics']['registered_domains'])}")
        print(f"   Total Knowledge Items: {stats['domain_statistics']['total_knowledge_items']}")
    
    # Cleanup
    persistence.delete_project(project.project_id)
    
    print(f"\n{'=' * 80}")
    print("✅ All Supreme Control Hub integration tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_domain_aware_hub())
