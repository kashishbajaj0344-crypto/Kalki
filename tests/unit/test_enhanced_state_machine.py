"""
Test Enhanced Construction Project State Machine
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.domains.construction_domain.construction_domain import (
    ConstructionProjectStateMachine,
    ConstructionPhase
)


async def test_state_machine():
    """Test the enhanced state machine features"""
    
    print("=" * 80)
    print("Testing Enhanced Construction Project State Machine")
    print("=" * 80)
    
    # Create project
    project = ConstructionProjectStateMachine(
        "test-project-123",
        "Build 3-story home in Vancouver"
    )
    
    # Set initial requirements
    project.location = "Vancouver, BC"
    project.building_type = "residential_multi_story"
    project.size_sqft = 2500
    project.stories = 3
    project.budget["estimated_total"] = 800000
    
    print("\n1. Initial Project Status")
    print("-" * 80)
    progress = project.get_phase_progress()
    print(f"Phase: {progress['phase']}")
    print(f"Milestones: {progress['completed_milestones']}/{progress['total_milestones']} ({progress['percent_complete']:.1f}%)")
    
    budget = project.get_budget_status()
    print(f"\nBudget: ${budget['estimated_total']:,.2f}")
    print(f"With Contingency: ${budget['budget_with_contingency']:,.2f}")
    print(f"Spent: ${budget['actual_spent']:,.2f}")
    print(f"Status: {budget['status']}")
    
    # Mark some milestones complete
    print("\n2. Completing Requirements Phase Milestones")
    print("-" * 80)
    project.mark_milestone_complete("Site survey complete")
    project.mark_milestone_complete("Budget approved")
    project.mark_milestone_complete("Design brief finalized")
    
    progress = project.get_phase_progress()
    print(f"Progress: {progress['completed_milestones']}/{progress['total_milestones']} ({progress['percent_complete']:.1f}%)")
    
    # Try to advance to design
    print("\n3. Advancing to Design Phase")
    print("-" * 80)
    validation = await project.validate_phase_complete(ConstructionPhase.REQUIREMENTS)
    print(f"Validation: {'✅ PASS' if validation.valid else '❌ FAIL'}")
    if validation.warnings:
        print(f"Warnings: {', '.join(validation.warnings)}")
    
    success = await project.advance_phase(ConstructionPhase.DESIGN)
    print(f"Phase advancement: {'✅ Success' if success else '❌ Failed'}")
    print(f"Current phase: {project.current_phase.value}")
    
    # Track some spending
    print("\n4. Budget Tracking")
    print("-" * 80)
    project.update_budget("design_fees", 50000, is_actual=True)
    project.update_budget("permits", 5000, is_actual=True)
    
    budget = project.get_budget_status()
    print(f"Spent: ${budget['actual_spent']:,.2f}")
    print(f"Remaining: ${budget['remaining']:,.2f}")
    print(f"Percent spent: {budget['percent_spent']:.1f}%")
    print(f"Status: {budget['status']}")
    
    # Test contextual help
    print("\n5. Contextual Help")
    print("-" * 80)
    help_text = await project.get_contextual_help("what should I do next?")
    print(help_text)
    
    # Test serialization
    print("\n6. JSON Serialization")
    print("-" * 80)
    project_dict = project.to_dict()
    print(f"Serialized keys: {list(project_dict.keys())}")
    print(f"Phase history entries: {len(project_dict['phase_history'])}")
    print(f"Budget entries: {list(project_dict['budget'].keys())}")
    
    # Test deserialization
    restored = ConstructionProjectStateMachine.from_dict(project_dict)
    print(f"\nRestored project:")
    print(f"  Current phase: {restored.current_phase.value}")
    print(f"  Location: {restored.location}")
    print(f"  Budget spent: ${restored.budget['actual_spent']:,.2f}")
    print(f"  Phase history: {len(restored.phase_history)} entries")
    
    # Simulate advancing through framing with budget tracking
    print("\n7. Simulating Project Progress")
    print("-" * 80)
    
    # Complete design milestones
    restored.mark_milestone_complete("Schematic design approved")
    restored.mark_milestone_complete("Construction drawings complete")
    restored.mark_milestone_complete("Structural calculations verified")
    
    # Advance through phases
    phases_to_complete = [
        ConstructionPhase.PERMIT_PREP,
        ConstructionPhase.FOUNDATION,
        ConstructionPhase.FRAMING
    ]
    
    for next_phase in phases_to_complete:
        # Complete critical milestones for current phase
        current_milestones = restored.milestones.get(restored.current_phase, [])
        for milestone in current_milestones:
            restored.mark_milestone_complete(milestone["name"])
        
        # Add some spending
        phase_cost = 100000
        restored.update_budget(f"{restored.current_phase.value}_costs", phase_cost, is_actual=True)
        
        # Try to advance
        validation = await restored.validate_phase_complete(restored.current_phase)
        if validation.valid:
            await restored.advance_phase(next_phase)
            print(f"✅ Advanced to {next_phase.value}")
        else:
            print(f"❌ Cannot advance: {', '.join(validation.errors)}")
    
    # Final status
    print("\n8. Final Project Status")
    print("-" * 80)
    print(f"Current phase: {restored.current_phase.value}")
    print(f"Phases completed: {len(restored.phase_history)}")
    
    budget = restored.get_budget_status()
    print(f"\nBudget Status:")
    print(f"  Estimated: ${budget['estimated_total']:,.2f}")
    print(f"  With Contingency (10%): ${budget['budget_with_contingency']:,.2f}")
    print(f"  Actual Spent: ${budget['actual_spent']:,.2f}")
    print(f"  Remaining: ${budget['remaining']:,.2f}")
    print(f"  Status: {budget['status']} ({budget['percent_spent']:.1f}% spent)")
    
    if budget['by_phase']:
        print(f"\nSpending by Phase:")
        for phase, amounts in budget['by_phase'].items():
            print(f"  {phase}: ${amounts['actual']:,.2f}")
    
    progress = restored.get_phase_progress()
    print(f"\nCurrent Phase Progress:")
    print(f"  {progress['completed_milestones']}/{progress['total_milestones']} milestones ({progress['percent_complete']:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_state_machine())
