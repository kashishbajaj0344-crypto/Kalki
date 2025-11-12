"""
Test the Construction Copilot - See "Real Life Copilot" in action!
"""

from modules.construction_copilot import ConstructionCopilot, ProjectState, ProjectPhase
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint

console = Console()

def test_foundation_phase():
    """
    Test the foundation phase - this is where Kalki guides a user 
    through building their foundation step-by-step
    """
    
    console.print("\n[bold cyan]🏗️  KALKI CONSTRUCTION COPILOT - FOUNDATION PHASE TEST[/bold cyan]\n")
    
    # Initialize copilot
    copilot = ConstructionCopilot()
    
    # Create a sample project at foundation phase
    project = ProjectState(
        phase=ProjectPhase.FOUNDATION,
        completed_steps=[],  # Starting fresh
        pending_tasks=[],
        budget_spent=25000.0,  # Already spent on permits, design
        budget_remaining=175000.0,  # $200k total budget
        timeline_days_elapsed=60,  # 2 months on design/permits
        timeline_days_remaining=240,  # 8 months to build
        hired_professionals=[
            {"role": "Architect", "name": "Jane Smith", "contact": "555-0100"},
            {"role": "Surveyor", "name": "Bob Johnson", "contact": "555-0200"}
        ],
        permits_obtained=["Building Permit", "Electrical Permit", "Plumbing Permit"],
        inspections_passed=[]
    )
    
    # Get next step from Kalki
    console.print("[yellow]📋 Asking Kalki: 'What should I do next?'[/yellow]\n")
    
    next_step = copilot.get_next_step(project)
    
    # Display the step beautifully
    console.print(Panel(
        f"[bold green]{next_step.title}[/bold green]",
        title="🎯 KALKI'S RECOMMENDATION",
        border_style="green"
    ))
    
    console.print("\n[bold]📖 DETAILED INSTRUCTIONS:[/bold]")
    console.print(Markdown(next_step.description))
    
    console.print(f"\n[bold]💡 WHY NOW:[/bold]")
    console.print(f"[dim]{next_step.why_now}[/dim]\n")
    
    console.print(f"[bold]💰 ESTIMATED COST:[/bold] ${next_step.estimated_cost:,.2f}")
    console.print(f"[bold]⏱️  ESTIMATED TIME:[/bold] {next_step.estimated_duration_days} days")
    console.print(f"[bold]👷 PROFESSIONAL NEEDED:[/bold] {next_step.professional_type if next_step.requires_professional else 'No - You can DIY!'}")
    console.print(f"[bold]📋 PERMIT REQUIRED:[/bold] {next_step.permit_type if next_step.requires_permit else 'No'}\n")
    
    if next_step.safety_warnings:
        console.print("[bold red]⚠️  SAFETY WARNINGS:[/bold red]")
        for warning in next_step.safety_warnings:
            console.print(f"   • {warning}")
        console.print()
    
    if next_step.material_list:
        console.print("[bold]🛠️  MATERIALS NEEDED:[/bold]")
        for material in next_step.material_list:
            console.print(f"   • {material}")
        console.print()
    
    if next_step.tool_list:
        console.print("[bold]🔧 TOOLS NEEDED:[/bold]")
        for tool in next_step.tool_list:
            console.print(f"   • {tool}")
        console.print()
    
    if next_step.success_criteria:
        console.print("[bold green]✅ SUCCESS CRITERIA:[/bold green]")
        for criterion in next_step.success_criteria:
            console.print(f"   • {criterion}")
        console.print()
    
    console.print("\n[bold cyan]💬 THIS IS THE VISION:[/bold cyan]")
    console.print("[dim]Kalki ALWAYS knows exactly what to do next.[/dim]")
    console.print("[dim]Cost estimates, time estimates, professional requirements, safety warnings...[/dim]")
    console.print("[dim]Everything a user needs to make informed decisions and take action.[/dim]")
    console.print("[dim]This is a 'Real Life Copilot' - not just information retrieval![/dim]\n")

def test_multiple_steps():
    """Show how Kalki guides through multiple steps"""
    
    console.print("\n[bold cyan]🔄 SIMULATING MULTI-STEP PROGRESSION[/bold cyan]\n")
    
    copilot = ConstructionCopilot()
    
    # Simulate user completing steps
    project = ProjectState(
        phase=ProjectPhase.FOUNDATION,
        completed_steps=[],
        pending_tasks=[],
        budget_spent=25000.0,
        budget_remaining=175000.0,
        timeline_days_elapsed=60,
        timeline_days_remaining=240,
        hired_professionals=[],
        permits_obtained=["Building Permit"],
        inspections_passed=[]
    )
    
    # Get first 3 steps
    for i in range(3):
        console.print(f"\n[bold yellow]STEP {i+1}:[/bold yellow]")
        next_step = copilot.get_next_step(project)
        console.print(f"   {next_step.title}")
        console.print(f"   Cost: ${next_step.estimated_cost:,.2f}")
        console.print(f"   Time: {next_step.estimated_duration_days} days")
        console.print(f"   Professional: {next_step.professional_type if next_step.requires_professional else 'DIY'}")
        
        # Simulate completing this step
        step_key = f"foundation_step_{i+1}_complete"
        project.completed_steps.append(step_key)
        project.budget_spent += next_step.estimated_cost or 0
        project.budget_remaining -= next_step.estimated_cost or 0
        project.timeline_days_elapsed += next_step.estimated_duration_days or 0
    
    console.print(f"\n[bold green]✅ PROGRESS UPDATE:[/bold green]")
    console.print(f"   Steps Completed: {len(project.completed_steps)}")
    console.print(f"   Budget Spent: ${project.budget_spent:,.2f}")
    console.print(f"   Days Elapsed: {project.timeline_days_elapsed}")
    console.print(f"\n[dim]Kalki tracks everything and always knows what's next![/dim]\n")

if __name__ == "__main__":
    test_foundation_phase()
    test_multiple_steps()
    
    console.print("\n[bold cyan]🚀 NEXT STEPS FOR DEVELOPMENT:[/bold cyan]")
    console.print("1. ✅ Foundation phase (Step 1 DONE - see above)")
    console.print("2. ⏳ Complete all 11 foundation steps")
    console.print("3. ⏳ Implement framing phase (12 steps)")
    console.print("4. ⏳ Add vision capabilities (GPT-4V for site photos)")
    console.print("5. ⏳ Expand to other domains (game dev, robotics)")
    console.print("\n[dim]Refer to KALKI_ROADMAP.md for complete development plan[/dim]\n")
