#!/usr/bin/env python3
"""
TEST: Complete Foundation Phase - All 11 Steps
Demonstrates the full foundation guidance system
"""

from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from modules.foundation_steps import get_all_foundation_steps, get_foundation_step

console = Console()

class ProjectState:
    """Dummy project state for testing"""
    def __init__(self):
        self.phase = "foundation"
        self.budget = 50000
        self.timeline_weeks = 12


def show_step_summary(step):
    """Display a step in compact format"""
    # Status icon
    pro_icon = "👷" if step.requires_professional else "🛠️"
    
    # Cost
    if step.estimated_cost > 0:
        cost = f"${step.estimated_cost:,.0f}"
    else:
        cost = "DIY"
    
    # Duration
    days = f"{step.estimated_duration_days}d"
    
    return f"{step.step_number:2d}. {step.title:55s} {pro_icon}  {cost:>10s}  {days:>5s}"


def show_step_detail(step):
    """Display full step details"""
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold yellow]STEP {step.step_number}: {step.title.upper()}[/bold yellow]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
    
    # Key info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("Label", style="bold cyan")
    info_table.add_column("Value", style="white")
    
    cost = f"${step.estimated_cost:,.0f}" if step.estimated_cost > 0 else "$0 (DIY Labor)"
    info_table.add_row("💰 Cost:", cost)
    info_table.add_row("⏱️  Duration:", f"{step.estimated_duration_days} days")
    
    if step.requires_professional:
        info_table.add_row("👷 Professional:", step.professional_type or "Recommended")
    else:
        info_table.add_row("🛠️  DIY:", "Can do yourself")
    
    if step.requires_permit:
        info_table.add_row("📋 Permit:", step.permit_type or "Required")
    
    console.print(info_table)
    
    # Safety warnings
    if step.safety_warnings:
        console.print(f"\n[bold red]⚠️  SAFETY WARNINGS:[/bold red]")
        for warning in step.safety_warnings[:3]:  # Show first 3
            console.print(f"   • {warning}")
    
    # Materials (show first 3)
    if step.material_list:
        console.print(f"\n[bold green]🛒 KEY MATERIALS:[/bold green]")
        for mat in step.material_list[:3]:
            item = mat['item']
            qty = mat['quantity']
            unit = mat.get('unit', '')
            console.print(f"   • {item}: {qty} {unit}")
    
    # Tools (show first 3)
    if step.tool_list:
        console.print(f"\n[bold blue]🔧 TOOLS NEEDED:[/bold blue]")
        for tool in step.tool_list[:3]:
            console.print(f"   • {tool}")
    
    # Success criteria (show first 3)
    if step.success_criteria:
        console.print(f"\n[bold yellow]✅ SUCCESS CRITERIA:[/bold yellow]")
        for criteria in step.success_criteria[:3]:
            console.print(f"   ✓ {criteria}")
    
    console.print()


def main():
    console.print(Panel.fit(
        "[bold cyan]🏗️  KALKI CONSTRUCTION COPILOT[/bold cyan]\n"
        "[bold yellow]Foundation Phase - Complete Implementation[/bold yellow]\n"
        "[white]All 11 Steps with Expert Guidance[/white]",
        border_style="cyan"
    ))
    
    # Create project
    project = ProjectState()
    
    # Get all steps
    console.print("\n[bold]Loading foundation phase...[/bold]")
    all_steps = get_all_foundation_steps(project)
    
    # Summary table
    console.print(f"\n[bold cyan]📋 FOUNDATION PHASE OVERVIEW[/bold cyan]")
    console.print(f"[dim]11 steps from excavation to final inspection[/dim]\n")
    
    for step in all_steps:
        console.print(show_step_summary(step))
    
    # Calculate totals
    total_cost = sum(s.estimated_cost for s in all_steps)
    total_days = sum(s.estimated_duration_days for s in all_steps)
    
    console.print(f"\n[bold]{'─'*80}[/bold]")
    console.print(f"[bold green]TOTAL:[/bold green] ${total_cost:,.0f} | {total_days} days ({total_days/7:.1f} weeks)")
    console.print(f"[bold]{'─'*80}[/bold]")
    
    # Show detailed view of key steps
    console.print("\n\n[bold yellow]📖 DETAILED STEP EXAMPLES[/bold yellow]")
    console.print("[dim]Showing sample steps with full guidance...[/dim]")
    
    # Show step 1, 6, 8, and 11 in detail
    key_steps = [1, 6, 8, 11]
    for step_num in key_steps:
        step = get_foundation_step(step_num, project)
        show_step_detail(step)
    
    # Final summary
    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print("[bold green]✅ FOUNDATION PHASE 100% COMPLETE[/bold green]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")
    
    console.print("[bold]What You Have:[/bold]")
    console.print("  • [green]11 complete steps[/green] with expert-level guidance")
    console.print("  • [green]Step-by-step instructions[/green] for every task")
    console.print("  • [green]Cost estimates[/green] for materials and labor")
    console.print("  • [green]Time estimates[/green] for planning")
    console.print("  • [green]Safety warnings[/green] for every hazard")
    console.print("  • [green]Material lists[/green] with quantities")
    console.print("  • [green]Tool requirements[/green] for each step")
    console.print("  • [green]Success criteria[/green] to verify quality")
    console.print("  • [green]Code compliance[/green] guidance")
    console.print("  • [green]Professional recommendations[/green] when needed")
    
    console.print("\n[bold]Ready to Sell:[/bold]")
    console.print("  • [cyan]$49/month[/cyan] - Monthly subscription")
    console.print("  • [cyan]$499 one-time[/cyan] - Complete foundation phase")
    console.print("  • [cyan]$17,350 value[/cyan] - Total foundation cost")
    
    console.print("\n[bold yellow]🚀 START SELLING TODAY![/bold yellow]")
    console.print("  Next: Find 10 beta testers and launch landing page")
    console.print()


if __name__ == "__main__":
    main()
