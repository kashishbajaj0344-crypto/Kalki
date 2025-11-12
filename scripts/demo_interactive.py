#!/usr/bin/env python3
"""
INTERACTIVE CONSTRUCTION COPILOT DEMO
Simulates being the first customer building a house
"""

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from modules.construction_copilot import ConstructionCopilot, ProjectState
from modules.foundation_steps import get_foundation_step
import json
import os

console = Console()

class InteractiveDemo:
    def __init__(self):
        self.copilot = ConstructionCopilot()
        self.project = None
        
    def welcome(self):
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]🏗️ KALKI CONSTRUCTION COPILOT[/bold cyan]\n"
            "[yellow]Your AI General Contractor[/yellow]\n\n"
            "[white]I'll guide you through building your dream home\n"
            "step by step, with expert advice at every stage.[/white]",
            border_style="cyan"
        ))
        console.print()
        
    def create_project(self):
        console.print("[bold yellow]📋 LET'S START YOUR PROJECT[/bold yellow]\n")
        
        # Get project details
        console.print("[cyan]First, tell me about your project:[/cyan]\n")
        
        project_name = Prompt.ask("Project name", default="My Dream House")
        location = Prompt.ask("Location (City, State)", default="Austin, TX")
        
        console.print("\n[cyan]What type of foundation?[/cyan]")
        console.print("  1. Slab-on-grade (most common, cheaper)")
        console.print("  2. Crawlspace (elevation, utilities access)")
        console.print("  3. Basement (extra living space, storage)")
        foundation_choice = Prompt.ask("Choose", choices=["1", "2", "3"], default="1")
        foundation_type = {
            "1": "slab",
            "2": "crawlspace", 
            "3": "basement"
        }[foundation_choice]
        
        console.print("\n[cyan]House size?[/cyan]")
        square_feet = int(Prompt.ask("Square footage", default="2000"))
        
        console.print("\n[cyan]Budget?[/cyan]")
        budget = float(Prompt.ask("Total budget ($)", default="300000"))
        
        console.print("\n[cyan]Timeline?[/cyan]")
        timeline_weeks = int(Prompt.ask("Target completion (weeks)", default="52"))
        
        # Create project
        self.project = ProjectState(
            phase="foundation",
            current_step=1,
            budget=budget,
            spent=0,
            timeline_weeks=timeline_weeks,
            weeks_elapsed=0,
            project_type=foundation_type,
            square_feet=square_feet,
            location=location,
            project_name=project_name
        )
        
        # Save project
        self.save_project()
        
        # Show summary
        console.print("\n" + "="*70)
        console.print(f"[bold green]✅ PROJECT CREATED: {project_name}[/bold green]")
        console.print("="*70)
        
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("📍 Location", location)
        table.add_row("🏠 Size", f"{square_feet:,} sq ft")
        table.add_row("🏗️ Foundation", foundation_type.title())
        table.add_row("💰 Budget", f"${budget:,.0f}")
        table.add_row("⏱️ Timeline", f"{timeline_weeks} weeks")
        table.add_row("📊 Current Phase", "Foundation (Step 1/11)")
        
        console.print(table)
        console.print()
        
    def show_next_step(self):
        """Show the next step in detail"""
        if self.project.phase == "foundation" and self.project.current_step <= 11:
            step = get_foundation_step(self.project.current_step, self.project)
            
            # Step header
            console.print("\n" + "="*70)
            console.print(f"[bold yellow]STEP {step.step_number}: {step.title.upper()}[/bold yellow]")
            console.print("="*70 + "\n")
            
            # Key info
            info_table = Table(show_header=False, box=None)
            info_table.add_column("", style="bold cyan", width=20)
            info_table.add_column("", style="white")
            
            cost = f"${step.estimated_cost:,.0f}" if step.estimated_cost > 0 else "$0 (DIY Labor)"
            info_table.add_row("💰 Cost:", cost)
            info_table.add_row("⏱️  Duration:", f"{step.estimated_duration_days} days")
            info_table.add_row("📊 Budget Used:", f"${self.project.spent:,.0f} / ${self.project.budget:,.0f} ({self.project.spent/self.project.budget*100:.1f}%)")
            
            if step.requires_professional:
                info_table.add_row("👷 Professional:", step.professional_type or "Recommended")
            else:
                info_table.add_row("🛠️  DIY:", "You can do this yourself")
            
            console.print(info_table)
            
            # Description (first 500 chars)
            console.print("\n[bold cyan]📖 WHAT TO DO:[/bold cyan]")
            desc_lines = step.description.split('\n')[:15]  # First 15 lines
            for line in desc_lines:
                console.print(line)
            console.print("\n[dim]...(more details in full guide)[/dim]")
            
            # Safety warnings
            if step.safety_warnings:
                console.print("\n[bold red]⚠️  SAFETY WARNINGS:[/bold red]")
                for i, warning in enumerate(step.safety_warnings[:3], 1):
                    console.print(f"   {i}. {warning}")
                if len(step.safety_warnings) > 3:
                    console.print(f"   [dim]...and {len(step.safety_warnings)-3} more[/dim]")
            
            # Materials needed
            if step.material_list:
                console.print("\n[bold green]🛒 MATERIALS NEEDED:[/bold green]")
                mat_table = Table(show_header=True)
                mat_table.add_column("Item", style="white")
                mat_table.add_column("Quantity", style="cyan")
                mat_table.add_column("Cost", style="green", justify="right")
                
                for mat in step.material_list[:5]:
                    item = mat['item']
                    qty = f"{mat['quantity']} {mat.get('unit', '')}"
                    cost = f"${mat.get('cost_per_unit', 0) * float(str(mat['quantity']).split()[0]):,.0f}"
                    mat_table.add_row(item, qty, cost)
                
                if len(step.material_list) > 5:
                    mat_table.add_row("[dim]...", f"[dim]{len(step.material_list)-5} more items[/dim]", "")
                
                console.print(mat_table)
            
            # Success criteria
            if step.success_criteria:
                console.print("\n[bold yellow]✅ SUCCESS CRITERIA:[/bold yellow]")
                for i, criteria in enumerate(step.success_criteria[:3], 1):
                    console.print(f"   {i}. {criteria}")
                if len(step.success_criteria) > 3:
                    console.print(f"   [dim]...and {len(step.success_criteria)-3} more[/dim]")
            
            console.print()
            return step
        else:
            console.print("[yellow]This phase is not yet implemented. Coming soon![/yellow]")
            return None
    
    def complete_step(self, step):
        """Mark step as complete and update project"""
        self.project.spent += step.estimated_cost
        self.project.weeks_elapsed += step.estimated_duration_days / 7
        self.project.current_step += 1
        
        # Check if phase complete
        if self.project.phase == "foundation" and self.project.current_step > 11:
            self.project.phase = "framing"
            self.project.current_step = 1
            console.print("[bold green]🎉 FOUNDATION PHASE COMPLETE![/bold green]")
            console.print("[yellow]Moving to Framing phase...[/yellow]\n")
        
        self.save_project()
    
    def show_progress(self):
        """Show overall project progress"""
        console.print("\n[bold cyan]📊 PROJECT PROGRESS[/bold cyan]\n")
        
        table = Table(show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Status", style="cyan")
        table.add_column("Details", style="white")
        
        # Budget
        budget_pct = (self.project.spent / self.project.budget * 100)
        budget_status = "✅ On Track" if budget_pct < 50 else "⚠️ Monitor"
        table.add_row(
            "Budget",
            budget_status,
            f"${self.project.spent:,.0f} / ${self.project.budget:,.0f} ({budget_pct:.1f}%)"
        )
        
        # Timeline
        timeline_pct = (self.project.weeks_elapsed / self.project.timeline_weeks * 100)
        timeline_status = "✅ On Track" if timeline_pct < 50 else "⚠️ Monitor"
        table.add_row(
            "Timeline",
            timeline_status,
            f"{self.project.weeks_elapsed:.1f} / {self.project.timeline_weeks} weeks ({timeline_pct:.1f}%)"
        )
        
        # Phase
        if self.project.phase == "foundation":
            phase_progress = f"Step {self.project.current_step}/11"
        else:
            phase_progress = f"Phase: {self.project.phase.title()}"
        
        table.add_row(
            "Current Phase",
            self.project.phase.title(),
            phase_progress
        )
        
        console.print(table)
        console.print()
    
    def save_project(self):
        """Save project state"""
        os.makedirs("data/projects", exist_ok=True)
        project_file = f"data/projects/{self.project.project_name.lower().replace(' ', '_')}.json"
        
        with open(project_file, 'w') as f:
            json.dump({
                'project_name': self.project.project_name,
                'location': self.project.location,
                'phase': self.project.phase,
                'current_step': self.project.current_step,
                'budget': self.project.budget,
                'spent': self.project.spent,
                'timeline_weeks': self.project.timeline_weeks,
                'weeks_elapsed': self.project.weeks_elapsed,
                'project_type': self.project.project_type,
                'square_feet': self.project.square_feet
            }, f, indent=2)
    
    def load_project(self, project_name):
        """Load existing project"""
        project_file = f"data/projects/{project_name.lower().replace(' ', '_')}.json"
        
        if os.path.exists(project_file):
            with open(project_file, 'r') as f:
                data = json.load(f)
            
            self.project = ProjectState(
                phase=data['phase'],
                current_step=data['current_step'],
                budget=data['budget'],
                spent=data['spent'],
                timeline_weeks=data['timeline_weeks'],
                weeks_elapsed=data['weeks_elapsed'],
                project_type=data['project_type'],
                square_feet=data['square_feet'],
                location=data['location'],
                project_name=data['project_name']
            )
            return True
        return False
    
    def run(self):
        """Main interactive loop"""
        self.welcome()
        
        # Check for existing project
        console.print("[cyan]Do you have an existing project to continue?[/cyan]")
        has_project = Confirm.ask("Load existing project?", default=False)
        
        if has_project:
            project_name = Prompt.ask("Project name")
            if not self.load_project(project_name):
                console.print("[red]Project not found. Creating new project...[/red]\n")
                self.create_project()
        else:
            self.create_project()
        
        # Main loop
        while True:
            console.print("\n" + "="*70)
            console.print("[bold cyan]WHAT WOULD YOU LIKE TO DO?[/bold cyan]")
            console.print("="*70 + "\n")
            
            console.print("  1. Show next step (detailed guidance)")
            console.print("  2. Complete current step (mark as done)")
            console.print("  3. View project progress")
            console.print("  4. Skip to next phase (testing)")
            console.print("  5. Exit")
            
            choice = Prompt.ask("\nChoose option", choices=["1", "2", "3", "4", "5"], default="1")
            
            if choice == "1":
                step = self.show_next_step()
                if step:
                    console.print("\n[dim]Press Enter to continue...[/dim]")
                    input()
            
            elif choice == "2":
                step = self.show_next_step()
                if step:
                    console.print()
                    if Confirm.ask(f"Mark step {step.step_number} as complete?", default=True):
                        self.complete_step(step)
                        console.print(f"[green]✅ Step {step.step_number} completed![/green]")
                        console.print(f"[cyan]Added ${step.estimated_cost:,.0f} to budget[/cyan]")
                        console.print(f"[cyan]Added {step.estimated_duration_days} days to timeline[/cyan]\n")
            
            elif choice == "3":
                self.show_progress()
                console.print("[dim]Press Enter to continue...[/dim]")
                input()
            
            elif choice == "4":
                console.print("\n[yellow]⚠️ TESTING MODE: Skipping to next phase[/yellow]")
                if self.project.phase == "foundation":
                    self.project.phase = "framing"
                    self.project.current_step = 1
                    console.print("[green]Moved to Framing phase[/green]\n")
                else:
                    console.print("[yellow]Framing and other phases coming soon![/yellow]\n")
                self.save_project()
            
            elif choice == "5":
                console.print("\n[bold green]Thanks for using Kalki Construction Copilot![/bold green]")
                console.print(f"[cyan]Your project has been saved: {self.project.project_name}[/cyan]")
                console.print("[white]Run this again anytime to continue building.[/white]\n")
                break


if __name__ == "__main__":
    demo = InteractiveDemo()
    demo.run()
