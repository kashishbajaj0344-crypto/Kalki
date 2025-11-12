#!/usr/bin/env python3
"""
KALKI CONSTRUCTION CHAT - Natural Language Interface
Talk to Kalki like you would talk to a general contractor
"""

import sys
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from modules.construction_copilot import ConstructionCopilot, ProjectState
from modules.foundation_steps import get_foundation_step

console = Console()


class KalkiChat:
    def __init__(self):
        self.copilot = ConstructionCopilot()
        self.project = None
        self.conversation_history = []
        
    def start(self):
        """Start the chat interface"""
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]🏗️  KALKI - Your AI General Contractor[/bold cyan]\n"
            "[white]Chat naturally about your construction project[/white]\n"
            "[dim]Type 'help' for commands, 'quit' to exit[/dim]",
            border_style="cyan"
        ))
        
        # Initial greeting
        self.show_message(
            "kalki",
            "Hi! I'm Kalki, your AI construction assistant. "
            "I can guide you through building your house from foundation to finish.\n\n"
            "To get started, tell me about your project:\n"
            "• What are you building? (house, garage, addition, etc.)\n"
            "• What's your budget?\n"
            "• What's your timeline?\n\n"
            "Or just say 'start new project' and I'll ask you step by step!"
        )
        
        # Main chat loop
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]")
                
                if not user_input.strip():
                    continue
                    
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self.show_message("kalki", "Goodbye! Good luck with your build! 🏗️")
                    break
                    
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'status':
                    self.show_project_status()
                    continue
                
                if user_input.lower() == 'next':
                    self.show_next_step()
                    continue
                
                # Process natural language input
                response = self.process_message(user_input)
                self.show_message("kalki", response)
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Chat interrupted. Type 'quit' to exit.[/yellow]")
                continue
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                continue
    
    def process_message(self, user_input: str) -> str:
        """Process user message and generate response"""
        lower_input = user_input.lower()
        
        # Project initialization
        if any(phrase in lower_input for phrase in ['start', 'new project', 'begin', 'let\'s go']):
            return self.start_new_project()
        
        # Budget questions
        if 'budget' in lower_input or 'cost' in lower_input or 'price' in lower_input:
            if self.project:
                return self.discuss_budget(user_input)
            else:
                return "Let me help you with budgeting! First, let's start your project. What are you building?"
        
        # Timeline questions
        if 'timeline' in lower_input or 'how long' in lower_input or 'when' in lower_input:
            if self.project:
                return self.discuss_timeline(user_input)
            else:
                return "I can help you plan your timeline! First, tell me - what are you building?"
        
        # Foundation questions
        if 'foundation' in lower_input:
            return self.discuss_foundation(user_input)
        
        # Framing questions
        if 'framing' in lower_input or 'frame' in lower_input or 'walls' in lower_input:
            return self.discuss_framing(user_input)
        
        # Phase questions
        if 'phase' in lower_input or 'step' in lower_input or 'what\'s next' in lower_input:
            if self.project:
                return self.show_next_step_detail()
            else:
                return "I'd love to walk you through the phases! First, let's start your project. Say 'start new project'."
        
        # General construction questions
        if '?' in user_input:
            return self.answer_question(user_input)
        
        # Default response
        if self.project:
            return (
                f"I hear you're interested in: '{user_input}'\n\n"
                "Let me help! Try asking:\n"
                "• 'What's my next step?'\n"
                "• 'Show me the foundation steps'\n"
                "• 'How much will this cost?'\n"
                "• 'How long will this take?'\n"
                "• 'What do I need for [step name]?'\n\n"
                "Or type 'help' to see all commands."
            )
        else:
            return (
                "I'm here to help you build! Let's get started.\n\n"
                "Say 'start new project' and I'll guide you through everything."
            )
    
    def start_new_project(self) -> str:
        """Initialize a new construction project"""
        if self.project:
            return "You already have a project started! Type 'status' to see where you are."
        
        # Create new project with all required fields
        from modules.construction_copilot import ProjectPhase
        self.project = ProjectState(
            phase=ProjectPhase.FOUNDATION,
            completed_steps=[],
            pending_tasks=[],
            budget_spent=0.0,
            budget_remaining=250000.0,
            timeline_days_elapsed=0,
            timeline_days_remaining=365,
            hired_professionals=[],
            permits_obtained=[],
            inspections_passed=[]
        )
        self.current_step = 1
        
        return (
            "🎉 Exciting! Let's build your house!\n\n"
            "**PROJECT INITIALIZED**\n"
            "• Phase: Foundation (Step 1 of 11)\n"
            "• Budget: $250,000 (we can adjust this)\n"
            "• Timeline: ~12 months\n\n"
            "**YOUR FIRST STEP:**\n"
            "You need to excavate the building site. This involves:\n"
            "• Calling 811 (utility locate) - **CRITICAL & FREE**\n"
            "• Marking foundation corners\n"
            "• Hiring excavator ($2,500, 1 day)\n"
            "• Digging to frost line\n"
            "• Installing perimeter drains\n\n"
            "**Want to:**\n"
            "• See full details? → type 'next'\n"
            "• Adjust budget? → ask 'what if my budget is $X?'\n"
            "• Ask questions? → just ask away!\n"
            "• See entire foundation? → ask 'show foundation steps'"
        )
    
    def show_next_step_detail(self) -> str:
        """Show detailed next step"""
        if not self.project:
            return "Start a project first! Say 'start new project'."
        
        try:
            step = get_foundation_step(self.current_step, self.project)
            
            cost = f"${step.estimated_cost:,.0f}" if step.estimated_cost > 0 else "DIY (materials only)"
            pro = "Yes - hire professional" if step.requires_professional else "No - you can DIY this"
            
            response = f"""
**STEP {step.step_number}: {step.title}**

**Why now:** {step.why_now}

**Cost:** {cost}
**Duration:** {step.estimated_duration_days} days
**Need professional?** {pro}

**Safety warnings:**
"""
            for warning in step.safety_warnings[:3]:
                response += f"⚠️  {warning}\n"
            
            if step.material_list:
                response += "\n**Key materials you'll need:**\n"
                for mat in step.material_list[:5]:
                    response += f"• {mat['item']}: {mat['quantity']} {mat.get('unit', '')}\n"
            
            response += "\n**Want to:**\n"
            response += "• See full instructions? → ask 'show me full details'\n"
            response += "• Mark complete? → say 'done with step [number]'\n"
            response += "• See what's next? → type 'what comes after this?'"
            
            return response
            
        except Exception as e:
            return f"I need to implement more steps! Currently have foundation (11 steps). Error: {e}"
    
    def discuss_foundation(self, user_input: str) -> str:
        """Discuss foundation phase"""
        return """
**FOUNDATION PHASE** (11 Steps, ~27 days, $17,350)

The foundation is the most critical part - everything builds on this!

**Here's what we'll do:**
1. **Site Excavation** ($2,500, 1 day) - Dig down to bearing soil
2. **Footing Layout** ($150, 1 day) - Mark exact corners
3. **Footing Forms** ($800, 2 days) - Build forms for concrete
4. **Rebar Installation** ($600, 1 day) - Steel reinforcement
5. **Pre-Pour Inspection** (FREE, 1 day) - Building inspector approval
6. **Concrete Pour** ($2,500, 1 day) - Pour footings
7. **Strip Forms** (FREE, 1 day) - Remove forms after curing
8. **Foundation Walls** ($8,000, 5 days) - Block or poured walls
9. **Waterproofing** ($2,000, 7 days) - Moisture barrier + drainage
10. **Backfill** ($800, 4 days) - Fill around walls, grade for drainage
11. **Final Inspection** (FREE, 3 days) - Get approval to frame

**Each step includes:**
✓ Detailed instructions
✓ Cost breakdowns
✓ Safety warnings
✓ Material lists
✓ Success criteria

**Want to:**
• Start foundation? → say 'begin foundation'
• See step 1 details? → type 'next'
• Ask about specific step? → ask 'tell me about [step name]'
"""
    
    def discuss_framing(self, user_input: str) -> str:
        """Discuss framing phase"""
        return """
**FRAMING PHASE** (Coming soon! Foundation first)

Framing is when your house really takes shape. You'll build:
• Floor system (joists, subfloor)
• Exterior walls
• Interior walls
• Roof structure
• Window/door openings

**But first:** You need to complete the foundation!

The foundation takes about 3-4 weeks and costs ~$17,350.
Once that's done and inspected, we'll start framing.

**Want to see foundation steps?** Say 'show foundation'
"""
    
    def discuss_budget(self, user_input: str) -> str:
        """Discuss project budget"""
        if not self.project:
            return "Let's start your project first! Say 'start new project' and we'll talk budget."
        
        total_budget = self.project.budget_spent + self.project.budget_remaining
        
        return f"""
**YOUR BUDGET: ${total_budget:,.0f}**

**Foundation Phase:** ~$17,350 (7% of budget)
• Site work: $2,500
• Footings: $4,050
• Walls: $8,000
• Waterproofing: $2,800

**Typical house budget breakdown:**
• Foundation: 7-10% (${total_budget * 0.07:,.0f} - ${total_budget * 0.10:,.0f})
• Framing: 15-20% (${total_budget * 0.15:,.0f} - ${total_budget * 0.20:,.0f})
• Mechanicals: 15-18% (${total_budget * 0.15:,.0f} - ${total_budget * 0.18:,.0f})
• Interior finish: 25-30% (${total_budget * 0.25:,.0f} - ${total_budget * 0.30:,.0f})
• Exterior finish: 15-20% (${total_budget * 0.15:,.0f} - ${total_budget * 0.20:,.0f})

**Need to adjust budget?** Tell me your actual budget and I'll recalculate!
"""
    
    def discuss_timeline(self, user_input: str) -> str:
        """Discuss project timeline"""
        if not self.project:
            return "Let's start your project first, then we'll plan the timeline!"
        
        total_days = self.project.timeline_days_elapsed + self.project.timeline_days_remaining
        weeks = total_days // 7
        months = weeks // 4
        
        return f"""
**YOUR TIMELINE: {weeks} weeks (~{months} months)**

**Foundation:** 4-5 weeks
• Excavation & footings: 1 week
• Foundation walls: 1 week
• Waterproofing & backfill: 1-2 weeks
• Inspections & curing time: 1 week

**Framing:** 4-6 weeks
**MEP Rough-in:** 3-4 weeks
**Insulation & Drywall:** 3-4 weeks
**Finish Work:** 8-12 weeks
**Final Inspection & Punch List:** 2-3 weeks

**You're currently at:** Week 0 (Foundation Step 1)

**Critical timeline factors:**
• Weather delays
• Inspection scheduling
• Material delivery
• Subcontractor availability

**Want to see weekly breakdown?** Ask me!
"""
    
    def answer_question(self, question: str) -> str:
        """Answer general construction questions"""
        lower_q = question.lower()
        
        if 'call 811' in lower_q or 'utility' in lower_q:
            return """
**CALLING 811 (Before You Dig)**

This is CRITICAL and FREE:
• Call 811 at least 3 business days before digging
• Utility companies mark underground lines (electric, gas, water, fiber)
• It's required by law
• Hitting a gas line = explosion risk
• Hitting electric = electrocution
• Fines up to $10,000+ if you don't call

**They mark:**
• Electrical lines (red)
• Gas lines (yellow)
• Water/sewer (blue)
• Telecom (orange)

**You MUST do this** before excavation. No exceptions!
"""
        
        if 'permit' in lower_q:
            return """
**BUILDING PERMITS**

You'll need permits for:
✓ Foundation work
✓ Framing
✓ Electrical
✓ Plumbing
✓ HVAC

**Cost:** Usually $1,500-3,000 total
**Process:** 
1. Submit plans to building department
2. Wait 2-4 weeks for approval
3. Pay fees
4. Post permit on site
5. Schedule inspections

**Inspections needed:**
• Footing inspection (before concrete)
• Foundation inspection (before backfill)
• Framing inspection (before drywall)
• Rough-in inspections (MEP)
• Final inspection (before move-in)

I'll tell you when to call for each inspection!
"""
        
        if 'diy' in lower_q or 'can i do' in lower_q:
            return """
**DIY vs HIRING PROS**

**You CAN DIY:**
✓ Footing layout ($150 vs $500)
✓ Footing forms ($800 vs $2,000)
✓ Rebar installation ($600 vs $1,500)
✓ Stripping forms (FREE vs $500)
✓ Waterproofing ($2,000 vs $4,000)
✓ Backfill (with rented compactor)

**You SHOULD hire pros:**
✓ Excavation (needs heavy equipment)
✓ Concrete pour (timing critical)
✓ Foundation walls (skilled labor)
✓ Electrical (code & safety)
✓ Plumbing (code & permits)

**DIY Savings:** $50,000-100,000 on full house!
**Time cost:** 2-3x longer than pros

I'll tell you for each step whether you can DIY or need help!
"""
        
        return f"""
Great question! I'm learning more every day.

**Try asking:**
• "Tell me about [foundation/framing/etc]"
• "How much does [step] cost?"
• "Can I DIY [task]?"
• "What's next?"
• "Show me the steps"

Or type 'help' for all commands!
"""
    
    def show_project_status(self):
        """Show current project status"""
        if not self.project:
            console.print("[yellow]No project started yet. Say 'start new project'![/yellow]")
            return
        
        console.print("\n[bold cyan]📊 YOUR PROJECT STATUS[/bold cyan]")
        console.print(f"Phase: {self.project.phase.value.title()}")
        console.print(f"Current Step: {self.current_step}")
        
        total_budget = self.project.budget_spent + self.project.budget_remaining
        console.print(f"Budget: ${total_budget:,.0f} (${self.project.budget_remaining:,.0f} remaining)")
        
        total_days = self.project.timeline_days_elapsed + self.project.timeline_days_remaining
        console.print(f"Timeline: {total_days // 7} weeks ({self.project.timeline_days_elapsed} days elapsed)")
        console.print(f"Steps Completed: {len(self.project.completed_steps)}")
        
        if self.project.completed_steps:
            console.print("\n[green]✅ Completed:[/green]")
            for step in self.project.completed_steps:
                console.print(f"  • Step {step}")
    
    def show_next_step(self):
        """Show what's next"""
        if not self.project:
            console.print("[yellow]Start a project first! Say 'start new project'[/yellow]")
            return
        
        try:
            step = get_foundation_step(self.current_step, self.project)
            console.print(f"\n[bold yellow]⏭️  NEXT: {step.title}[/bold yellow]")
            console.print(f"Cost: ${step.estimated_cost:,.0f} | Time: {step.estimated_duration_days} days")
            console.print("\nType 'What's my next step?' for full details!")
        except:
            console.print("[yellow]Working on building out more steps![/yellow]")
    
    def show_help(self):
        """Show help information"""
        help_text = """
**KALKI CHAT COMMANDS**

**Natural Language:**
• "Start new project" - Begin your build
• "What's next?" - See your next step
• "Show me foundation steps" - See all foundation steps
• "How much will this cost?" - Budget discussion
• "How long will this take?" - Timeline planning
• "Can I DIY [task]?" - DIY vs professional guidance
• "Tell me about [topic]" - Learn about anything

**Quick Commands:**
• status - See project status
• next - Quick view of next step
• help - Show this help
• quit - Exit chat

**Example conversations:**
→ "I want to build a house, what do I do first?"
→ "How much does foundation cost?"
→ "Can I do the rebar installation myself?"
→ "What tools do I need for step 3?"
→ "I'm nervous about calling 811, what is it?"

**Just talk naturally!** I understand construction questions.
"""
        console.print(Panel(help_text, title="Help", border_style="cyan"))
    
    def show_message(self, sender: str, message: str):
        """Display a chat message"""
        if sender == "kalki":
            console.print(f"\n[bold cyan]🏗️  Kalki:[/bold cyan]")
            # Try to render as markdown if it has markdown syntax
            if any(marker in message for marker in ['**', '•', '→', '✓', '✅', '⚠️']):
                console.print(Markdown(message))
            else:
                console.print(f"[white]{message}[/white]")
        else:
            console.print(f"\n[bold green]You:[/bold green] {message}")


def main():
    """Main entry point"""
    chat = KalkiChat()
    chat.start()


if __name__ == "__main__":
    main()
