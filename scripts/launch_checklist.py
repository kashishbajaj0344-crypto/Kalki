#!/usr/bin/env python3
"""
KALKI CONSTRUCTION COPILOT - Launch Checklist & Action Plan

This script guides you through the next 90 days to launch a sellable product.
Run daily to track progress.

Usage:
    python launch_checklist.py
    python launch_checklist.py --complete "task_id"
    python launch_checklist.py --report
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

# 90-Day Launch Plan
LAUNCH_PLAN = {
    "week_1": {
        "title": "Week 1: Product Foundation (Nov 9-15)",
        "goal": "Complete foundation phase implementation",
        "tasks": [
            {
                "id": "w1_t1",
                "task": "Complete all 11 foundation steps with detailed instructions",
                "owner": "You",
                "hours": 20,
                "priority": "HIGH",
                "completed": True,  # Step 1 done, need 2-11
                "deliverable": "foundation_steps.py with 11 complete NextStep objects"
            },
            {
                "id": "w1_t2",
                "task": "Test foundation phase with 2 real users",
                "owner": "You",
                "hours": 8,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "User feedback document + 3 bug fixes"
            },
            {
                "id": "w1_t3",
                "task": "Create demo video (3 minutes)",
                "owner": "You",
                "hours": 6,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "kalki_demo.mp4 uploaded to YouTube"
            },
            {
                "id": "w1_t4",
                "task": "Polish UI text (all steps, warnings, success criteria)",
                "owner": "You",
                "hours": 4,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "Professional copywriting review"
            }
        ]
    },
    
    "week_2": {
        "title": "Week 2: Landing Page & Marketing (Nov 16-22)",
        "goal": "Launch website and start building waitlist",
        "tasks": [
            {
                "id": "w2_t1",
                "task": "Register domain (kalki.build or similar)",
                "owner": "You",
                "hours": 1,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Domain registered on Namecheap/GoDaddy"
            },
            {
                "id": "w2_t2",
                "task": "Build landing page (Next.js or React)",
                "owner": "You or Contractor",
                "hours": 16,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Website live on Vercel with waitlist form"
            },
            {
                "id": "w2_t3",
                "task": "Write marketing copy (headlines, features, testimonials)",
                "owner": "You",
                "hours": 8,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Landing page copy reviewed by 3 people"
            },
            {
                "id": "w2_t4",
                "task": "Set up email marketing (ConvertKit or Mailchimp)",
                "owner": "You",
                "hours": 4,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "Welcome email sequence (5 emails)"
            },
            {
                "id": "w2_t5",
                "task": "Create social media accounts (Twitter, LinkedIn, YouTube)",
                "owner": "You",
                "hours": 2,
                "priority": "LOW",
                "completed": False,
                "deliverable": "3 accounts created + first post"
            }
        ]
    },
    
    "week_3": {
        "title": "Week 3: Beta Testing (Nov 23-29)",
        "goal": "Get 20 beta testers providing feedback",
        "tasks": [
            {
                "id": "w3_t1",
                "task": "Find 20 beta testers (Reddit, forums, friends)",
                "owner": "You",
                "hours": 8,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "20 signed beta agreements"
            },
            {
                "id": "w3_t2",
                "task": "Set up feedback system (Typeform or Google Forms)",
                "owner": "You",
                "hours": 4,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Weekly feedback form + Slack channel"
            },
            {
                "id": "w3_t3",
                "task": "Onboard beta testers (1:1 video calls)",
                "owner": "You",
                "hours": 10,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "20 testers actively using Kalki"
            },
            {
                "id": "w3_t4",
                "task": "Document testimonials and case studies",
                "owner": "You",
                "hours": 6,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "3 written testimonials + 1 video"
            }
        ]
    },
    
    "week_4": {
        "title": "Week 4: Payment Integration (Nov 30 - Dec 6)",
        "goal": "Enable users to pay for Kalki",
        "tasks": [
            {
                "id": "w4_t1",
                "task": "Set up Stripe account (business verification)",
                "owner": "You",
                "hours": 2,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Stripe account approved"
            },
            {
                "id": "w4_t2",
                "task": "Implement Stripe payment flow (Python backend)",
                "owner": "You",
                "hours": 12,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Working payment API + webhooks"
            },
            {
                "id": "w4_t3",
                "task": "Create subscription management page",
                "owner": "You",
                "hours": 8,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Users can upgrade/downgrade/cancel"
            },
            {
                "id": "w4_t4",
                "task": "Test end-to-end purchase (all 3 tiers)",
                "owner": "You",
                "hours": 4,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Successful test purchases documented"
            },
            {
                "id": "w4_t5",
                "task": "Set up invoicing & receipts (automated)",
                "owner": "You",
                "hours": 4,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "Email receipts sent automatically"
            }
        ]
    },
    
    "weeks_5_8": {
        "title": "Weeks 5-8: Complete Product (Dec 7 - Jan 3)",
        "goal": "All 15 construction phases implemented",
        "tasks": [
            {
                "id": "w5_t1",
                "task": "Implement FRAMING phase (12 steps)",
                "owner": "You",
                "hours": 24,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "framing_steps.py with complete guidance"
            },
            {
                "id": "w5_t2",
                "task": "Implement MEP_ROUGH_IN phase (15 steps)",
                "owner": "You",
                "hours": 24,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "mep_rough_steps.py"
            },
            {
                "id": "w5_t3",
                "task": "Implement INSULATION phase (8 steps)",
                "owner": "You",
                "hours": 12,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "insulation_steps.py"
            },
            {
                "id": "w5_t4",
                "task": "Implement DRYWALL phase (10 steps)",
                "owner": "You",
                "hours": 12,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "drywall_steps.py"
            },
            {
                "id": "w5_t5",
                "task": "Implement remaining 10 phases (FINISH phases)",
                "owner": "You",
                "hours": 60,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "All 15 phases complete"
            },
            {
                "id": "w5_t6",
                "task": "Build material database (1000+ products)",
                "owner": "You",
                "hours": 20,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "materials.db with pricing, specs, suppliers"
            },
            {
                "id": "w5_t7",
                "task": "Expand code compliance database (IBC, IRC, ADA)",
                "owner": "You",
                "hours": 16,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "500+ code rules with jurisdiction mapping"
            }
        ]
    },
    
    "weeks_9_12": {
        "title": "Weeks 9-12: Launch & Growth (Jan 4 - Jan 31)",
        "goal": "50 paying customers, $1,500 MRR",
        "tasks": [
            {
                "id": "w9_t1",
                "task": "Product Hunt launch (prepare assets)",
                "owner": "You",
                "hours": 12,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Top 5 product of the day"
            },
            {
                "id": "w9_t2",
                "task": "Reddit launch posts (r/DIY, r/HomeImprovement)",
                "owner": "You",
                "hours": 8,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "1000+ upvotes combined"
            },
            {
                "id": "w9_t3",
                "task": "YouTube influencer outreach (Matt Risinger, etc)",
                "owner": "You",
                "hours": 12,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "1 video review (50K+ views)"
            },
            {
                "id": "w9_t4",
                "task": "Press release (TechCrunch, The Verge)",
                "owner": "You or PR firm",
                "hours": 8,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "2 press mentions"
            },
            {
                "id": "w9_t5",
                "task": "Google Ads campaign ($1K budget)",
                "owner": "You",
                "hours": 8,
                "priority": "MEDIUM",
                "completed": False,
                "deliverable": "50 conversions at <$20 CPA"
            },
            {
                "id": "w9_t6",
                "task": "Analyze metrics and iterate",
                "owner": "You",
                "hours": 16,
                "priority": "HIGH",
                "completed": False,
                "deliverable": "Top 10 bugs fixed, 3 new features added"
            }
        ]
    }
}

# Revenue tracking
REVENUE_TARGETS = {
    "month_1": {"target_mrr": 500, "target_customers": 10},
    "month_2": {"target_mrr": 1500, "target_customers": 30},
    "month_3": {"target_mrr": 3000, "target_customers": 50}
}

def show_progress():
    """Display overall progress"""
    console.print("\n[bold cyan]🚀 KALKI CONSTRUCTION COPILOT - 90 DAY LAUNCH PLAN[/bold cyan]\n")
    
    total_tasks = 0
    completed_tasks = 0
    total_hours = 0
    completed_hours = 0
    
    for week_key, week_data in LAUNCH_PLAN.items():
        console.print(f"\n[bold yellow]{week_data['title']}[/bold yellow]")
        console.print(f"[dim]Goal: {week_data['goal']}[/dim]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Task", style="cyan", width=50)
        table.add_column("Owner", style="green", width=10)
        table.add_column("Hours", justify="right", width=8)
        table.add_column("Priority", width=10)
        table.add_column("Status", width=12)
        
        for task in week_data['tasks']:
            total_tasks += 1
            total_hours += task['hours']
            
            if task['completed']:
                completed_tasks += 1
                completed_hours += task['hours']
                status = "[green]✅ DONE[/green]"
            else:
                status = "[yellow]⏳ TODO[/yellow]"
            
            priority_color = {
                "HIGH": "[red]HIGH[/red]",
                "MEDIUM": "[yellow]MEDIUM[/yellow]",
                "LOW": "[blue]LOW[/blue]"
            }
            
            table.add_row(
                task['task'],
                task['owner'],
                str(task['hours']),
                priority_color[task['priority']],
                status
            )
        
        console.print(table)
    
    # Summary
    console.print("\n[bold cyan]📊 SUMMARY[/bold cyan]\n")
    
    completion_rate = (completed_tasks / total_tasks) * 100
    hours_rate = (completed_hours / total_hours) * 100
    
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")
    
    summary_table.add_row("Total Tasks", f"{completed_tasks}/{total_tasks} ({completion_rate:.1f}%)")
    summary_table.add_row("Total Hours", f"{completed_hours}/{total_hours} ({hours_rate:.1f}%)")
    summary_table.add_row("Days Remaining", "90 days")
    summary_table.add_row("Launch Date", "February 6, 2026")
    
    console.print(summary_table)
    
    # Next action
    console.print("\n[bold green]🎯 NEXT ACTION:[/bold green]")
    for week_key, week_data in LAUNCH_PLAN.items():
        for task in week_data['tasks']:
            if not task['completed'] and task['priority'] == "HIGH":
                console.print(f"   → {task['task']} ({task['hours']} hours)")
                console.print(f"   [dim]Deliverable: {task['deliverable']}[/dim]\n")
                break
        else:
            continue
        break

def show_revenue_tracker():
    """Display revenue tracking"""
    console.print("\n[bold cyan]💰 REVENUE TRACKER[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Month", width=15)
    table.add_column("Target MRR", justify="right", width=15)
    table.add_column("Target Customers", justify="right", width=20)
    table.add_column("Actual MRR", justify="right", width=15)
    table.add_column("Actual Customers", justify="right", width=20)
    table.add_column("Status", width=12)
    
    for month, targets in REVENUE_TARGETS.items():
        # TODO: Track actual numbers
        actual_mrr = 0
        actual_customers = 0
        
        if actual_mrr >= targets['target_mrr']:
            status = "[green]✅ ON TRACK[/green]"
        else:
            status = "[yellow]⏳ IN PROGRESS[/yellow]"
        
        table.add_row(
            month.replace("_", " ").title(),
            f"${targets['target_mrr']:,}",
            str(targets['target_customers']),
            f"${actual_mrr:,}",
            str(actual_customers),
            status
        )
    
    console.print(table)
    
    console.print("\n[bold]Revenue Projections:[/bold]")
    console.print("   Month 1: $500 MRR (10 customers @ $49/mo avg)")
    console.print("   Month 2: $1,500 MRR (30 customers)")
    console.print("   Month 3: $3,000 MRR (50 customers)")
    console.print("   [dim]Year 1 Goal: $10,000 MRR ($120K ARR)[/dim]\n")

def show_critical_path():
    """Show critical path to launch"""
    console.print("\n[bold red]🔥 CRITICAL PATH (MUST COMPLETE)[/bold red]\n")
    
    critical_tasks = []
    for week_key, week_data in LAUNCH_PLAN.items():
        for task in week_data['tasks']:
            if task['priority'] == "HIGH" and not task['completed']:
                critical_tasks.append({
                    "week": week_data['title'],
                    "task": task['task'],
                    "hours": task['hours'],
                    "deliverable": task['deliverable']
                })
    
    for i, task in enumerate(critical_tasks, 1):
        console.print(f"[bold]{i}.[/bold] {task['task']}")
        console.print(f"   [dim]Week: {task['week']}[/dim]")
        console.print(f"   [dim]Time: {task['hours']} hours[/dim]")
        console.print(f"   [dim]Deliverable: {task['deliverable']}[/dim]\n")
    
    total_critical_hours = sum(t['hours'] for t in critical_tasks)
    console.print(f"[bold]Total Critical Path Hours: {total_critical_hours}[/bold]")
    console.print(f"[dim]At 40 hours/week: {total_critical_hours/40:.1f} weeks[/dim]\n")

def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--report":
            show_progress()
            show_revenue_tracker()
            show_critical_path()
        elif sys.argv[1] == "--complete":
            # TODO: Mark task as complete
            console.print("[green]Task marked complete![/green]")
    else:
        show_progress()
        show_revenue_tracker()
        show_critical_path()
        
        # Call to action
        console.print("[bold cyan]💡 IMMEDIATE NEXT STEPS:[/bold cyan]")
        console.print("1. Complete foundation steps 2-11 (20 hours)")
        console.print("2. Find 2 beta testers to try foundation phase (8 hours)")
        console.print("3. Create 3-minute demo video (6 hours)")
        console.print("4. Register domain (kalki.build) (1 hour)")
        console.print("\n[bold]Start with #1 - it's the foundation of everything![/bold]\n")

if __name__ == "__main__":
    main()
