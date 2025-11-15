#!/usr/bin/env python3
"""
Construction Copilot Test Interface
Interactive interface to test all Construction Copilot features
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.construction_copilot_enhanced import EnhancedConstructionCopilot, ProjectState
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConstructionCopilotTester:
    """Interactive test interface for Construction Copilot"""
    
    def __init__(self):
        self.copilot = None
        self.current_project_id = None
        
    async def initialize(self):
        """Initialize the Construction Copilot"""
        print("\n" + "="*60)
        print("🏗️  CONSTRUCTION COPILOT TEST INTERFACE")
        print("="*60)
        print("\nInitializing Construction Copilot...")
        print("(This may take a moment as systems load lazily)\n")
        
        try:
            self.copilot = EnhancedConstructionCopilot()
            await self.copilot.initialize()
            print("✅ Construction Copilot initialized successfully!\n")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_menu(self):
        """Print main menu"""
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1.  Start New Project")
        print("2.  Answer Construction Question")
        print("3.  Get Recommendation (with WHY reasoning)")
        print("4.  Handle Unknown Situation (autonomous research)")
        print("5.  Validate Critical Decision (multi-agent)")
        print("6.  Answer with Automatic Diagrams")
        print("7.  Update Progress from Photo")
        print("8.  Predict Upcoming Issues")
        print("9.  Learn from Feedback")
        print("10. Optimize Workflow (self-evolution)")
        print("11. Generate Deliverable")
        print("12. Validate Deliverable (QA)")
        print("13. View Project Status")
        print("14. List All Projects")
        print("15. Load Project")
        print("0.  Exit")
        print("="*60)
    
    async def test_start_new_project(self):
        """Test starting a new project"""
        print("\n" + "-"*60)
        print("TEST: Start New Project")
        print("-"*60)
        
        print("\nEnter project details:")
        user_input = input("Project description (e.g., 'I want to build an ADU at 123 Main St, 800 sq ft'): ").strip()
        
        if not user_input:
            print("❌ No input provided")
            return
        
        print("\n🔄 Processing...")
        try:
            result = await self.copilot.start_new_project(user_input)
            
            print("\n✅ Project Created!")
            print(f"Project ID: {result.get('project_id', 'N/A')}")
            print(f"Project Type: {result.get('project_type', 'N/A')}")
            print(f"Current Stage: {result.get('current_stage', 'N/A')}")
            print(f"Timeline: {result.get('timeline_estimate_weeks', 'N/A')} weeks")
            print(f"Budget: ${result.get('budget_estimate', 0):,.2f}")
            
            if 'roadmap' in result:
                print("\n📋 Roadmap Preview:")
                roadmap = result['roadmap']
                if 'phases' in roadmap:
                    for phase in roadmap['phases'][:3]:  # Show first 3 phases
                        print(f"  • {phase.get('name', 'Phase')}")
            
            self.current_project_id = result.get('project_id')
            print(f"\n💡 Project ID saved: {self.current_project_id}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_answer_question(self):
        """Test answering a construction question"""
        print("\n" + "-"*60)
        print("TEST: Answer Construction Question")
        print("-"*60)
        
        question = input("\nEnter your question: ").strip()
        if not question:
            print("❌ No question provided")
            return
        
        print("\n🔄 Processing...")
        try:
            result = await self.copilot.answer_with_automatic_diagrams(
                query=question,
                context={"project_id": self.current_project_id} if self.current_project_id else {}
            )
            
            print("\n✅ Answer:")
            print(result.get('answer', result))
            
            if 'diagrams' in result and result['diagrams']:
                print(f"\n📊 Found {len(result['diagrams'])} relevant diagrams")
                for i, diagram in enumerate(result['diagrams'][:3], 1):
                    print(f"  {i}. {diagram.get('title', 'Diagram')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_recommendation(self):
        """Test getting a recommendation with WHY reasoning"""
        print("\n" + "-"*60)
        print("TEST: Get Recommendation (with WHY reasoning)")
        print("-"*60)
        
        question = input("\nWhat do you need a recommendation for? ").strip()
        if not question:
            print("❌ No question provided")
            return
        
        print("\n🔄 Processing with consciousness-powered reasoning...")
        try:
            result = await self.copilot.explain_recommendation_with_consciousness(
                user_question=question,
                context={"project_id": self.current_project_id} if self.current_project_id else {}
            )
            
            print("\n✅ Recommendation:")
            print(result.get('recommendation', result))
            
            if 'reasoning' in result:
                print("\n🧠 WHY Reasoning:")
                print(result['reasoning'])
            
            if 'confidence' in result:
                print(f"\n📊 Confidence: {result['confidence']:.0%}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_unknown_situation(self):
        """Test handling unknown situation with autonomous research"""
        print("\n" + "-"*60)
        print("TEST: Handle Unknown Situation (Autonomous Research)")
        print("-"*60)
        
        situation = input("\nDescribe an unknown/novel situation: ").strip()
        if not situation:
            print("❌ No situation provided")
            return
        
        print("\n🔄 Researching autonomously...")
        try:
            result = await self.copilot.handle_unknown_situation(
                situation=situation,
                context={"project_id": self.current_project_id} if self.current_project_id else {}
            )
            
            print("\n✅ Research Complete!")
            print(result.get('answer', result))
            
            if 'research_methods' in result:
                print(f"\n🔍 Research methods used: {', '.join(result['research_methods'])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_validate_decision(self):
        """Test multi-agent decision validation"""
        print("\n" + "-"*60)
        print("TEST: Validate Critical Decision (Multi-Agent)")
        print("-"*60)
        
        decision = input("\nDescribe the decision to validate: ").strip()
        if not decision:
            print("❌ No decision provided")
            return
        
        print("\n🔄 Validating with 3-agent consensus...")
        try:
            result = await self.copilot.validate_critical_decision(
                decision=decision,
                context={"project_id": self.current_project_id} if self.current_project_id else {}
            )
            
            print("\n✅ Validation Complete!")
            print(f"Consensus: {result.get('consensus', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 0):.0%}")
            
            if 'agent_opinions' in result:
                print("\n🤖 Agent Opinions:")
                for agent, opinion in result['agent_opinions'].items():
                    print(f"  {agent}: {opinion.get('opinion', 'N/A')[:100]}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_update_progress_photo(self):
        """Test updating progress from photo"""
        print("\n" + "-"*60)
        print("TEST: Update Progress from Photo")
        print("-"*60)
        
        if not self.current_project_id:
            print("❌ No active project. Please start a project first.")
            return
        
        photo_path = input("\nEnter path to site photo: ").strip()
        if not photo_path:
            print("❌ No photo path provided")
            return
        
        photo_path = Path(photo_path)
        if not photo_path.exists():
            print(f"❌ Photo not found: {photo_path}")
            return
        
        print("\n🔄 Analyzing photo with vision AI...")
        try:
            result = await self.copilot.auto_update_progress_from_photo(
                project_id=self.current_project_id,
                site_photo_path=str(photo_path)
            )
            
            print("\n✅ Progress Updated!")
            print(f"Completion: {result.get('new_completion_percentage', 0):.1f}%")
            print(f"Completed Items: {len(result.get('completed_items', []))}")
            
            if 'quality_issues' in result and result['quality_issues']:
                print(f"\n⚠️  Quality Issues Found: {len(result['quality_issues'])}")
                for issue in result['quality_issues'][:3]:
                    print(f"  • {issue.get('issue', 'Issue')}")
            
            if 'next_expected_work' in result:
                print(f"\n📋 Next Expected Work: {result['next_expected_work']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_predict_issues(self):
        """Test predicting upcoming issues"""
        print("\n" + "-"*60)
        print("TEST: Predict Upcoming Issues")
        print("-"*60)
        
        if not self.current_project_id:
            print("❌ No active project. Please start a project first.")
            return
        
        print("\n🔄 Analyzing project to predict issues...")
        try:
            result = await self.copilot.predict_upcoming_issues(
                project_id=self.current_project_id,
                horizon_days=30
            )
            
            print("\n✅ Predictions Complete!")
            predictions = result.get('predictions', [])
            print(f"Found {len(predictions)} potential issues:")
            
            for i, pred in enumerate(predictions[:5], 1):
                print(f"\n{i}. {pred.get('issue', 'Issue')}")
                print(f"   Probability: {pred.get('probability', 0):.0%}")
                print(f"   Impact: {pred.get('impact', 'N/A')}")
                if 'mitigation_strategies' in pred:
                    print(f"   Mitigation: {pred['mitigation_strategies'][0] if pred['mitigation_strategies'] else 'N/A'}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_learn_feedback(self):
        """Test learning from feedback"""
        print("\n" + "-"*60)
        print("TEST: Learn from Feedback")
        print("-"*60)
        
        feedback_type = input("\nFeedback type (positive/negative): ").strip().lower()
        if feedback_type not in ['positive', 'negative']:
            print("❌ Invalid feedback type")
            return
        
        recommendation = input("What recommendation was this about? ").strip()
        if not recommendation:
            print("❌ No recommendation provided")
            return
        
        print("\n🔄 Learning from feedback...")
        try:
            result = await self.copilot.learn_from_user_feedback(
                feedback_type=feedback_type,
                recommendation=recommendation,
                context={"project_id": self.current_project_id} if self.current_project_id else {}
            )
            
            print("\n✅ Learning Complete!")
            print(result.get('message', 'Feedback processed'))
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_optimize_workflow(self):
        """Test self-evolution workflow optimization"""
        print("\n" + "-"*60)
        print("TEST: Optimize Workflow (Self-Evolution)")
        print("-"*60)
        
        print("\n🔄 Analyzing workflow and optimizing...")
        try:
            result = await self.copilot.optimize_own_workflow()
            
            print("\n✅ Optimization Complete!")
            print(f"Improvements Identified: {len(result.get('improvements', []))}")
            
            for i, improvement in enumerate(result.get('improvements', [])[:3], 1):
                print(f"\n{i}. {improvement.get('description', 'Improvement')}")
                print(f"   Impact: {improvement.get('impact', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_generate_deliverable(self):
        """Test generating a deliverable"""
        print("\n" + "-"*60)
        print("TEST: Generate Deliverable")
        print("-"*60)
        
        if not self.current_project_id:
            print("❌ No active project. Please start a project first.")
            return
        
        print("\nAvailable deliverable types:")
        print("1. CAD Drawing")
        print("2. Blueprint")
        print("3. Bill of Materials (BOM)")
        print("4. Schedule")
        print("5. Cost Estimate")
        
        choice = input("\nSelect deliverable type (1-5): ").strip()
        type_map = {
            "1": "cad_drawing",
            "2": "blueprint",
            "3": "bill_of_materials",
            "4": "schedule",
            "5": "cost_estimate"
        }
        
        deliverable_type = type_map.get(choice)
        if not deliverable_type:
            print("❌ Invalid choice")
            return
        
        print("\n🔄 Generating deliverable...")
        try:
            from modules.professional_deliverable_generator import DeliverableType
            
            project = self.copilot.active_projects.get(self.current_project_id)
            if not project:
                print("❌ Project not found")
                return
            
            generator = await self.copilot.get_deliverable_generator()
            result = await generator.generate_deliverable(
                deliverable_type=DeliverableType[deliverable_type.upper()],
                project=project,
                specifications={"query": f"Generate {deliverable_type} for project"},
                output_format="pdf"
            )
            
            print(f"\n✅ Deliverable Generated!")
            print(f"Path: {result}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_validate_deliverable(self):
        """Test validating a deliverable"""
        print("\n" + "-"*60)
        print("TEST: Validate Deliverable (QA)")
        print("-"*60)
        
        deliverable_path = input("\nEnter path to deliverable: ").strip()
        if not deliverable_path:
            print("❌ No path provided")
            return
        
        deliverable_path = Path(deliverable_path)
        if not deliverable_path.exists():
            print(f"❌ File not found: {deliverable_path}")
            return
        
        print("\n🔄 Validating deliverable...")
        try:
            from modules.professional_deliverable_generator import DeliverableType
            from modules.quality_assurance_framework import QualityStandard
            
            qa_framework = await self.copilot.get_quality_framework()
            result = await qa_framework.validate_deliverable(
                deliverable=deliverable_path,
                deliverable_type=DeliverableType.TECHNICAL_DOCUMENT,
                quality_standard=QualityStandard.BUILDING_CODE,
                domain="construction"
            )
            
            print("\n✅ Validation Complete!")
            print(f"Valid: {result.valid}")
            print(f"Score: {result.overall_score:.0%}")
            print(f"Critical Issues: {result.critical_issues}")
            print(f"Major Issues: {result.major_issues}")
            print(f"Minor Issues: {result.minor_issues}")
            
            if result.recommendations:
                print("\n💡 Recommendations:")
                for rec in result.recommendations[:3]:
                    print(f"  • {rec}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def test_view_project_status(self):
        """View current project status"""
        print("\n" + "-"*60)
        print("TEST: View Project Status")
        print("-"*60)
        
        if not self.current_project_id:
            print("❌ No active project")
            return
        
        project = self.copilot.active_projects.get(self.current_project_id)
        if not project:
            print("❌ Project not found")
            return
        
        print(f"\n📊 Project Status:")
        print(f"ID: {project.project_id}")
        print(f"Type: {project.project_type}")
        print(f"Stage: {project.current_stage}")
        print(f"Address: {project.address}")
        print(f"Completion: {project.completion_percentage:.1f}%")
        print(f"Timeline: {project.timeline_estimate_weeks} weeks")
        print(f"Budget: ${project.budget_estimate:,.2f}")
        print(f"Spent: ${project.actual_budget_spent:,.2f}")
        print(f"Milestones Completed: {len(project.milestones_completed)}")
        print(f"Issues: {len(project.issues_encountered)}")
    
    async def test_list_projects(self):
        """List all projects"""
        print("\n" + "-"*60)
        print("TEST: List All Projects")
        print("-"*60)
        
        projects = self.copilot.active_projects
        if not projects:
            print("No projects found")
            return
        
        print(f"\nFound {len(projects)} project(s):\n")
        for project_id, project in projects.items():
            print(f"  • {project_id}: {project.project_type} - {project.current_stage} ({project.completion_percentage:.1f}%)")
    
    async def test_load_project(self):
        """Load a project"""
        print("\n" + "-"*60)
        print("TEST: Load Project")
        print("-"*60)
        
        project_id = input("\nEnter project ID: ").strip()
        if not project_id:
            print("❌ No project ID provided")
            return
        
        try:
            project = await self.copilot.load_project_state(project_id)
            if project:
                self.current_project_id = project_id
                print(f"✅ Project loaded: {project_id}")
            else:
                print("❌ Project not found")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    async def run(self):
        """Run the test interface"""
        if not await self.initialize():
            return
        
        while True:
            try:
                self.print_menu()
                choice = input("\nSelect option: ").strip()
                
                if choice == "0":
                    print("\n👋 Goodbye!")
                    break
                elif choice == "1":
                    await self.test_start_new_project()
                elif choice == "2":
                    await self.test_answer_question()
                elif choice == "3":
                    await self.test_recommendation()
                elif choice == "4":
                    await self.test_unknown_situation()
                elif choice == "5":
                    await self.test_validate_decision()
                elif choice == "6":
                    await self.test_answer_question()  # Uses automatic diagrams
                elif choice == "7":
                    await self.test_update_progress_photo()
                elif choice == "8":
                    await self.test_predict_issues()
                elif choice == "9":
                    await self.test_learn_feedback()
                elif choice == "10":
                    await self.test_optimize_workflow()
                elif choice == "11":
                    await self.test_generate_deliverable()
                elif choice == "12":
                    await self.test_validate_deliverable()
                elif choice == "13":
                    await self.test_view_project_status()
                elif choice == "14":
                    await self.test_list_projects()
                elif choice == "15":
                    await self.test_load_project()
                else:
                    print("❌ Invalid option")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                input("\nPress Enter to continue...")


async def main():
    """Main entry point"""
    tester = ConstructionCopilotTester()
    await tester.run()


if __name__ == "__main__":
    asyncio.run(main())

