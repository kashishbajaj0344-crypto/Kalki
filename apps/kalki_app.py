#!/usr/bin/env python3
"""
KALKI CONSTRUCTION APP - WITH AI BRAIN 🧠
Beautiful web-based chat interface powered by Llama 3.1 8B
Talk to Kalki AI like a chatbot - your AI general contractor
"""

import streamlit as st
from datetime import datetime
from modules.construction_copilot import ProjectState, ProjectPhase
from modules.foundation_steps import get_foundation_step, get_all_foundation_steps
from modules.llm import get_llm_engine, initialize_llm_engine
import json
import asyncio

# Page config
st.set_page_config(
    page_title="Kalki - AI General Contractor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful chat interface
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .chat-container {
        background: white;
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .user-message {
        background: #667eea;
        color: white;
        padding: 15px;
        border-radius: 18px 18px 5px 18px;
        margin: 10px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kalki-message {
        background: #f0f2f6;
        color: #1f2937;
        padding: 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 70%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .welcome-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: #667eea;
        color: white;
        border-radius: 25px;
        padding: 10px 30px;
        border: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: #764ba2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'project' not in st.session_state:
    st.session_state.project = None
    
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

if 'welcome_shown' not in st.session_state:
    st.session_state.welcome_shown = False


def init_project():
    """Initialize a new construction project"""
    st.session_state.project = ProjectState(
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
    st.session_state.current_step = 1
    return "✅ Project initialized! You're starting with the foundation phase."


def process_message(user_input: str) -> str:
    """Process user message and generate Kalki's response"""
    lower_input = user_input.lower()
    
    # Project initialization
    if any(phrase in lower_input for phrase in ['start', 'new project', 'begin', 'let\'s go', 'get started']):
        if st.session_state.project:
            return "You already have a project started! Check your status in the sidebar."
        init_project()
        return """🎉 **Exciting! Let's build your house!**

**PROJECT STARTED**
- Phase: Foundation (Step 1 of 11)
- Budget: $250,000
- Timeline: ~12 months

**YOUR FIRST STEP: Site Excavation**

You need to excavate the building site. Here's what's involved:

**Key Tasks:**
• Call 811 (utility locate) - **CRITICAL & FREE & REQUIRED BY LAW**
• Mark foundation corners with surveyor
• Hire excavator ($2,500, 1 day)
• Dig to frost line depth
• Install perimeter drains

**Safety First:** ⚠️
- Never dig without calling 811 first!
- Hitting gas line = explosion risk
- Hitting electric = electrocution
- Fines up to $10,000 if you skip this

Want to see full details? Just ask "tell me more about excavation" or "what's step 2?"
"""
    
    # Status check
    if 'status' in lower_input or 'where am i' in lower_input or 'progress' in lower_input:
        if not st.session_state.project:
            return "You haven't started a project yet! Say **'start new project'** to begin."
        
        total_budget = st.session_state.project.budget_spent + st.session_state.project.budget_remaining
        return f"""📊 **YOUR PROJECT STATUS**

**Current Phase:** Foundation
**Current Step:** {st.session_state.current_step} of 11
**Budget:** ${total_budget:,.0f} (${st.session_state.project.budget_remaining:,.0f} remaining)
**Timeline:** {(st.session_state.project.timeline_days_elapsed + st.session_state.project.timeline_days_remaining) // 7} weeks
**Steps Completed:** {len(st.session_state.project.completed_steps)}

Next: Type **'what's next'** to see your next step!
"""
    
    # Next step
    if 'next' in lower_input or 'what do i do' in lower_input or 'what now' in lower_input:
        if not st.session_state.project:
            return "Start a project first! Say **'start new project'**"
        
        try:
            step = get_foundation_step(st.session_state.current_step, st.session_state.project)
            cost = f"${step.estimated_cost:,.0f}" if step.estimated_cost > 0 else "DIY"
            pro = "👷 Hire Professional" if step.requires_professional else "🛠️ You can DIY"
            
            response = f"""**STEP {step.step_number}: {step.title}**

**Why now:** {step.why_now}

**⏱️ Time:** {step.estimated_duration_days} days
**💰 Cost:** {cost}
**👨‍🔧 Professional?** {pro}

**⚠️ Safety Warnings:**
"""
            for warning in step.safety_warnings[:3]:
                response += f"• {warning}\n"
            
            if step.material_list:
                response += "\n**🛒 Key Materials:**\n"
                for mat in step.material_list[:5]:
                    response += f"• {mat['item']}: {mat['quantity']} {mat.get('unit', '')}\n"
            
            return response
        except Exception as e:
            return f"Working on more steps! Currently have foundation (11 steps). {str(e)}"
    
    # Foundation overview
    if 'foundation' in lower_input or 'all steps' in lower_input or 'show steps' in lower_input:
        return """**🏗️ FOUNDATION PHASE** (11 Steps, 27 days, $17,350)

The foundation is the most critical part - everything builds on this!

**Steps:**
1. **Site Excavation** ($2,500, 1d) - Dig to bearing soil
2. **Footing Layout** ($150, 1d) - Mark corners precisely
3. **Footing Forms** ($800, 2d) - Build concrete forms
4. **Rebar Installation** ($600, 1d) - Steel reinforcement
5. **Pre-Pour Inspection** (FREE, 1d) - Inspector approval
6. **Concrete Pour** ($2,500, 1d) - Pour footings
7. **Strip Forms** (FREE, 1d) - Remove forms
8. **Foundation Walls** ($8,000, 5d) - Block/concrete walls
9. **Waterproofing** ($2,000, 7d) - Moisture barrier
10. **Backfill** ($800, 4d) - Fill & grade
11. **Final Inspection** (FREE, 3d) - Foundation approval

Each step includes detailed instructions, safety warnings, material lists, and success criteria!

Want details on any step? Just ask "tell me about step [number]"
"""
    
    # Budget questions
    if 'budget' in lower_input or 'cost' in lower_input or 'how much' in lower_input:
        if not st.session_state.project:
            return "Let's start your project first! Say **'start new project'**"
        
        total_budget = st.session_state.project.budget_spent + st.session_state.project.budget_remaining
        return f"""💰 **BUDGET BREAKDOWN**

**Your Total Budget:** ${total_budget:,.0f}

**Foundation Phase:** $17,350 (7%)
• Site work: $2,500
• Footings: $4,050  
• Walls: $8,000
• Waterproofing: $2,800

**Full House Estimate:**
• Foundation: 7-10%
• Framing: 15-20%
• Mechanicals (MEP): 15-18%
• Interior Finish: 25-30%
• Exterior Finish: 15-20%

**DIY Savings Potential:** 30-60% on labor costs!

Want to adjust your budget? Just tell me your actual budget!
"""
    
    # Timeline questions
    if 'timeline' in lower_input or 'how long' in lower_input or 'when' in lower_input:
        return """⏰ **PROJECT TIMELINE**

**Foundation:** 4-5 weeks ✓ (You are here!)
• Excavation: 1 week
• Footings: 1 week  
• Walls & waterproofing: 2-3 weeks

**Framing:** 4-6 weeks
**MEP Rough-in:** 3-4 weeks
**Insulation & Drywall:** 3-4 weeks
**Finish Work:** 8-12 weeks
**Final Inspection:** 2-3 weeks

**Total:** ~10-12 months from start to move-in

Weather, inspections, and material delivery can affect timing. I'll guide you through scheduling!
"""
    
    # Help
    if 'help' in lower_input:
        return """**🤖 HOW TO USE KALKI**

Just chat naturally! I understand questions like:

**Getting Started:**
• "Start new project"
• "What do I do first?"

**Navigation:**
• "What's next?"
• "Show me all foundation steps"
• "Tell me about step 3"

**Questions:**
• "How much will this cost?"
• "Can I do this myself?"
• "What tools do I need?"
• "Is this safe?"

**Status:**
• "Where am I?"
• "What's my progress?"
• "Show status"

**No commands needed!** Just ask like you're talking to a contractor.
"""
    
    # Default helpful response
    if '?' in user_input:
        return f"""I hear you asking: *"{user_input}"*

I'm here to help! Try asking:

• **"Start new project"** - Begin building
• **"What's next?"** - See your next step
• **"How much does [something] cost?"** - Budget info
• **"Can I DIY [task]?"** - DIY guidance
• **"Show foundation steps"** - See all steps

Or just chat naturally - I understand construction questions!
"""
    
    return f"""I understand you said: *"{user_input}"*

Let me help! Some things you can ask:

• **"Start new project"** if you're just beginning
• **"What's next?"** to see your next step
• **"How much will this cost?"** for budget info
• **"Show me the steps"** to see the roadmap

Or ask any construction question naturally!
"""


# Main App Layout
def main():
    # Sidebar
    with st.sidebar:
        st.title("🏗️ Kalki AI")
        st.markdown("### Your AI General Contractor")
        
        if st.session_state.project:
            st.markdown("---")
            st.markdown("### 📊 Project Overview")
            
            total_budget = st.session_state.project.budget_spent + st.session_state.project.budget_remaining
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Phase", "Foundation")
                st.metric("Step", f"{st.session_state.current_step}/11")
            with col2:
                st.metric("Budget", f"${total_budget/1000:.0f}K")
                st.metric("Done", f"{len(st.session_state.project.completed_steps)}")
            
            st.markdown("---")
            st.markdown("### 🎯 Quick Actions")
            if st.button("📋 Show Status", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "status"
                })
                st.rerun()
            
            if st.button("⏭️ What's Next?", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "what's next"
                })
                st.rerun()
            
            if st.button("🏗️ All Steps", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "show foundation steps"
                })
                st.rerun()
        else:
            st.markdown("---")
            st.markdown("### 🚀 Get Started")
            st.markdown("Start chatting to begin your construction journey!")
            
            if st.button("▶️ Start New Project", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": "start new project"
                })
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.markdown("""
        - Just chat naturally!
        - Ask "what's next?" anytime
        - Say "help" for guidance
        - I know all about construction 🏗️
        """)
        
        st.markdown("---")
        st.markdown("### 📱 Works Everywhere")
        st.markdown("""
        ✅ Desktop browsers
        ✅ Mobile browsers
        ✅ Tablets
        """)
    
    # Main Chat Area
    st.title("💬 Chat with Kalki")
    
    # Welcome message
    if not st.session_state.messages and not st.session_state.welcome_shown:
        st.markdown("""
        <div class="welcome-card">
            <h1>🏗️ Welcome to Kalki!</h1>
            <h3>Your AI General Contractor</h3>
            <p style="font-size: 18px; color: #666; margin: 20px 0;">
                I'll guide you through building your house from foundation to finish.<br>
                Expert-level guidance, step-by-step, with costs, timelines, and safety tips.
            </p>
            <p style="font-size: 16px; margin-top: 30px;">
                👇 Start by saying <strong>"Start new project"</strong> or ask me anything!
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.welcome_shown = True
    
    # Chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="kalki-message">
                    <strong>🏗️ Kalki:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    user_input = st.chat_input("Type your message here... (e.g., 'start new project')")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get Kalki's response
        response = process_message(user_input)
        
        # Add Kalki's response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
        
        # Rerun to show new messages
        st.rerun()


if __name__ == "__main__":
    main()
