#!/usr/bin/env python3
"""
KALKI CONSTRUCTION APP - WITH FULL AI BRAIN 🧠
Beautiful web-based chat interface powered by:
- Llama 3.1 8B (reasoning)
- Vector DB (semantic search)
- Knowledge DBs (1000+ construction specs)
Your intelligent AI general contractor that thinks and reasons
"""

import streamlit as st
from datetime import datetime
from modules.construction_copilot import ProjectState, ProjectPhase
from modules.foundation_steps import get_foundation_step, get_all_foundation_steps
from modules.llm import get_llm_engine, initialize_llm_engine
from modules.learning.vectordb import VectorDBManager
import json
import asyncio
import sqlite3
import os

# Page config
st.set_page_config(
    page_title="Kalki AI - Intelligent Construction Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    }
    .stButton>button:hover {
        background: #764ba2;
        transform: translateY(-2px);
    }
    .ai-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
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

if 'llm_engine' not in st.session_state:
    st.session_state.llm_engine = None
    st.session_state.llm_ready = False

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
    st.session_state.vectordb_ready = False

if 'welcome_shown' not in st.session_state:
    st.session_state.welcome_shown = False


@st.cache_resource
def initialize_llm():
    """Initialize LLM engine (cached so it only runs once)"""
    async def init():
        success = await initialize_llm_engine()
        if success:
            return get_llm_engine()
        return None
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    engine = loop.run_until_complete(init())
    return engine


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


def build_context(user_input: str) -> str:
    """Build context for the LLM based on project state and knowledge"""
    context_parts = []
    
    # Project context
    if st.session_state.project:
        total_budget = st.session_state.project.budget_spent + st.session_state.project.budget_remaining
        context_parts.append(f"""
CURRENT PROJECT STATUS:
- Phase: Foundation
- Current Step: {st.session_state.current_step} of 11
- Budget: ${total_budget:,.0f} (${st.session_state.project.budget_remaining:,.0f} remaining)
- Timeline: {(st.session_state.project.timeline_days_elapsed + st.session_state.project.timeline_days_remaining) // 7} weeks
- Completed: {len(st.session_state.project.completed_steps)} steps
""")
        
        # Add current step details
        if st.session_state.current_step <= 11:
            step = get_foundation_step(st.session_state.current_step, st.session_state.project)
            context_parts.append(f"""
CURRENT STEP DETAILS:
- Step {step.step_number}: {step.title}
- Cost: ${step.estimated_cost:,.0f}
- Duration: {step.estimated_duration_days} days
- Requires Professional: {step.requires_professional}
- Why Now: {step.why_now}
""")
    
    # Add foundation knowledge
    context_parts.append("""
FOUNDATION STEPS OVERVIEW:
1. Site Excavation - $2,500, 1 day (MUST call 811 first!)
2. Footing Layout - $150, 1 day
3. Footing Forms - $800, 2 days
4. Rebar Installation - $600, 1 day
5. Pre-Pour Inspection - FREE, 1 day (REQUIRED)
6. Concrete Pour - $2,500, 1 day
7. Strip Forms - FREE, 1 day
8. Foundation Walls - $8,000, 5 days
9. Waterproofing - $2,000, 7 days
10. Backfill - $800, 4 days
11. Final Inspection - FREE, 3 days

Total: $17,350 | 27 days
""")
    
    return "\n".join(context_parts)


async def process_message_ai(user_input: str) -> str:
    """Process user message using Llama 3.1 8B AI"""
    
    # Check for project initialization
    if any(phrase in user_input.lower() for phrase in ['start', 'new project', 'begin', 'let\'s go']):
        if st.session_state.project:
            return "You already have a project started! Let me know how I can help with your current foundation work."
        init_project()
        user_input = "The user just started a new construction project. Welcome them and explain step 1 (Site Excavation) briefly but enthusiastically."
    
    # Build context
    context = build_context(user_input)
    
    # Create system prompt
    system_prompt = """You are Kalki, an expert AI construction copilot and general contractor with decades of experience. 

Your role:
- Help homeowners build houses step-by-step
- Provide expert construction advice
- Ensure safety and code compliance
- Explain complex topics simply
- Be encouraging and supportive

Style:
- Conversational and friendly
- Use emojis sparingly (🏗️ ⚠️ ✅)
- Keep responses focused and practical
- Emphasize safety when relevant
- Give actionable advice

Current context:
{context}

User question: {question}

Provide a helpful, expert response (2-4 paragraphs max):"""
    
    # Format prompt
    full_prompt = system_prompt.format(context=context, question=user_input)
    
    # Generate response
    engine = st.session_state.llm_engine
    response = await engine.generate(full_prompt, max_new_tokens=300, temperature=0.7)
    
    return response


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='ai-badge'>🧠 Powered by Llama 3.1 8B</div>", unsafe_allow_html=True)
        st.title("🏗️ Kalki AI")
        st.markdown("---")
        
        # Initialize LLM if not ready
        if not st.session_state.llm_ready:
            with st.spinner("🧠 Loading AI brain (Llama 3.1 8B)..."):
                engine = initialize_llm()
                if engine:
                    st.session_state.llm_engine = engine
                    st.session_state.llm_ready = True
                    st.success("✅ AI Ready!")
                else:
                    st.error("❌ AI failed to load")
                    st.stop()
        
        # Project overview
        if st.session_state.project:
            st.markdown("### 📊 Project Overview")
            st.markdown(f"**Phase:** Foundation")
            st.markdown(f"**Step:** {st.session_state.current_step}/11")
            total_budget = st.session_state.project.budget_spent + st.session_state.project.budget_remaining
            st.markdown(f"**Budget:** ${total_budget:,.0f}")
            completion = (st.session_state.current_step / 11) * 100
            st.progress(completion / 100)
            st.markdown(f"**Done:** {completion:.0f}%")
        else:
            st.info("👋 No project yet! Say 'start new project' in chat")
        
        st.markdown("---")
        st.markdown("### 💡 Try asking:")
        st.markdown("• What's my next step?")
        st.markdown("• How much will this cost?")
        st.markdown("• Can I do this myself?")
        st.markdown("• What safety gear do I need?")
        st.markdown("• Tell me about [topic]")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("Kalki uses **Llama 3.1 8B** to provide intelligent, context-aware construction guidance.")
        st.markdown("🖥️ Running locally on your M4 Max")
        st.markdown("🔒 Your data stays private")
    
    # Main chat area
    st.title("💬 Chat with Kalki")
    
    # Welcome message
    if not st.session_state.welcome_shown:
        st.markdown("""
<div class='welcome-card'>
    <h2>🏗️ Welcome to Kalki AI</h2>
    <p style='font-size: 18px; color: #666;'>
        Your intelligent AI general contractor powered by <strong>Llama 3.1 8B</strong>
    </p>
    <p style='color: #888;'>
        I'll guide you through every step of building your house, from foundation to roof.
        <br>
        Ask me anything about construction, costs, safety, or your project!
    </p>
    <p style='color: #667eea; font-weight: bold;'>
        💬 Say "start new project" to begin!
    </p>
</div>
        """, unsafe_allow_html=True)
        st.session_state.welcome_shown = True
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"<div class='user-message'>{content}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='kalki-message'>{content}</div>", unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about construction..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        
        # Generate AI response
        with st.spinner("🧠 Kalki is thinking..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(process_message_ai(prompt))
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.markdown(f"<div class='kalki-message'>{response}</div>", unsafe_allow_html=True)
        
        # Rerun to update display
        st.rerun()


if __name__ == "__main__":
    main()
