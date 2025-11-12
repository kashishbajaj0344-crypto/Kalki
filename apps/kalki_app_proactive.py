#!/usr/bin/env python3
"""
KALKI PROACTIVE CONSTRUCTION MANAGER 🎯
- Asks questions to understand project state
- Searches internet for location-specific requirements
- Breaks down pre-construction steps
- Coordinates with professionals
- Guides through every decision point
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
from enum import Enum

# Page config
st.set_page_config(
    page_title="Kalki AI - Proactive Construction Manager",
    page_icon="🎯",
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
    .kalki-question {
        background: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 70%;
        border-left: 4px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .kalki-action {
        background: #d1ecf1;
        color: #0c5460;
        padding: 15px;
        border-radius: 18px 18px 18px 5px;
        margin: 10px 0;
        max-width: 70%;
        border-left: 4px solid #17a2b8;
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
    .decision-point {
        background: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


class ProjectStage(Enum):
    """Track detailed project stage"""
    INITIAL_CONTACT = "initial_contact"
    LOT_ASSESSMENT = "lot_assessment"
    PERMITS_PLANNING = "permits_planning"
    SITE_PREP = "site_prep"
    FOUNDATION = "foundation"
    FRAMING = "framing"
    # ... more stages


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'project' not in st.session_state:
    st.session_state.project = None

if 'project_stage' not in st.session_state:
    st.session_state.project_stage = ProjectStage.INITIAL_CONTACT
    
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

if 'location_info' not in st.session_state:
    st.session_state.location_info = {}

if 'pending_decisions' not in st.session_state:
    st.session_state.pending_decisions = []

if 'professionals_needed' not in st.session_state:
    st.session_state.professionals_needed = []

if 'llm_engine' not in st.session_state:
    st.session_state.llm_engine = None
    st.session_state.llm_ready = False

if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None

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


def query_knowledge_db(query_type: str, params: dict = None) -> str:
    """Query the construction knowledge databases"""
    results = []
    
    try:
        if query_type == "procedures":
            conn = sqlite3.connect('data/knowledge/procedures.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM procedures LIMIT 3")
            rows = cursor.fetchall()
            for row in rows:
                results.append(f"Procedure: {row[1] if len(row) > 1 else 'N/A'}")
            conn.close()
        
        elif query_type == "inspection":
            conn = sqlite3.connect('data/knowledge/inspection_criteria.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspection_criteria LIMIT 3")
            rows = cursor.fetchall()
            for row in rows:
                results.append(f"Inspection: {row[1] if len(row) > 1 else 'N/A'}")
            conn.close()
        
        elif query_type == "costs":
            conn = sqlite3.connect('data/knowledge/cost_data.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cost_data LIMIT 3")
            rows = cursor.fetchall()
            for row in rows:
                results.append(f"Cost item: {row[1] if len(row) > 1 else 'N/A'}")
            conn.close()
    
    except Exception as e:
        results.append(f"Error querying {query_type}: {e}")
    
    return "\n".join(results) if results else "No data found"


async def search_location_info(location: str) -> dict:
    """Simulate searching for location-specific building requirements"""
    # In production, this would use real APIs:
    # - Google Maps API for address validation
    # - Municipal databases for zoning
    # - Building department APIs
    # - Weather APIs for climate data
    
    location_lower = location.lower()
    
    info = {
        "address": location,
        "municipality": "Unknown",
        "zoning": "To be determined",
        "frost_depth": "Check local codes",
        "seismic_zone": "To be determined",
        "wind_speed": "To be determined",
        "snow_load": "To be determined",
        "requirements": []
    }
    
    # Example: Sechelt, BC
    if "sechelt" in location_lower:
        info.update({
            "municipality": "District of Sechelt, BC",
            "zoning": "Likely R1 or R2 (verify with district)",
            "frost_depth": "18 inches (450mm) - BC Building Code",
            "seismic_zone": "High - West Coast seismic activity",
            "wind_speed": "120 km/h (coastal exposure)",
            "snow_load": "2.0 kPa ground snow load",
            "requirements": [
                "🏛️ Building permit from District of Sechelt",
                "📋 Development permit (may be required)",
                "🌊 Covenant for shoreline properties (if applicable)",
                "🏗️ Geotechnical report (REQUIRED for most lots)",
                "🌲 Tree cutting permit (if removing trees)",
                "💧 Septic system approval (if not on municipal sewer)",
                "⚡ BC Hydro service connection",
                "🚰 Water connection or well permit",
                "🔥 Fire department review (access requirements)"
            ]
        })
    
    return info


async def determine_next_questions(user_input: str, conversation_history: list) -> dict:
    """AI determines what questions to ask next based on context"""
    
    engine = st.session_state.llm_engine
    
    # Build conversation context
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]])
    
    prompt = f"""You are Kalki, a proactive construction project manager. Based on this conversation, determine:

1. What information do you still need?
2. What's the next logical question to ask?
3. Are there any decisions the user needs to make?
4. Should you take any actions (search, calculate, recommend)?

Conversation so far:
{context}

Latest user input: {user_input}

Current project stage: {st.session_state.project_stage.value}
Location info: {json.dumps(st.session_state.location_info, indent=2) if st.session_state.location_info else "None"}

Respond with JSON:
{{
    "needs_info": ["what info you need"],
    "next_question": "the most important question to ask",
    "pending_decisions": ["decisions user needs to make"],
    "suggested_actions": ["actions you should take"],
    "response_type": "question|action|guidance|answer"
}}
"""
    
    response = await engine.generate(prompt, max_new_tokens=200, temperature=0.7)
    
    # Try to extract JSON from response
    try:
        # Find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end > start:
            json_str = response[start:end]
            return json.loads(json_str)
    except:
        pass
    
    # Fallback
    return {
        "needs_info": ["project details"],
        "next_question": "Tell me more about your project",
        "pending_decisions": [],
        "suggested_actions": [],
        "response_type": "question"
    }


async def process_message_proactive(user_input: str) -> dict:
    """Process message with proactive questioning and guidance"""
    
    user_lower = user_input.lower()
    
    # === STAGE 1: INITIAL CONTACT ===
    if st.session_state.project_stage == ProjectStage.INITIAL_CONTACT:
        
        # Check if user mentioned location
        if any(word in user_lower for word in ['in', 'at', 'building', 'house', 'property', 'lot']):
            
            # Try to extract location
            words = user_input.split()
            potential_location = None
            for i, word in enumerate(words):
                if word.lower() == 'in' and i < len(words) - 1:
                    potential_location = ' '.join(words[i+1:i+3])
                    break
            
            if potential_location:
                # Search for location info
                st.session_state.location_info = await search_location_info(potential_location)
                st.session_state.project_stage = ProjectStage.LOT_ASSESSMENT
                
                return {
                    "message": f"""🎯 Excellent! Building in **{st.session_state.location_info['address']}**.

Let me gather some information about your location...

🔍 **LOCATION ANALYSIS:**
📍 Municipality: {st.session_state.location_info['municipality']}
🏘️ Zoning: {st.session_state.location_info['zoning']}
❄️ Frost Depth: {st.session_state.location_info['frost_depth']}
🌊 Seismic Zone: {st.session_state.location_info['seismic_zone']}

**KEY REQUIREMENTS FOR YOUR AREA:**
""" + "\n".join(st.session_state.location_info['requirements']) + """

---

**🤔 CRITICAL FIRST QUESTION:**

**Is your lot ready to build on, or do we need to prepare it first?**

By "ready" I mean:
✅ Zoning allows residential construction
✅ Utilities available (or plan in place)
✅ Lot is cleared and accessible
✅ No obvious drainage/stability issues
✅ You own the property or have approval

Let me know, and I'll guide you through the next steps!
""",
                    "type": "question",
                    "next_action": "lot_assessment"
                }
        
        # No location mentioned - ask for it
        return {
            "message": """🏗️ **I'd love to help you build your house!**

To give you accurate guidance, I need to know:

**📍 Where are you building?**

Please tell me:
- City/town name
- Province/state (if in Canada/US)
- Or full address if you have it

Why? Because construction requirements vary by location:
- Building codes differ by region
- Frost depth affects foundation design
- Seismic zones determine structural requirements
- Local bylaws affect permits and approvals
- Climate impacts material choices

Example: "I'm building in Sechelt, BC" or "123 Main St, Vancouver"

What's your location?
""",
            "type": "question",
            "next_action": "get_location"
        }
    
    # === STAGE 2: LOT ASSESSMENT ===
    elif st.session_state.project_stage == ProjectStage.LOT_ASSESSMENT:
        
        if 'no' in user_lower or 'not ready' in user_lower:
            # Lot not ready - need to assess what's needed
            return {
                "message": f"""🔍 **Got it - let's assess what needs to be done to prepare your lot.**

To determine the requirements, I need to understand the current state:

**❓ LOT ASSESSMENT QUESTIONS:**

1. **Do you own the lot already?** (Yes/No)

2. **Is the lot currently:**
   - [ ] Empty/cleared land
   - [ ] Forested (trees need removal)
   - [ ] Has existing structures
   - [ ] Vacant/overgrown

3. **Utilities - what's available?**
   - [ ] Municipal water
   - [ ] Sewer connection
   - [ ] Electricity at property line
   - [ ] None of the above

4. **Terrain - is it:**
   - [ ] Flat/level
   - [ ] Sloped (gentle or steep?)
   - [ ] Rocky
   - [ ] Concerns about drainage/water

5. **Has anyone done a survey or geotechnical assessment?** (Yes/No)

**Please answer these questions, or just describe your lot in your own words!**

(This helps me determine if you need: surveyor, geotech engineer, tree removal, utility connections, grading, etc.)
""",
                "type": "question",
                "next_action": "assess_lot_condition"
            }
        
        elif 'yes' in user_lower or 'ready' in user_lower:
            # Lot is ready - move to permits
            st.session_state.project_stage = ProjectStage.PERMITS_PLANNING
            init_project()  # Initialize construction project
            
            return {
                "message": """✅ **Perfect! Your lot is ready to build on.**

Now let's talk about **PERMITS AND APPROVALS**.

**For {location}, you'll typically need:**

1. **🏛️ BUILDING PERMIT** (REQUIRED)
   - Cost: ~$2,000-5,000
   - Timeline: 4-8 weeks to approve
   - Requires: Detailed plans, engineered drawings

2. **📐 PROFESSIONAL STAMP** (May be required)
   - Structural engineer stamp on plans
   - Cost: $1,500-3,000
   - Required for foundations, trusses, beams

3. **🌲 SITE PLAN** (REQUIRED)
   - Shows where house sits on lot
   - Setbacks from property lines
   - Surveyor may be needed: $800-1,500

**🎯 MY RECOMMENDATION:**

Let's start with a **SITE PLAN** because:
- You need it for the building permit anyway
- It determines where you can build
- Identifies any issues early
- Helps estimate excavation needs

**❓ NEXT DECISION:**

**Do you already have:**
1. House plans/blueprints? (Yes/No)
2. A builder/contractor lined up? (Yes/No)
3. Financing arranged? (Yes/No)

Let me know, and I'll create a custom timeline and budget for YOUR project!
""".format(location=st.session_state.location_info['address']),
                "type": "action",
                "next_action": "planning_permits"
            }
    
    # === STAGE 3: PERMITS & PLANNING ===
    elif st.session_state.project_stage == ProjectStage.PERMITS_PLANNING:
        # Guide through permit process
        pass
    
    # === DEFAULT: USE AI TO DETERMINE RESPONSE ===
    
    # Build context from knowledge DBs
    procedures_context = query_knowledge_db("procedures")
    inspection_context = query_knowledge_db("inspection")
    
    # Build full context
    context = f"""
LOCATION: {st.session_state.location_info.get('address', 'Unknown')}
STAGE: {st.session_state.project_stage.value}
PROJECT: {st.session_state.project is not None}

KNOWLEDGE BASE CONTEXT:
{procedures_context}

{inspection_context}

USER QUESTION: {user_input}
"""
    
    # Determine what to do next
    next_steps = await determine_next_questions(user_input, st.session_state.messages)
    
    # Generate AI response
    engine = st.session_state.llm_engine
    
    prompt = f"""You are Kalki, a proactive AI construction manager. 

Context:
{context}

Based on the analysis:
{json.dumps(next_steps, indent=2)}

Provide a helpful, proactive response that:
1. Answers the user's question
2. Asks the next logical question
3. Suggests specific actions
4. Keeps the project moving forward

Be conversational, use emojis sparingly, and focus on actionable advice.

Response (2-3 paragraphs):"""
    
    response = await engine.generate(prompt, max_new_tokens=350, temperature=0.7)
    
    return {
        "message": response,
        "type": next_steps.get("response_type", "guidance"),
        "next_action": None
    }


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("<div class='ai-badge'>🎯 Proactive AI Manager</div>", unsafe_allow_html=True)
        st.title("🏗️ Kalki AI")
        st.markdown("---")
        
        # Initialize LLM if not ready
        if not st.session_state.llm_ready:
            with st.spinner("🧠 Loading AI brain..."):
                engine = initialize_llm()
                if engine:
                    st.session_state.llm_engine = engine
                    st.session_state.llm_ready = True
                    st.success("✅ AI Ready!")
                else:
                    st.error("❌ AI failed to load")
                    st.stop()
        
        # Project stage indicator
        st.markdown("### 📊 Project Stage")
        stages = {
            ProjectStage.INITIAL_CONTACT: "📞 Initial Contact",
            ProjectStage.LOT_ASSESSMENT: "🔍 Lot Assessment",
            ProjectStage.PERMITS_PLANNING: "📋 Permits & Planning",
            ProjectStage.SITE_PREP: "🚧 Site Preparation",
            ProjectStage.FOUNDATION: "🏗️ Foundation Work"
        }
        st.info(stages.get(st.session_state.project_stage, "Unknown"))
        
        # Location info
        if st.session_state.location_info:
            st.markdown("### 📍 Location")
            st.markdown(f"**{st.session_state.location_info.get('address', 'Unknown')}**")
            st.markdown(f"🏘️ {st.session_state.location_info.get('municipality', 'N/A')}")
        
        # Pending decisions
        if st.session_state.pending_decisions:
            st.markdown("### ⚠️ Decisions Needed")
            for decision in st.session_state.pending_decisions:
                st.warning(decision)
        
        # Professionals needed
        if st.session_state.professionals_needed:
            st.markdown("### 👷 Professionals Needed")
            for pro in st.session_state.professionals_needed:
                st.info(f"• {pro}")
        
        st.markdown("---")
        st.markdown("### 💡 How I Help")
        st.markdown("""
- 🤔 Ask proactive questions
- 🔍 Research your location
- 📋 Break down requirements
- 👷 Coordinate professionals
- ✅ Guide every decision
        """)
    
    # Main chat area
    st.title("💬 Chat with Kalki")
    
    # Welcome message
    if not st.session_state.welcome_shown:
        st.markdown("""
<div class='welcome-card'>
    <h2>🎯 Welcome to Kalki AI - Your Proactive Construction Manager</h2>
    <p style='font-size: 18px; color: #666;'>
        I don't just answer questions - I <strong>guide your entire project</strong>
    </p>
    <p style='color: #888;'>
        I'll ask questions to understand your situation, research requirements for your location,
        break down pre-construction needs, coordinate professionals, and guide you through every decision.
    </p>
    <p style='color: #667eea; font-weight: bold;'>
        💬 Tell me: "I want to build a house in [your location]"
    </p>
</div>
        """, unsafe_allow_html=True)
        st.session_state.welcome_shown = True
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        msg_type = message.get("type", "normal")
        
        if role == "user":
            st.markdown(f"<div class='user-message'>{content}</div>", unsafe_allow_html=True)
        else:
            if msg_type == "question":
                st.markdown(f"<div class='kalki-question'>🤔 {content}</div>", unsafe_allow_html=True)
            elif msg_type == "action":
                st.markdown(f"<div class='kalki-action'>🎯 {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='kalki-message'>{content}</div>", unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Tell me about your project..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='user-message'>{prompt}</div>", unsafe_allow_html=True)
        
        # Generate AI response
        with st.spinner("🎯 Kalki is analyzing..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(process_message_proactive(prompt))
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["message"],
            "type": result.get("type", "normal")
        })
        
        if result.get("type") == "question":
            st.markdown(f"<div class='kalki-question'>🤔 {result['message']}</div>", unsafe_allow_html=True)
        elif result.get("type") == "action":
            st.markdown(f"<div class='kalki-action'>🎯 {result['message']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='kalki-message'>{result['message']}</div>", unsafe_allow_html=True)
        
        # Rerun to update display
        st.rerun()


if __name__ == "__main__":
    main()
