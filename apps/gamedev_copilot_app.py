#!/usr/bin/env python3
"""
GameDevCopilot App - Streamlit Interface
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A beautiful, modern interface for KALKI's Game Development Copilot.
Create games from scratch through guided conversation.
"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import GameDevCopilot
from modules.game_dev_copilot import GameDevCopilot, ProjectRequirements

# Page configuration
st.set_page_config(
    page_title="GameDev Copilot - KALKI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .project-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-active {
        background: #d4edda;
        color: #155724;
    }
    .status-complete {
        background: #d1ecf1;
        color: #0c5460;
    }
    .status-needs-input {
        background: #fff3cd;
        color: #856404;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-user {
        background: #e3f2fd;
        margin-left: 20%;
    }
    .chat-assistant {
        background: #f5f5f5;
        margin-right: 20%;
    }
    .question-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .file-list {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION
# ============================================================================

@st.cache_resource
def init_copilot():
    """Initialize GameDevCopilot (cached)"""
    return GameDevCopilot()

# Initialize copilot
if 'copilot' not in st.session_state:
    with st.spinner("🎮 Initializing GameDev Copilot..."):
        st.session_state.copilot = init_copilot()

copilot = st.session_state.copilot

# Session state management
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'project_status' not in st.session_state:
    st.session_state.project_status = None

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎮 GameDev Copilot")
    st.markdown("---")
    
    # Project management
    st.markdown("### 📁 Projects")
    
    # List active projects
    active_projects = list(copilot.active_projects.keys())
    if active_projects:
        selected_project = st.selectbox(
            "Active Projects",
            ["New Project"] + active_projects,
            key="project_selector"
        )
        
        if selected_project != "New Project":
            st.session_state.current_project_id = selected_project
            # Load project info
            if selected_project in copilot.generated_projects:
                project_info = copilot.generated_projects[selected_project]
                st.json(project_info)
    else:
        st.info("No active projects")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🆕 New Project", use_container_width=True):
        st.session_state.current_session_id = None
        st.session_state.current_project_id = None
        st.session_state.chat_history = []
        st.session_state.project_status = None
        st.rerun()
    
    if st.session_state.current_project_id:
        if st.button("📂 View Files", use_container_width=True):
            st.session_state.show_files = True
        
        if st.button("🚀 Deploy", use_container_width=True):
            st.session_state.deploy_project = True
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Stats")
    st.metric("Active Projects", len(active_projects))
    st.metric("Total Sessions", len(copilot.requirement_sessions))

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🎮 GameDev Copilot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Create games from scratch through intelligent conversation</p>', unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["💬 Create Game", "📁 Projects", "📚 Help"])

# ============================================================================
# TAB 1: CREATE GAME
# ============================================================================

with tab1:
    # Check if we have an active session
    if st.session_state.current_session_id is None:
        # New project flow
        st.markdown("### 🚀 Start a New Game Project")
        st.markdown("Tell me what kind of game you want to create!")
        
        # Example prompts
        st.markdown("**Examples:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎯 Make a carjam style game", use_container_width=True):
                user_input = "make me a carjam style game"
                st.session_state.user_input = user_input
        with col2:
            if st.button("🐦 Create a flappy bird clone", use_container_width=True):
                user_input = "make me a flappy bird style game"
                st.session_state.user_input = user_input
        with col3:
            if st.button("🧩 Build a puzzle game", use_container_width=True):
                user_input = "make me a puzzle game"
                st.session_state.user_input = user_input
        
        # Custom input
        user_input = st.text_input(
            "Or describe your game idea:",
            value=st.session_state.get('user_input', ''),
            placeholder="e.g., make me a racing game like carjam",
            key="game_input"
        )
        
        if st.button("🚀 Start Project", type="primary", use_container_width=True):
            if user_input:
                with st.spinner("🔍 Researching and analyzing your game idea..."):
                    result = asyncio.run(copilot.start_new_game_project(user_input))
                
                # Store session
                st.session_state.current_session_id = result.get('session_id')
                st.session_state.project_status = result.get('status')
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input,
                    'timestamp': datetime.now()
                })
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': result.get('message', ''),
                    'timestamp': datetime.now(),
                    'status': result.get('status'),
                    'next_question': result.get('next_question')
                })
                
                st.rerun()
            else:
                st.warning("Please enter a game idea!")
    
    else:
        # Active session - show conversation
        st.markdown("### 💬 Conversation")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-message chat-user"><strong>You:</strong> {msg["content"]}</div>', 
                          unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message chat-assistant"><strong>KALKI:</strong> {msg["content"]}</div>', 
                          unsafe_allow_html=True)
                
                # Show recommendation if available
                if msg.get('recommendation'):
                    rec = msg['recommendation']
                    st.info(rec.get('message', ''))
                    
                    # Quick accept button for recommendation
                    if rec.get('recommendation'):
                        rec_value = rec.get('recommendation')
                        if st.button(f"✅ Accept: {rec_value}", key=f"accept_rec_{len(st.session_state.chat_history)}"):
                            # Auto-submit the recommendation
                            with st.spinner("Processing..."):
                                result = asyncio.run(
                                    copilot.answer_question(st.session_state.current_session_id, rec_value)
                                )
                            
                            st.session_state.chat_history.append({
                                'role': 'user',
                                'content': rec_value,
                                'timestamp': datetime.now()
                            })
                            st.session_state.chat_history.append({
                                'role': 'assistant',
                                'content': result.get('message', ''),
                                'timestamp': datetime.now(),
                                'status': result.get('status'),
                                'next_question': result.get('next_question'),
                                'project_id': result.get('project_id')
                            })
                            
                            st.session_state.project_status = result.get('status')
                            if result.get('project_id'):
                                st.session_state.current_project_id = result.get('project_id')
                            
                            st.rerun()
                
                # Show next question if available
                if msg.get('next_question'):
                    next_q = msg['next_question']
                    st.markdown(f'''
                    <div class="question-box">
                        <strong>❓ {next_q.question}</strong><br>
                        <small>{next_q.context}</small>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Show options if available
                    if next_q.options:
                        st.markdown("**Options:**")
                        for i, option in enumerate(next_q.options, 1):
                            st.markdown(f"{i}. {option}")
        
        # Current status
        current_status = st.session_state.project_status
        
        if current_status == 'needs_input':
            # Show answer input
            st.markdown("---")
            st.markdown("### ✍️ Your Answer")
            
            # Get last question
            last_msg = st.session_state.chat_history[-1]
            if last_msg.get('next_question'):
                next_q = last_msg['next_question']
                
                # Quick answer buttons if options available
                if next_q.options:
                    st.markdown("**Quick Select:**")
                    cols = st.columns(min(3, len(next_q.options)))
                    for i, option in enumerate(next_q.options):
                        with cols[i % 3]:
                            if st.button(option, key=f"option_{i}", use_container_width=True):
                                answer = option
                                st.session_state.pending_answer = answer
                                st.rerun()
                
                # Text input
                answer = st.text_input(
                    "Your answer:",
                    value=st.session_state.get('pending_answer', ''),
                    key="answer_input"
                )
                
                if st.button("✅ Submit Answer", type="primary", use_container_width=True):
                    if answer:
                        with st.spinner("Processing your answer..."):
                            result = asyncio.run(
                                copilot.answer_question(st.session_state.current_session_id, answer)
                            )
                        
                        # Update chat history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': answer,
                            'timestamp': datetime.now()
                        })
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': result.get('message', ''),
                            'timestamp': datetime.now(),
                            'status': result.get('status'),
                            'next_question': result.get('next_question'),
                            'project_id': result.get('project_id'),
                            'recommendation': result.get('recommendation')  # Include recommendation
                        })
                        
                        st.session_state.project_status = result.get('status')
                        if result.get('project_id'):
                            st.session_state.current_project_id = result.get('project_id')
                        
                        # Clear pending answer
                        if 'pending_answer' in st.session_state:
                            del st.session_state.pending_answer
                        
                        st.rerun()
                    else:
                        st.warning("Please provide an answer!")
        
        elif current_status == 'project_created':
            # Project created - show success and next steps
            st.success("🎉 Project Created Successfully!")
            
            project_id = st.session_state.current_project_id
            if project_id:
                # Show project info
                if project_id in copilot.generated_projects:
                    project_info = copilot.generated_projects[project_id]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Files Generated", project_info.get('files_generated', 0))
                        st.metric("Engine", project_info.get('engine', 'N/A'))
                    with col2:
                        st.metric("Platforms", ', '.join(project_info.get('platforms', [])))
                        st.metric("Output Directory", project_info.get('output_dir', 'N/A'))
                    
                    # Show generated files
                    if st.checkbox("📁 Show Generated Files"):
                        files = project_info.get('files', [])
                        st.markdown(f'<div class="file-list">{"<br>".join(files[:20])}</div>', 
                                  unsafe_allow_html=True)
                    
                    # Actions
                    st.markdown("### 🚀 Next Steps")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("📦 Generate Assets", use_container_width=True):
                            with st.spinner("Generating assets..."):
                                requirements = copilot.requirement_sessions.get(st.session_state.current_session_id)
                                if requirements:
                                    result = asyncio.run(
                                        copilot.generate_game_assets(project_id, requirements)
                                    )
                                    st.success(result.get('message', 'Assets generated!'))
                    
                    with col2:
                        if st.button("🚀 Deploy Game", use_container_width=True):
                            with st.spinner("Deploying..."):
                                result = asyncio.run(copilot.deploy_game(project_id))
                                st.success(result.get('message', 'Deployment started!'))
                    
                    with col3:
                        if st.button("✨ Polish Game", use_container_width=True):
                            with st.spinner("Polishing..."):
                                result = asyncio.run(copilot.polish_game(project_id, "standard"))
                                st.success(result.get('message', 'Polish complete!'))

# ============================================================================
# TAB 2: PROJECTS
# ============================================================================

with tab2:
    st.markdown("### 📁 Your Projects")
    
    if copilot.generated_projects:
        for project_id, project_info in copilot.generated_projects.items():
            with st.expander(f"🎮 {project_id}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Engine:** {project_info.get('engine', 'N/A')}")
                    st.markdown(f"**Platforms:** {', '.join(project_info.get('platforms', []))}")
                    st.markdown(f"**Files:** {project_info.get('files_generated', 0)} files")
                    st.markdown(f"**Created:** {project_info.get('generated_at', 'N/A')}")
                    st.markdown(f"**Location:** `{project_info.get('output_dir', 'N/A')}`")
                
                with col2:
                    if st.button("📂 Open", key=f"open_{project_id}"):
                        st.info(f"Project location: {project_info.get('output_dir')}")
                    
                    if st.button("🚀 Deploy", key=f"deploy_{project_id}"):
                        with st.spinner("Deploying..."):
                            result = asyncio.run(copilot.deploy_game(project_id))
                            st.success(result.get('message', 'Deployment started!'))
                    
                    if st.button("✨ Polish", key=f"polish_{project_id}"):
                        with st.spinner("Polishing..."):
                            result = asyncio.run(copilot.polish_game(project_id, "standard"))
                            st.success(result.get('message', 'Polish complete!'))
                
                # Show files
                if st.checkbox(f"Show files for {project_id}", key=f"files_{project_id}"):
                    files = project_info.get('files', [])
                    st.code('\n'.join(files[:20]), language='text')
    else:
        st.info("No projects yet. Create your first game in the 'Create Game' tab!")

# ============================================================================
# TAB 3: HELP
# ============================================================================

with tab3:
    st.markdown("### 📚 GameDev Copilot Guide")
    
    st.markdown("""
    #### 🎯 How It Works
    
    1. **Describe Your Game Idea**
       - Just tell me what kind of game you want (e.g., "make me a carjam style game")
       - I'll research similar games and understand what you're looking for
    
    2. **Answer Questions**
       - I'll ask you about platforms, engine, monetization, etc.
       - You can answer naturally or use the quick-select buttons
    
    3. **Get Your Game**
       - I'll generate all the code, assets, and project files
       - Your game will be ready to build and deploy!
    
    #### 💡 Tips
    
    - **Be specific**: "make me a racing game like carjam" is better than "make a game"
    - **Use examples**: Reference games you like (carjam, flappy bird, etc.)
    - **Answer questions**: The more you answer, the better your game will be
    
    #### 🎮 Supported Engines
    
    - **Unity** - Best for mobile games, 3D games
    - **Flutter** - Cross-platform mobile games
    - **React Native** - Web + mobile games
    - **Web** - HTML5 games (playable in browser)
    
    #### 📱 Supported Platforms
    
    - Android
    - iOS
    - Web
    - PC (via Unity/Flutter)
    
    #### 🚀 Features
    
    - ✅ Automatic research of game references
    - ✅ Smart question flow
    - ✅ Complete code generation
    - ✅ Asset generation
    - ✅ Build and deployment
    - ✅ Polish and optimization
    
    #### ❓ FAQ
    
    **Q: How long does it take?**  
    A: Usually 5-10 minutes from idea to deployable game!
    
    **Q: Can I customize the generated code?**  
    A: Yes! All code is in `output/games/{project_id}/` - edit as needed.
    
    **Q: What if I want to change something?**  
    A: Use the "Polish" feature or manually edit the generated files.
    
    **Q: Can I deploy to app stores?**  
    A: Yes! I generate build scripts and deployment guides for Android/iOS.
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("**GameDev Copilot** - Powered by KALKI | Create games from scratch through conversation 🎮")

