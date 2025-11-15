# 🤖 How the Kalki AI Chatbot Works

## Overview

The **Kalki Unified Chatbot** (`apps/kalki_unified_chat.py`) is a centralized, intelligent interface that provides access to ALL Kalki capabilities across multiple domains through a single chat interface. It automatically detects domains, routes queries intelligently, and provides context-aware responses.

---

## 🏗️ Architecture

### High-Level Flow

```
User Query
    ↓
Domain Detection (DomainRegistry.infer_domain())
    ↓
┌──────────────────────┬──────────────────────┐
│ Domain Detected      │ No Domain Detected    │
└──────────────────────┴──────────────────────┘
         ↓                        ↓
Supreme Control Hub      Kalki Orchestrator
(domain-aware)          (20-phase system)
         ↓                        ↓
Domain Handler          Multi-Agent System
(Copilot or Domain)     (General Intelligence)
         ↓                        ↓
Response with Domain    General AI Response
```

---

## 🔍 Core Components

### 1. **UnifiedKalkiChat** (`apps/kalki_unified_chat.py`)

The main chatbot class that orchestrates everything:

**Key Features:**
- Automatic domain detection from queries
- Intelligent routing to domain handlers or general orchestrator
- Chat history and context management
- Beautiful CLI interface with Rich console
- Statistics tracking

**Initialization:**
```python
class UnifiedKalkiChat:
    def __init__(self):
        self.domain_registry = DomainRegistry()  # Auto-discovers domains
        self.supreme_hub = None  # Lazy-loaded
        self.orchestrator = None  # Lazy-loaded
        self.chat_history = []  # Context management
```

### 2. **DomainRegistry** (`modules/domains/domain_registry.py`)

Auto-discovers and manages all domain modules:

**Key Methods:**
- `infer_domain(query)` - Detects which domain(s) a query belongs to
- `list_domains()` - Lists all available domains
- `get_domain(name, prefer_copilot=True)` - Gets domain or copilot instance

**Domain Detection Algorithm:**
1. **Keyword-Based Matching** (current implementation)
   - Each domain has weighted keywords
   - Construction: "joist", "beam", "foundation", "building code" (high weight)
   - Game Dev: "unity", "unreal", "game", "character controller" (high weight)
   - Robotics: "robot", "kinematics", "pid", "slam" (high weight)
   - Scores domains by keyword matches
   - Returns top-scoring domain(s)

2. **Future: LLM-Based Detection** (planned)
   - Uses local Llama models for semantic understanding
   - More accurate domain classification
   - Handles ambiguous queries better

**Supported Domains:**
- 🏗️ **Construction** - Building codes, structural engineering, construction phases
- 🎮 **Game Development** - Unity, Unreal, game mechanics, assets
- 🤖 **Robotics** - Kinematics, control systems, ROS, sensors
- ✈️ **Aerospace** - Aerodynamics, propulsion, flight dynamics, VTOL
- ⚡ **Power Systems** - Batteries, solar, power electronics, energy storage
- 🔬 **General Intelligence** - 20-phase cognitive framework for any topic

### 3. **SupremeControlHub** (`modules/supreme_control_hub.py`)

Domain-aware query processor that routes to appropriate handlers:

**Key Method:**
```python
async def process_domain_aware_query(
    query: str,
    context: Dict[str, Any],
    project_id: Optional[str] = None
) -> Dict[str, Any]
```

**Processing Flow:**
1. **Domain Inference** - Uses DomainRegistry to detect domain
2. **Copilot Detection** - Checks if domain has an enhanced copilot
3. **Query Routing**:
   - **Game Dev Copilot**: Handles project creation, Q&A workflow
   - **Construction Copilot**: Handles project creation, step-by-step guidance
   - **Domain Handler**: Standard domain processing
4. **Response Generation** - Returns structured response with metadata

**Copilot vs Domain:**
- **Copilot**: Enhanced interface with project management, workflows
- **Domain**: Core domain expertise, knowledge extraction

### 4. **KalkiOrchestrator** (`src/kalki_complete.py`)

General intelligence system for non-domain queries:

**Features:**
- 20-phase cognitive framework
- Multi-agent coordination
- Meta-reasoning and synthesis
- General knowledge processing

**Used When:**
- No specific domain detected
- General knowledge questions
- Cross-domain queries
- Abstract reasoning tasks

---

## 🔄 Query Processing Flow

### Step-by-Step Process

1. **User Input**
   ```python
   user_input = "What size joists for a 16 foot span?"
   ```

2. **Domain Detection**
   ```python
   inferred_domains = await domain_registry.infer_domain(user_input)
   # Returns: ["construction"]
   ```

3. **Routing Decision**
   ```python
   if inferred_domains:
       # Domain-specific query
       result = await supreme_hub.process_domain_aware_query(
           query=user_input,
           context=chat_context
       )
   else:
       # General query
       result = await orchestrator.process_user_query(user_input)
   ```

4. **Domain Processing** (if domain detected)
   ```python
   # Supreme Hub:
   # 1. Gets domain/copilot instance
   domain = domain_registry.get_domain("construction", prefer_copilot=True)
   
   # 2. Processes query with domain knowledge
   # 3. Returns structured response
   ```

5. **Response Formatting**
   ```python
   response = {
       "response": "For a 16-foot span, use 2x10 joists...",
       "domain": "construction",
       "confidence": 0.9,
       "metadata": {
           "domains_detected": ["construction"],
           "processing_time": 1.2
       }
   }
   ```

6. **Context Update**
   ```python
   chat_history.append({
       "timestamp": datetime.now().isoformat(),
       "user": user_input,
       "response": response,
       "domain": "construction"
   })
   ```

---

## 🎯 Special Features

### 1. **Copilot Integration**

When a domain has a copilot (enhanced interface), the chatbot automatically uses it:

**Game Dev Copilot:**
- Detects project creation requests ("make a game", "create a game")
- Starts interactive Q&A workflow
- Manages project sessions
- Auto-triggers complete build workflow

**Construction Copilot:**
- Detects project creation ("new project", "build house")
- Provides step-by-step guidance
- Manages construction phases
- Generates deliverables

### 2. **Context Management**

The chatbot maintains conversation context:

```python
def _get_chat_context(self) -> Dict[str, Any]:
    return {
        "recent_exchanges": last_5_messages,
        "current_domain": self.current_domain,
        "project_id": self.current_project_id
    }
```

**Benefits:**
- Follow-up questions work naturally
- Project context persists
- Domain context maintained across conversation

### 3. **Session Management**

For copilots that use workflows (like Game Dev):

```python
# Session ID stored in chat history
if result.get("session_id"):
    chat_context["session_id"] = result["session_id"]
    
# Next query uses same session
if session_id:
    result = await copilot.answer_question(session_id, query)
```

### 4. **Statistics Tracking**

The chatbot tracks usage:

```python
stats = {
    "total_queries": 0,
    "domain_queries": 0,
    "general_queries": 0,
    "domains_used": set()
}
```

---

## 💬 User Interface

### Commands

- `/help` - Show help information
- `/domains` - List available domains
- `/stats` - Show chat statistics
- `/history` - Show recent chat history
- `/clear` - Clear chat history
- `/project <id>` - Set current project ID
- `/exit` - Exit chat

### Example Session

```
🤖 KALKI Chatbot

Available domains: construction, game_development, robotics, aerospace, power_systems

You: What size joists for a 16 foot span?

[construction] Kalki:
For a 16-foot span with standard residential loading (40 PSF live load), 
you would typically use:

- **2x10 SPF joists** at 16" spacing
- Grade: #2 or better
- Maximum span: 16' 4" (per span tables)

This assumes:
- Dead load: 10 PSF
- Live load: 40 PSF (residential)
- Standard BC Building Code requirements

You: How about for a game character controller?

[game_development] Kalki:
To create a Unity character controller, you'll need...
```

---

## 🔧 Technical Details

### Lazy Loading

Heavy components are loaded on-demand:

```python
async def _initialize_systems(self):
    if self.supreme_hub is None:
        self.supreme_hub = SupremeControlHub()
    
    if self.orchestrator is None:
        self.orchestrator = KalkiOrchestrator()
        await self.orchestrator.initialize_system()
```

**Benefits:**
- Fast startup time
- Only loads what's needed
- Reduces memory usage

### Error Handling

Graceful fallbacks:

```python
if domain_processing_fails:
    # Fallback to general orchestrator
    return await self._process_general_query(user_input)
```

### Response Formatting

Smart formatting based on content:

```python
if "```" in response_text or "#" in response_text[:50]:
    # Format as markdown
    console.print(Markdown(response_text))
else:
    # Plain text
    console.print(response_text)
```

---

## 🚀 Usage

### Start the Chatbot

```bash
python apps/kalki_unified_chat.py
```

### Example Queries

**Construction:**
- "What size joists for a 16 foot span?"
- "How do I calculate foundation loads?"
- "Create a new construction project"

**Game Development:**
- "How do I create a Unity character controller?"
- "Make a 2D platformer game"
- "What's the best way to handle collisions?"

**Robotics:**
- "Design a PID controller for a robot arm"
- "How do I implement SLAM?"
- "Calculate forward kinematics for a 6-DOF arm"

**General:**
- "Explain quantum computing"
- "What is machine learning?"
- "How does photosynthesis work?"

---

## 📊 Architecture Benefits

1. **Single Entry Point** - One chatbot for everything
2. **Automatic Domain Detection** - No configuration needed
3. **Intelligent Routing** - Uses best handler for each query
4. **Context Awareness** - Maintains conversation context
5. **Extensible** - New domains automatically work
6. **User-Friendly** - Beautiful CLI interface
7. **Performance** - Lazy loading, caching, optimization

---

## 🔮 Future Enhancements

1. **LLM-Based Domain Detection** - More accurate classification
2. **Multi-Domain Queries** - Handle queries spanning multiple domains
3. **Web Interface** - Browser-based chat UI
4. **API Endpoint** - REST API for integration
5. **Context Persistence** - Save conversations to database
6. **Voice Interface** - Speech input/output
7. **Visual Understanding** - Image analysis in chat

---

## 📝 Summary

The Kalki chatbot is a **unified, intelligent interface** that:

1. **Auto-detects** which domain your query belongs to
2. **Routes intelligently** to the best handler (copilot, domain, or general AI)
3. **Maintains context** across the conversation
4. **Provides beautiful** CLI interface
5. **Scales automatically** as new domains are added

It leverages the full power of Kalki's domain expertise while providing a simple, user-friendly interface for all capabilities.

