# 💬 How to Chat with Kalki

## Quick Start

### Option 1: Unified Chat (Recommended) ✅

**The main chat interface** - supports all domains and features:

```bash
cd /Users/kashish/Desktop/Kalki
python3 apps/kalki_unified_chat.py
```

**Features:**
- ✅ Auto-detects domain from your queries
- ✅ Supports all domains (Construction, Game Dev, etc.)
- ✅ Beautiful CLI interface with Rich
- ✅ Chat history and context
- ✅ YouTube ingestion support
- ✅ All Kalki capabilities

**Usage:**
```
$ python3 apps/kalki_unified_chat.py

╭─────────────────────────────────────────╮
│  🤖 KALKI Unified Chat                  │
│  Your AI Assistant for Everything       │
╰─────────────────────────────────────────╮

You: What is artificial intelligence?
Kalki: [Responds with improved conversation quality!]

You: Build me a solitaire game
Kalki: [Routes to Game Dev Copilot]

You: /help
Kalki: [Shows available commands]
```

### Option 2: Simple Chat Interface

```bash
cd /Users/kashish/Desktop/Kalki
python3 src/chat_with_kalki.py
```

**Features:**
- ✅ Simple, direct chat interface
- ✅ General Kalki intelligence
- ✅ Good for quick questions

---

## Available Chat Interfaces

### 1. **kalki_unified_chat.py** (Main - Recommended) ⭐

**Location:** `apps/kalki_unified_chat.py`

**Best for:**
- Full-featured conversations
- Domain-specific queries (Game Dev, Construction, etc.)
- Multi-domain support
- YouTube ingestion
- All Kalki features

**Start:**
```bash
python3 apps/kalki_unified_chat.py
```

**Commands:**
- `/help` - Show help
- `/domains` - List available domains
- `/stats` - Show statistics
- `/clear` - Clear chat history
- `youtube ingest <URL>` - Ingest YouTube video
- `yt ingest <URL>` - Short form

### 2. **chat_with_kalki.py** (Simple)

**Location:** `src/chat_with_kalki.py`

**Best for:**
- Quick questions
- Simple conversations
- General AI queries

**Start:**
```bash
python3 src/chat_with_kalki.py
```

### 3. **Streamlit Apps** (Web Interface)

**Available apps:**
- `apps/kalki_app.py` - Basic web interface
- `apps/kalki_app_ai.py` - AI-powered interface
- `apps/kalki_app_enhanced.py` - Enhanced features
- `apps/kalki_app_proactive.py` - Proactive guidance

**Start:**
```bash
streamlit run apps/kalki_app.py
```

**Best for:**
- Web-based interface
- Visual interactions
- Project management
- Construction-specific workflows

---

## Recommended: Unified Chat

**Why use `kalki_unified_chat.py`:**

1. ✅ **All domains supported** - Game Dev, Construction, etc.
2. ✅ **Auto-detection** - Automatically routes to right handler
3. ✅ **Improved conversation quality** - Uses the fixes we just applied!
4. ✅ **Beautiful interface** - Rich console with colors and formatting
5. ✅ **Full features** - YouTube ingestion, domain routing, etc.

**Start chatting:**
```bash
cd /Users/kashish/Desktop/Kalki
python3 apps/kalki_unified_chat.py
```

**Example conversation:**
```
You: What is artificial intelligence?
Kalki: [Detailed, thoughtful response with 2048 tokens]

You: Can you explain that in more detail?
Kalki: [Remembers context, provides deeper explanation]

You: Build me a solitaire game
Kalki: [Routes to Game Dev Copilot, asks smart questions]
```

---

## Testing the Improved Conversation Quality

After the fixes we just applied, try these:

### Test 1: Simple Question
```
You: What is artificial intelligence?
```
**Expected:** Detailed, thoughtful response (up to 2048 tokens)

### Test 2: Follow-up (Tests Conversation History)
```
You: Can you explain that in more detail?
```
**Expected:** Remembers previous context, provides deeper explanation

### Test 3: Complex Reasoning
```
You: How does machine learning differ from traditional programming?
```
**Expected:** Comprehensive, detailed answer with proper reasoning

### Test 4: Domain-Specific
```
You: Build me a solitaire game
```
**Expected:** Routes to Game Dev Copilot, asks smart questions

---

## Troubleshooting

### If chat doesn't start:

1. **Check Python version:**
   ```bash
   python3 --version  # Should be 3.8+
   ```

2. **Check if models are loaded:**
   ```bash
   ls models/llama_3.1_8b/
   ```

3. **Check dependencies:**
   ```bash
   pip install rich transformers torch
   ```

### If conversation quality is still poor:

1. **Verify fixes are applied:**
   - Check `modules/llm.py` for `max_new_tokens=2048`
   - Check for `top_p=0.9`, `top_k=40`, `repetition_penalty=1.1`

2. **Try explicit parameters:**
   ```python
   response = await llm.generate(
       "Your question",
       max_new_tokens=2048,
       top_p=0.9,
       top_k=40,
       repetition_penalty=1.1
   )
   ```

---

## Quick Reference

| Interface | Command | Best For |
|-----------|---------|----------|
| **Unified Chat** ⭐ | `python3 apps/kalki_unified_chat.py` | Everything |
| Simple Chat | `python3 src/chat_with_kalki.py` | Quick questions |
| Web App | `streamlit run apps/kalki_app.py` | Web interface |

**Start with Unified Chat for the best experience!** 🚀

