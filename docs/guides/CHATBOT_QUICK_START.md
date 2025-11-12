# 🤖 Kalki Chatbot - Quick Start

## ✅ The Chatbot is Ready!

The unified chatbot has been created at: **`apps/kalki_unified_chat.py`**

---

## 🚀 How to Run

```bash
python3 apps/kalki_unified_chat.py
```

Or if you're in a virtual environment:

```bash
python apps/kalki_unified_chat.py
```

---

## 💬 What It Does

The chatbot automatically:
- **Detects domains** from your questions (Construction, Game Dev, Robotics, etc.)
- **Routes intelligently** to the right system
- **Handles all Kalki capabilities** in one place

---

## 📝 Example Queries

### Construction
```
You: What size joists for a 16 foot span?
[construction] Kalki: For a 16-foot span...
```

### Game Development
```
You: How do I create a Unity character controller?
[game_development] Kalki: To create a Unity character controller...
```

### Robotics
```
You: Design a PID controller for a robot arm
[robotics] Kalki: A PID controller consists of...
```

### General Questions
```
You: Explain quantum computing
[general] Kalki: Quantum computing is...
```

---

## 🎮 Commands

While chatting, you can use:

- `/help` - Show all commands
- `/domains` - List available domains
- `/stats` - Show chat statistics
- `/history` - Show recent chat history
- `/clear` - Clear chat history
- `/project <id>` - Set project context
- `/exit` or `quit` - Exit chatbot

---

## ✨ Features

✅ **Automatic Domain Detection** - No configuration needed  
✅ **Intelligent Routing** - Uses the right system for each query  
✅ **Beautiful Interface** - Rich formatting with colors  
✅ **Chat History** - Remembers your conversation  
✅ **Multi-Domain Support** - All Kalki domains in one place  

---

## 🔧 Technical Details

The chatbot uses:
- **DomainRegistry** - Auto-detects which domain your query needs
- **SupremeControlHub** - Handles domain-specific queries
- **KalkiOrchestrator** - Handles general queries with 20-phase intelligence

---

## 🎯 Next Steps

1. **Run the chatbot**: `python3 apps/kalki_unified_chat.py`
2. **Try different domains** - Ask questions about construction, game dev, robotics, etc.
3. **Use commands** - Try `/domains` to see what's available
4. **Check stats** - Use `/stats` to see usage statistics

---

## 🐛 Troubleshooting

If you get import errors:
- Make sure you're in the project root directory
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version: `python3 --version` (should be 3.8+)

---

**Enjoy your Kalki chatbot!** 🎉


