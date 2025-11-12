# 💬 How to Use the Kalki Chatbot

## 🚀 Quick Start

### Step 1: Open Terminal
Open your terminal/command prompt and navigate to the Kalki project directory:

```bash
cd /Users/kashish/Desktop/Kalki
```

### Step 2: Run the Chatbot
```bash
python3 apps/kalki_unified_chat.py
```

Or if you're using a virtual environment:
```bash
python apps/kalki_unified_chat.py
```

### Step 3: Start Chatting!
Once it starts, you'll see a welcome screen. Just type your questions!

---

## 📝 Example Usage

### Construction Questions
```
You: What size joists do I need for a 16 foot span?
```

### Game Development
```
You: How do I create a Unity character controller?
```

### Robotics
```
You: Design a PID controller for a robot arm
```

### General Questions
```
You: Explain quantum computing
```

---

## 🎮 Available Commands

While chatting, you can use these commands:

- **`/help`** - Show all commands and features
- **`/domains`** - List all available domains
- **`/stats`** - Show your chat statistics
- **`/history`** - Show recent chat history
- **`/clear`** - Clear your chat history
- **`/project <id>`** - Set a project ID for context
- **`/exit`** or **`quit`** - Exit the chatbot

---

## 💡 Tips

1. **Just ask naturally** - The chatbot automatically detects which domain your question belongs to
2. **No configuration needed** - It works immediately
3. **Use `/help`** - If you're not sure what to do
4. **Check `/domains`** - To see what domains are available
5. **Set project context** - Use `/project <id>` if you're working on a specific project

---

## 🎯 What Happens When You Ask a Question?

1. The chatbot **automatically detects** which domain your question needs
2. It **routes your question** to the right system
3. You get a **context-aware response** with domain tags
4. The conversation is **saved in history** for context

---

## 🐛 Troubleshooting

### "Command not found: python3"
Try:
```bash
python apps/kalki_unified_chat.py
```

### Import errors
Make sure you're in the project directory:
```bash
cd /Users/kashish/Desktop/Kalki
```

### Module not found
Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📖 Full Example Session

```
$ python3 apps/kalki_unified_chat.py

┌─────────────────────────────────────────┐
│  🤖 KALKI Chatbot              │
│                                         │
│  Your AI assistant for ALL domains     │
│  Construction • Game Dev • Robotics... │
│                                         │
│  Type your question or /help           │
└─────────────────────────────────────────┘

Available domains: construction, game_development, robotics, aerospace, power_systems

You: What size joists for 16 foot span?

🔍 Detected domain: construction
[construction] Kalki:
For a 16-foot span, you'll need 2x10 or 2x12 joists depending on...

You: /domains

┌─────────────────────────────────────┐
│  Available Kalki Domains             │
├─────────────┬──────────┬───────────┤
│ Domain      │ Status   │ Knowledge │
├─────────────┼──────────┼───────────┤
│ construction│ ✅ Loaded│ 150       │
│ game_dev    │ ✅ Loaded│ 45        │
└─────────────┴──────────┴───────────┘

You: /stats

┌─────────────────────────────┐
│  Chat Statistics            │
├──────────────┬──────────────┤
│ Metric       │ Value        │
├──────────────┼──────────────┤
│ Total Queries│ 1            │
│ Domain Queries│ 1           │
│ Domains Used │ construction │
└──────────────┴──────────────┘

You: /exit
👋 Goodbye!
```

---

**That's it! Just run the command and start chatting!** 🎉


