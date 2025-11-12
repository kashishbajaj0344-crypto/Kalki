# 🎮 GameDevCopilot - Complete Usage Guide

## 🚀 How to Use

### **Option 1: Through Unified Chatbot** (Recommended)

```bash
# Start unified chatbot
python3 apps/kalki_unified_chat.py
```

**Then simply chat:**
```
You: make me a game like car jam
KALKI: [Researches car jam] 
       "I found car jam is a puzzle game. 
        What platforms? (Android/iOS/both)"

You: Both
KALKI: "What engine? (Unity/Flutter/React Native)"

You: Unity
KALKI: "Monetization? (Premium/Freemium/Ads)"

You: Freemium
KALKI: [Creates project] [Generates code] [Builds] [Deploys] [Polishes]
       "✅ Your game is ready! 
        Code: output/games/{project_id}/
        Build: Run build_android.sh or build_ios.sh"
```

### **Option 2: Direct Python API**

```python
import asyncio
from modules.game_dev_copilot import GameDevCopilot

async def main():
    copilot = GameDevCopilot()
    
    # Start project
    result = await copilot.start_new_game_project("make me a carjam style game")
    print(result['message'])
    
    # Answer questions
    while result.get('status') == 'needs_input':
        next_q = result.get('next_question')
        if next_q:
            print(f"\n{next_q.question}")
            answer = input("Your answer: ")
            result = await copilot.answer_question(result['session_id'], answer)
            print(result['message'])
    
    # Build complete game
    if result.get('status') == 'project_created':
        build_result = await copilot.build_complete_game(
            result['session_id'],
            auto_deploy=True,
            auto_polish=True,
            polish_level="standard"
        )
        print(build_result['message'])

asyncio.run(main())
```

---

## 📋 Complete Workflow

### **Step 1: Research**
- Automatically researches game references (car jam, flappy bird, etc.)
- Extracts genre, mechanics, platforms, monetization
- Uses `AutonomousResearchSystem` for web search

### **Step 2: Guided Questions**
- Asks critical questions: platforms, engine, monetization
- Asks high-priority questions: genre, art style
- Asks medium-priority questions: team size, budget
- Provides context and suggestions for each question

### **Step 3: Code Generation**
- Generates complete source code for chosen engine
- Creates project structure
- Generates all necessary files
- Includes setup instructions

### **Step 4: Deployment**
- Generates build scripts
- Creates deployment guides
- Configures for target platforms
- Provides step-by-step instructions

### **Step 5: Polish**
- Generates test suites
- Creates optimization guides
- Adds UI/UX polish
- Provides bug fix recommendations

---

## 🎯 Supported Engines & Platforms

### **Engines:**
- ✅ Unity (C#)
- ✅ Flutter (Dart)
- ✅ React Native (JavaScript/TypeScript)
- ✅ Web (HTML/CSS/JavaScript)
- ✅ Generic (Python, C++, etc.)

### **Platforms:**
- ✅ Android
- ✅ iOS
- ✅ Web
- ✅ PC (via Unity/Flutter)

---

## 📁 Output Structure

```
output/games/{project_id}/
├── Assets/                    # Unity assets
│   └── Scripts/
│       ├── GameManager.cs
│       └── PlayerController.cs
├── lib/                       # Flutter source
│   └── main.dart
├── App.js                     # React Native
├── index.html                 # Web game
├── game.js
├── styles.css
├── SETUP_INSTRUCTIONS.md
├── build_android.sh
├── build_ios.sh
├── tests/
│   └── game_tests.py
├── OPTIMIZATION.md
├── BUG_FIXES.md
├── UIPolish.cs
└── DEPLOYMENT.md
```

---

## ✨ Features

### **Research Capability**
- ✅ Web search for game references
- ✅ Extracts mechanics, genre, platforms
- ✅ Understands game styles

### **Smart Questions**
- ✅ Asks only what's needed
- ✅ Provides context and suggestions
- ✅ Prioritizes critical requirements

### **Code Generation**
- ✅ Complete, runnable code
- ✅ Multiple engine support
- ✅ Template fallbacks for reliability

### **Deployment**
- ✅ Build scripts for all platforms
- ✅ Deployment guides
- ✅ Store upload instructions

### **Polish**
- ✅ Testing
- ✅ Optimization
- ✅ UI/UX improvements
- ✅ Bug fixes

---

## 🎉 Result

**You get a complete, deployable game/app from a simple request like "make me a game like car jam"!**

The system:
1. ✅ Researches the game
2. ✅ Asks the right questions
3. ✅ Generates all code
4. ✅ Builds and deploys
5. ✅ Polishes extensively

**Ready to replace human developers!** 🚀

