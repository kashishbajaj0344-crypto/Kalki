# 🚀 How to Run GameDev Copilot App

## Quick Start

### **Option 1: Direct Streamlit Command** (Recommended)

```bash
cd /Users/kashish/Desktop/Kalki
streamlit run apps/gamedev_copilot_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

### **Option 2: Using Python Module**

```bash
cd /Users/kashish/Desktop/Kalki
python3 -m streamlit run apps/gamedev_copilot_app.py
```

---

### **Option 3: Make it Executable**

```bash
chmod +x apps/gamedev_copilot_app.py
./apps/gamedev_copilot_app.py
```

---

## 📋 Prerequisites

Make sure you have Streamlit installed:

```bash
pip3 install streamlit
```

If you're using the project's virtual environment:

```bash
source kalki_env/bin/activate  # or your venv path
pip install streamlit
```

---

## 🎮 Using the App

1. **Start the app** using one of the commands above
2. **Browser opens** automatically at `http://localhost:8501`
3. **If browser doesn't open**, manually go to: `http://localhost:8501`
4. **Start creating games!**

---

## 🛑 Stopping the App

Press `Ctrl+C` in the terminal to stop the app.

---

## 🔧 Troubleshooting

**Port already in use?**
```bash
streamlit run apps/gamedev_copilot_app.py --server.port 8502
```

**Can't find the app?**
- Make sure you're in the project root: `/Users/kashish/Desktop/Kalki`
- Check the file exists: `ls apps/gamedev_copilot_app.py`

**Import errors?**
- Make sure you're using the correct Python environment
- Check that all dependencies are installed: `pip install -r requirements.txt`

---

## 📱 Access from Other Devices

If you want to access from another device on your network:

```bash
streamlit run apps/gamedev_copilot_app.py --server.address 0.0.0.0
```

Then access via: `http://YOUR_IP_ADDRESS:8501`

---

## ✅ Quick Test

To verify everything works:

```bash
cd /Users/kashish/Desktop/Kalki
streamlit run apps/gamedev_copilot_app.py
```

You should see:
- Terminal output showing "You can now view your Streamlit app..."
- Browser opens automatically
- App loads with "🎮 GameDev Copilot" header

---

## 🎉 That's It!

Once the app is running, you can:
- Create new games
- View existing projects
- Deploy and polish games
- All through a beautiful web interface!

