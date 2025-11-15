# 🎥 YouTube Ingestion - Quick Start

**Make KALKI learn from YouTube videos!**

---

## ⚡ Quick Setup

### **1. Install Dependencies**

```bash
pip install yt-dlp openai-whisper moviepy opencv-python
```

**Note:** You also need FFmpeg:
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/

### **2. Use in Chat**

```bash
python kalki.py --chat
```

Then type:
```
youtube ingest https://www.youtube.com/watch?v=VIDEO_ID
```

Or short form:
```
yt ingest https://www.youtube.com/watch?v=VIDEO_ID
```

---

## 🎯 What It Does

1. **Downloads** video and audio
2. **Transcribes** audio using Whisper
3. **Extracts** key frames for visual content
4. **Analyzes** frames with Vision Agent
5. **Detects** domain automatically
6. **Extracts** knowledge (formulas, procedures, etc.)
7. **Stores** in vector DB for querying

---

## 💡 Example

```bash
# In KALKI chat
> youtube ingest https://www.youtube.com/watch?v=construction_tutorial

🎥 Ingesting YouTube video...
✅ Video ingested: "How to Frame a Wall"
📝 Transcript: 5,234 characters
🎬 Key frames: 10
🏷️  Domain: construction

✅ Successfully ingested YouTube video!
You can now query this video's content!
```

Then query:
```
> What construction techniques were shown in the video?
```

---

## 📚 Full Documentation

See `YOUTUBE_INGESTION_GUIDE.md` for complete documentation.

---

*Start learning from YouTube videos today!*

