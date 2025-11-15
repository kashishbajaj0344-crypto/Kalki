# 🎥 Ingest YouTube Video - Quick Guide

**Video URL:** https://youtu.be/LA-hZDnn5Hc?si=1COZ_2rjxXY1eiTJ

---

## 🚀 Quick Method: Run Script

I've created a script to ingest this video. Just run:

```bash
cd /Users/kashish/Desktop/Kalki
python3 ingest_youtube_video.py
```

The script will:
- ✅ Check dependencies
- ✅ Download the video
- ✅ Transcribe audio
- ✅ Extract key frames
- ✅ Analyze with Vision Agent
- ✅ Extract knowledge
- ✅ Show you the results

---

## 💬 Alternative: Use Chat Interface

1. **Start KALKI chat:**
   ```bash
   python3 kalki.py
   ```

2. **In the chat, type:**
   ```
   youtube ingest https://youtu.be/LA-hZDnn5Hc?si=1COZ_2rjxXY1eiTJ
   ```

3. **Wait for ingestion to complete**

4. **Then query the video:**
   ```
   What did the video teach about?
   What were the key points?
   What formulas or procedures were mentioned?
   ```

---

## 📋 Prerequisites

Make sure you have dependencies installed:

```bash
pip install yt-dlp openai-whisper moviepy opencv-python
brew install ffmpeg  # macOS
```

---

## 🎯 What Happens

1. **Download:** Video and audio files saved to `data/youtube/`
2. **Transcribe:** Audio transcribed with Whisper
3. **Extract Frames:** 10 key frames extracted and analyzed
4. **Domain Detection:** Auto-detects domain (construction, game dev, etc.)
5. **Knowledge Extraction:** Extracts formulas, procedures, design rules
6. **Storage:** Everything stored in vector DB and structured DBs

---

## 🔍 After Ingestion

You can query the video content like any other knowledge:

```
> What did the video cover?
> What techniques were shown?
> What calculations or formulas were mentioned?
> Summarize the main points
```

KALKI will use the transcript, frame analysis, and extracted knowledge to answer!

---

*Ready to analyze! Run the script or use the chat interface.*

