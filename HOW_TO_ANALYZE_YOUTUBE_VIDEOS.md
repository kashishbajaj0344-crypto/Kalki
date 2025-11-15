# 🎥 How to Analyze YouTube Videos with KALKI

**Step-by-step guide to make KALKI learn from YouTube videos**

---

## 🚀 Step 1: Install Dependencies

First, install the required packages:

```bash
pip install yt-dlp openai-whisper moviepy opencv-python
```

**Also install FFmpeg** (required for audio/video processing):
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/download.html

---

## 🎯 Step 2: Start KALKI Chat

Open your terminal and run:

```bash
cd /Users/kashish/Desktop/Kalki
python3 kalki.py
```

Or directly:
```bash
python3 apps/kalki_unified_chat.py
```

You'll see the KALKI welcome screen.

---

## 📹 Step 3: Ingest a YouTube Video

In the chat, type:

```
youtube ingest https://www.youtube.com/watch?v=VIDEO_ID
```

Or use the short form:
```
yt ingest https://www.youtube.com/watch?v=VIDEO_ID
```

**Replace `VIDEO_ID` with the actual video ID from YouTube.**

---

## ✨ What Happens

KALKI will:

1. **Download** the video and extract audio
2. **Transcribe** the audio using Whisper (high-quality speech-to-text)
3. **Extract key frames** from the video (10 frames by default)
4. **Analyze frames** using Vision Agent + Llama 3.2 Vision
5. **Detect domain** automatically (construction, game dev, etc.)
6. **Extract knowledge** (formulas, procedures, design rules)
7. **Store everything** in vector DB for querying

You'll see progress messages like:
```
🎥 Ingesting YouTube video...
✅ Video ingested: "How to Frame a Wall"
📝 Transcript: 5,234 characters
🎬 Key frames: 10
🏷️  Domain: construction
```

---

## 🔍 Step 4: Query the Video Content

After ingestion, you can ask questions about the video:

```
What construction techniques were shown in the video?
```

```
What formulas or calculations were mentioned?
```

```
What visual elements did the video show?
```

```
Summarize the key points from the video
```

KALKI will use the transcript, frame analysis, and extracted knowledge to answer!

---

## 💡 Complete Example

```bash
# 1. Start KALKI
$ python3 kalki.py

# 2. In the chat, ingest a video
> youtube ingest https://www.youtube.com/watch?v=dQw4w9WgXcQ

🎥 Ingesting YouTube video...
✅ Video ingested: "Example Tutorial"
📝 Transcript: 3,456 characters
🎬 Key frames: 10
🏷️  Domain: construction

✅ Successfully ingested YouTube video!
You can now query this video's content!

# 3. Ask questions about the video
> What did the video teach about framing?

[construction] Kalki: Based on the video transcript and frame analysis, 
the video covered the following framing techniques:
- Proper spacing of studs (16" on center)
- Header sizing for door openings
- Corner framing techniques
...

# 4. Query specific details
> What formulas were mentioned in the video?

[construction] Kalki: The video mentioned these formulas:
- Load calculation: W = (L × W × D) / 144
- Span table lookup for 2x8 joists
...
```

---

## 🎮 Alternative: Python Script

You can also use it programmatically:

```python
import asyncio
from modules.youtube_ingestion import YouTubeIngestionSystem

async def main():
    system = YouTubeIngestionSystem()
    
    # Ingest a video
    result = await system.ingest_youtube_video(
        url="https://www.youtube.com/watch?v=VIDEO_ID",
        extract_knowledge=True
    )
    
    print(f"Title: {result['result']['metadata']['title']}")
    print(f"Transcript: {result['result']['transcript'][:200]}...")
    print(f"Domain: {result['result']['domain']}")

asyncio.run(main())
```

---

## 📊 What Gets Stored

### **Files Created:**
```
data/youtube/
├── videos/          # Downloaded video files
├── audio/           # Extracted audio (MP3)
├── frames/          # Key frame images
│   └── VIDEO_ID/
│       ├── frame_0000.jpg
│       └── ...
└── transcripts/     # Transcript text files
    └── VIDEO_ID_transcript.txt
```

### **Knowledge Storage:**
- **Vector DB:** Transcript chunks for semantic search
- **Structured DB:** Extracted formulas, procedures, design rules
- **Domain DBs:** Domain-specific knowledge

---

## 🔧 Advanced Options

### **Batch Ingest Multiple Videos:**

```python
from modules.youtube_ingestion import YouTubeIngestionSystem

system = YouTubeIngestionSystem()

urls = [
    "https://www.youtube.com/watch?v=video1",
    "https://www.youtube.com/watch?v=video2",
    "https://www.youtube.com/watch?v=video3",
]

results = await system.batch_ingest(urls, extract_knowledge=True)
```

### **Custom Settings:**

```python
# Download only audio (faster, smaller)
download_result = await system.download_video(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    download_audio=True,
    download_video=False  # Skip video download
)

# Extract more frames
frames = await system.extract_key_frames(
    video_path,
    num_frames=20,  # More frames
    analyze_frames=True
)
```

---

## 🐛 Troubleshooting

### **"Missing dependencies" error:**
```bash
pip install yt-dlp openai-whisper moviepy opencv-python
brew install ffmpeg  # macOS
```

### **"FFmpeg not found":**
- Install FFmpeg (see Step 1)
- Make sure it's in your PATH

### **"Whisper model download":**
- First run will download Whisper model (~150MB)
- This is automatic, just wait

### **Memory issues with large videos:**
- Use `download_audio=True, download_video=False` (audio only)
- Reduce `num_frames` parameter
- Use smaller Whisper model (`tiny` or `base`)

---

## 🎯 Tips

1. **Start with short videos** (5-10 minutes) to test
2. **Use domain hints** if you know the domain:
   ```
   youtube ingest <URL> --domain=construction
   ```
3. **Query immediately after ingestion** - knowledge is available right away
4. **Batch process playlists** - ingest multiple videos at once
5. **Check `/help`** in chat for all commands

---

## 📚 More Information

- **Full Guide:** See `YOUTUBE_INGESTION_GUIDE.md`
- **Quick Start:** See `YOUTUBE_QUICK_START.md`
- **API Reference:** See `modules/youtube_ingestion.py`

---

## ✅ Summary

**To analyze YouTube videos:**

1. ✅ Install: `pip install yt-dlp openai-whisper moviepy opencv-python`
2. ✅ Install FFmpeg: `brew install ffmpeg`
3. ✅ Start KALKI: `python3 kalki.py`
4. ✅ Ingest: `youtube ingest <URL>`
5. ✅ Query: Ask questions about the video!

**That's it! KALKI will learn from the video and you can query it like any other knowledge source.**

---

*Happy learning! 🎥*

