# 🎥 KALKI YouTube Ingestion Guide

**Learn from YouTube videos - Multi-modal knowledge extraction**

---

## 🚀 Quick Start

### **Install Dependencies**

```bash
pip install yt-dlp openai-whisper moviepy opencv-python
```

### **Basic Usage**

```python
from modules.youtube_ingestion import YouTubeIngestionSystem

# Initialize system
youtube_system = YouTubeIngestionSystem()

# Ingest a single video
result = await youtube_system.ingest_youtube_video(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    extract_knowledge=True
)

print(result)
```

---

## 📋 Features

### **1. Video Download**
- Downloads audio (MP3) and/or video
- Extracts metadata (title, description, channel, etc.)
- Quality selection

### **2. Audio Transcription**
- **Whisper** (default): High-quality transcription with timestamps
- **AudioAgent** (fallback): Rule-based transcription
- Language detection and support

### **3. Visual Content Extraction**
- Extracts key frames from video
- Analyzes frames with Vision Agent
- Uses Llama 3.2 Vision 11B for detailed descriptions
- Timestamp tracking

### **4. Domain-Aware Knowledge Extraction**
- Auto-detects domain from video content
- Integrates with Hybrid Learning System
- Extracts structured knowledge (formulas, procedures, etc.)
- Stores in domain-specific databases

### **5. Vector Embeddings**
- Stores transcript in vector DB for semantic search
- Enables querying video content
- Cross-modal search (text + visual)

---

## 🔧 API Reference

### **YouTubeIngestionSystem**

#### **`ingest_youtube_video(url, extract_knowledge=True, domain_hint=None)`**

Complete ingestion pipeline for a YouTube video.

**Parameters:**
- `url` (str): YouTube video URL
- `extract_knowledge` (bool): Extract knowledge using Hybrid Learning System
- `domain_hint` (str, optional): Domain hint for knowledge extraction

**Returns:**
```python
{
    "status": "success",
    "result": {
        "video_id": "VIDEO_ID",
        "metadata": {...},
        "transcript": "Full transcript text...",
        "transcript_timestamps": [...],
        "key_frames": [...],
        "audio_path": "path/to/audio.mp3",
        "video_path": "path/to/video.mp4",
        "extraction_time": 45.2,
        "domain": "construction"  # Inferred domain
    },
    "download": {...},
    "transcription": {...},
    "frames": {...},
    "knowledge": {...},
    "vector_db": {...}
}
```

#### **`download_video(url, download_audio=True, download_video=False, quality='best')`**

Download YouTube video and/or audio.

**Parameters:**
- `url` (str): YouTube video URL
- `download_audio` (bool): Download audio (MP3)
- `download_video` (bool): Download video (larger files)
- `quality` (str): Quality setting ('best', 'worst', '720p', etc.)

#### **`transcribe_audio(audio_path, use_whisper=True, language='en')`**

Transcribe audio from video.

**Parameters:**
- `audio_path` (str): Path to audio file
- `use_whisper` (bool): Use Whisper for transcription
- `language` (str): Language code (default: 'en')

#### **`extract_key_frames(video_path, num_frames=10, analyze_frames=True)`**

Extract key frames from video for visual content analysis.

**Parameters:**
- `video_path` (str): Path to video file
- `num_frames` (int): Number of key frames to extract
- `analyze_frames` (bool): Analyze frames with Vision Agent

#### **`batch_ingest(urls, extract_knowledge=True)`**

Batch ingest multiple YouTube videos.

**Parameters:**
- `urls` (List[str]): List of YouTube URLs
- `extract_knowledge` (bool): Extract knowledge from each video

---

## 💡 Usage Examples

### **Example 1: Ingest Construction Tutorial**

```python
from modules.youtube_ingestion import YouTubeIngestionSystem

system = YouTubeIngestionSystem()

# Ingest a construction tutorial
result = await system.ingest_youtube_video(
    url="https://www.youtube.com/watch?v=construction_tutorial",
    extract_knowledge=True,
    domain_hint="construction"  # Optional: hint the domain
)

# Access results
print(f"Title: {result['result']['metadata']['title']}")
print(f"Transcript: {result['result']['transcript'][:200]}...")
print(f"Domain: {result['result']['domain']}")
print(f"Key Frames: {len(result['result']['key_frames'])}")
```

### **Example 2: Batch Ingest Playlist**

```python
# List of video URLs
playlist_urls = [
    "https://www.youtube.com/watch?v=video1",
    "https://www.youtube.com/watch?v=video2",
    "https://www.youtube.com/watch?v=video3",
]

# Batch ingest
results = await system.batch_ingest(
    urls=playlist_urls,
    extract_knowledge=True
)

print(f"Processed: {results['successful']}/{results['total']}")
```

### **Example 3: Download Only (No Processing)**

```python
# Just download video/audio
download_result = await system.download_video(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    download_audio=True,
    download_video=False
)

print(f"Audio saved to: {download_result['audio_path']}")
```

### **Example 4: Extract Knowledge from Transcript**

```python
# After ingestion, extract knowledge
if result['result']['transcript']:
    # The transcript is automatically processed by Hybrid Learning System
    # You can query it like any other knowledge:
    
    from modules.hybrid_learning_system import get_hybrid_system
    hybrid = get_hybrid_system()
    
    # Query video content
    answer = await hybrid.query(
        "What construction techniques were mentioned in the video?",
        domain="construction"
    )
```

---

## 🔄 Integration with KALKI

### **1. Domain-Aware Extraction**

YouTube ingestion automatically:
- Detects domain from video content
- Uses domain-specific extractors
- Stores knowledge in domain databases

### **2. Vector DB Integration**

Transcripts are stored in vector DB for:
- Semantic search across all videos
- Querying video content
- Cross-modal retrieval

### **3. Hybrid Learning System**

Integrates with existing knowledge extraction:
- Extracts formulas, procedures, design rules
- Stores in structured databases
- Enables precise lookups

### **4. Vision Intelligence**

Uses KALKI's vision capabilities:
- Analyzes key frames
- Extracts visual information
- Cross-modal understanding

---

## 📊 Output Structure

### **Files Created**

```
data/youtube/
├── videos/          # Downloaded video files
├── audio/           # Extracted audio files (MP3)
├── frames/          # Key frame images
│   └── VIDEO_ID/
│       ├── frame_0000.jpg
│       ├── frame_0001.jpg
│       └── ...
└── transcripts/     # Transcript text files
    └── VIDEO_ID_transcript.txt
```

### **Knowledge Storage**

- **Vector DB:** Transcript chunks for semantic search
- **Structured DB:** Extracted knowledge (formulas, procedures, etc.)
- **Domain DBs:** Domain-specific knowledge storage

---

## 🎯 Use Cases

### **1. Learn from Tutorials**
- Ingest construction tutorials
- Extract procedures and techniques
- Query: "How do I frame a wall?"

### **2. Educational Content**
- Ingest educational videos
- Extract concepts and formulas
- Build knowledge base from video content

### **3. Technical Documentation**
- Ingest technical explainer videos
- Extract design rules and best practices
- Cross-reference with PDF documentation

### **4. Multi-Modal Learning**
- Combine video + PDF knowledge
- Visual + textual understanding
- Comprehensive knowledge base

---

## ⚙️ Configuration

### **Output Directory**

```python
# Custom output directory
system = YouTubeIngestionSystem(output_dir="data/my_youtube/")
```

### **Whisper Model Size**

Edit `transcribe_audio()` method to change model:
- `tiny`: Fastest, least accurate
- `base`: Balanced (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

```python
model = whisper.load_model("medium")  # Change here
```

### **Frame Extraction**

```python
# Extract more frames
frames = await system.extract_key_frames(
    video_path,
    num_frames=20,  # More frames
    analyze_frames=True
)
```

---

## 🐛 Troubleshooting

### **Missing Dependencies**

```bash
# Install all dependencies
pip install yt-dlp openai-whisper moviepy opencv-python
```

### **FFmpeg Required**

Whisper and yt-dlp require FFmpeg:
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`
- **Windows:** Download from https://ffmpeg.org/

### **Memory Issues**

For large videos:
- Use `download_audio=True, download_video=False` (audio only)
- Reduce `num_frames` in frame extraction
- Use smaller Whisper model (`tiny` or `base`)

### **Transcription Quality**

- Use `use_whisper=True` for best quality
- Specify `language` parameter if known
- Larger Whisper models = better accuracy

---

## 🚀 Advanced Features

### **Custom Domain Extractors**

```python
# Create custom extractor for video content
from modules.domains.construction_domain.knowledge_extractors import ConstructionExtractor

# System will use domain-specific extractors automatically
```

### **Streaming Processing**

```python
# Process videos as they download
async for chunk in system.stream_ingest(url):
    print(f"Processed: {chunk}")
```

### **Real-Time Transcription**

```python
# Transcribe while video is playing (future feature)
transcript_stream = await system.stream_transcribe(audio_path)
```

---

## 📈 Performance

### **Typical Processing Times**

- **Download (audio):** 10-30 seconds
- **Download (video):** 30-120 seconds
- **Transcription (Whisper base):** 30-60 seconds per 10 minutes
- **Frame extraction:** 5-15 seconds
- **Knowledge extraction:** 10-30 seconds

### **Storage Requirements**

- **Audio (MP3):** ~1-2 MB per minute
- **Video (720p):** ~10-20 MB per minute
- **Frames:** ~100-200 KB per frame
- **Transcripts:** ~1-2 KB per minute

---

## 🎉 Next Steps

1. **Install dependencies:** `pip install yt-dlp openai-whisper moviepy opencv-python`
2. **Try a video:** Use the quick start example
3. **Query knowledge:** Use Hybrid Learning System to query video content
4. **Batch process:** Ingest playlists or channels
5. **Integrate:** Use in your KALKI workflows

---

*YouTube Ingestion System - Learn from video content!*

