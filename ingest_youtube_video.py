#!/usr/bin/env python3
"""
Quick script to ingest a YouTube video into KALKI
Usage: python ingest_youtube_video.py <youtube_url>
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.youtube_ingestion import YouTubeIngestionSystem
from rich.console import Console
from rich.panel import Panel

console = Console()

async def main():
    # Video URL from user
    video_url = "https://youtu.be/LA-hZDnn5Hc?si=1COZ_2rjxXY1eiTJ"
    
    # Also accept full YouTube URL format
    if "youtu.be" in video_url:
        # Convert short URL to full format for better compatibility
        video_id = video_url.split("youtu.be/")[1].split("?")[0]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Allow URL from command line if provided
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    
    console.print(Panel.fit(
        "[bold cyan]🎥 KALKI YouTube Video Ingestion[/bold cyan]\n\n"
        f"[yellow]Video URL:[/yellow] {video_url}\n\n"
        "[dim]This will download, transcribe, and analyze the video...[/dim]",
        border_style="cyan"
    ))
    
    try:
        # Initialize system
        system = YouTubeIngestionSystem()
        
        # Check dependencies
        deps_ok, missing = system._check_dependencies()
        if not deps_ok:
            console.print(f"[red]❌ Missing dependencies:[/red]")
            for dep in missing:
                console.print(f"   • {dep}")
            console.print("\n[yellow]Install with:[/yellow]")
            console.print("   pip install yt-dlp openai-whisper moviepy opencv-python")
            console.print("   brew install ffmpeg  # macOS")
            return
        
        # Ingest video
        console.print("\n[cyan]🚀 Starting ingestion...[/cyan]\n")
        
        result = await system.ingest_youtube_video(
            url=video_url,
            extract_knowledge=True
        )
        
        if result.get("status") == "success":
            metadata = result["result"]["metadata"]
            transcript = result["result"]["transcript"]
            frames = result["result"]["key_frames"]
            domain = result["result"]["domain"]
            
            # Display results
            console.print("\n[bold green]✅ Video Successfully Ingested![/bold green]\n")
            
            console.print(Panel(
                f"[bold]Title:[/bold] {metadata['title']}\n"
                f"[bold]Channel:[/bold] {metadata['channel']}\n"
                f"[bold]Duration:[/bold] {metadata['duration']:.1f} seconds\n"
                f"[bold]Views:[/bold] {metadata['view_count']:,}\n"
                f"[bold]Domain:[/bold] {domain or 'Not detected'}\n"
                f"[bold]Transcript Length:[/bold] {len(transcript):,} characters\n"
                f"[bold]Key Frames:[/bold] {len(frames)}",
                title="Video Information",
                border_style="green"
            ))
            
            # Show transcript preview
            if transcript:
                preview = transcript[:500] + "..." if len(transcript) > 500 else transcript
                console.print("\n[bold]Transcript Preview:[/bold]")
                console.print(f"[dim]{preview}[/dim]")
            
            # Show frame analysis preview
            if frames:
                console.print(f"\n[bold]Key Frames Extracted:[/bold] {len(frames)}")
                for i, frame in enumerate(frames[:3], 1):  # Show first 3
                    if frame.get("description"):
                        console.print(f"  [cyan]Frame {i}[/cyan] (t={frame['timestamp']:.1f}s): {frame['description'][:100]}...")
            
            # Show where files are stored
            console.print("\n[bold]Files Stored:[/bold]")
            console.print(f"  • Audio: {result['result']['audio_path']}")
            if result['result']['video_path']:
                console.print(f"  • Video: {result['result']['video_path']}")
            console.print(f"  • Frames: data/youtube/frames/{result['result']['video_id']}/")
            console.print(f"  • Transcript: data/youtube/transcripts/{result['result']['video_id']}_transcript.txt")
            
            # Next steps
            console.print("\n[bold green]🎯 Next Steps:[/bold green]")
            console.print("  1. Query the video content in KALKI chat:")
            console.print("     > What did the video teach about [topic]?")
            console.print("     > What formulas or procedures were mentioned?")
            console.print("     > Summarize the key points from the video")
            console.print("\n  2. Or use the unified chat:")
            console.print("     python3 kalki.py")
            console.print("     Then ask questions about the video!")
            
        else:
            console.print(f"\n[red]❌ Ingestion failed:[/red] {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        console.print(f"\n[red]❌ Error:[/red] {e}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())

