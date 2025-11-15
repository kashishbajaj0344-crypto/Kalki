#!/usr/bin/env python3
"""
Analyze YouTube video for code snippets and detailed content
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from modules.youtube_ingestion import YouTubeIngestionSystem
from modules.llm import LLMEngine

console = Console()

async def analyze_video_code(video_id: str = "LA-hZDnn5Hc"):
    """Analyze video frames for code and provide detailed analysis"""
    
    console.print(Panel.fit(
        "[bold cyan]🔍 Analyzing Video for Code & Content[/bold cyan]\n\n"
        f"[yellow]Video ID:[/yellow] {video_id}",
        border_style="cyan"
    ))
    
    # Initialize systems
    youtube_system = YouTubeIngestionSystem()
    await youtube_system._ensure_components_loaded()
    
    # Read transcript
    transcript_file = Path(f"data/youtube/transcripts/{video_id}_transcript.txt")
    if transcript_file.exists():
        transcript = transcript_file.read_text()
        console.print(f"\n[green]✅ Loaded transcript: {len(transcript):,} characters[/green]")
    else:
        console.print(f"[red]❌ Transcript not found: {transcript_file}[/red]")
        return
    
    # Analyze frames for code
    frames_dir = Path(f"data/youtube/frames/{video_id}")
    if not frames_dir.exists():
        console.print(f"[red]❌ Frames directory not found: {frames_dir}[/red]")
        return
    
    frame_files = sorted(frames_dir.glob("*.jpg"))
    console.print(f"\n[cyan]📸 Analyzing {len(frame_files)} frames for code...[/cyan]\n")
    
    code_snippets = []
    frame_analyses = []
    
    llm = youtube_system._llm if youtube_system._llm else None
    
    for i, frame_file in enumerate(frame_files, 1):
        console.print(f"[dim]Analyzing frame {i}/{len(frame_files)}: {frame_file.name}[/dim]")
        
        try:
            # Use Vision Agent
            if youtube_system._vision_agent:
                vision_result = await youtube_system._vision_agent.execute({
                    "action": "analyze",
                    "params": {"image_path": str(frame_file)}
                })
            
            # Use LLM Vision for detailed code extraction
            if llm and hasattr(llm, 'vision_engine') and llm.vision_engine and llm.vision_engine.is_initialized:
                vision_prompt = """Analyze this video frame in detail. 

If you see code on the screen, extract it EXACTLY as shown, including:
- All import statements
- Variable names
- Function definitions
- Code structure
- Comments

If you see text/instructions, extract those too.
Describe what's happening in this frame."""
                
                try:
                    vision_analysis = await llm.vision_engine.analyze_image(
                        str(frame_file),
                        vision_prompt
                    )
                except Exception as e:
                    vision_analysis = f"Vision analysis failed: {e}"
                
                frame_analyses.append({
                    "frame": i,
                    "file": frame_file.name,
                    "analysis": vision_analysis
                })
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Frame {i} analysis failed: {e}[/yellow]")
    
    # Generate comprehensive analysis
    console.print("\n[cyan]🧠 Generating comprehensive analysis...[/cyan]\n")
    
    if llm:
        analysis_prompt = f"""Analyze this YouTube video transcript and provide a detailed breakdown:

TRANSCRIPT:
{transcript[:4000]}

Please provide:
1. **Video Summary**: What is this video about?
2. **Main Topics Covered**: List all major topics
3. **Code Examples**: Extract and explain all code snippets mentioned
4. **Step-by-Step Instructions**: What are the key steps shown?
5. **Tools & Technologies**: What tools, libraries, and platforms are mentioned?
6. **Key Takeaways**: What are the main learning points?

Format your response clearly with sections."""
        
        try:
            analysis_result = await llm.generate(analysis_prompt)
            analysis_text = analysis_result.get("text", "") if isinstance(analysis_result, dict) else str(analysis_result)
        except Exception as e:
            console.print(f"[yellow]⚠️  LLM analysis failed: {e}[/yellow]")
            analysis_text = "LLM analysis unavailable"
    else:
        analysis_text = "LLM not available for analysis"
    
    # Display results
    console.print(Panel.fit(
        "[bold green]📊 Video Analysis Complete[/bold green]",
        border_style="green"
    ))
    
    # Show comprehensive analysis
    console.print("\n[bold cyan]📝 Comprehensive Analysis:[/bold cyan]")
    console.print(Panel(analysis_text, border_style="cyan", title="Video Analysis"))
    
    # Show frame analyses with code
    if frame_analyses:
        console.print("\n[bold cyan]💻 Code Extracted from Frames:[/bold cyan]")
        for frame_info in frame_analyses:
            console.print(f"\n[bold]Frame {frame_info['frame']}:[/bold] {frame_info['file']}")
            analysis = frame_info['analysis']
            
            # Try to extract code from analysis
            if isinstance(analysis, str):
                # Look for code blocks
                import re
                code_blocks = re.findall(r'```(?:python|bash|text)?\n(.*?)```', analysis, re.DOTALL)
                if code_blocks:
                    for code in code_blocks:
                        console.print(Syntax(code.strip(), "python", theme="monokai"))
                else:
                    # Show analysis text
                    console.print(f"[dim]{analysis[:500]}...[/dim]")
            else:
                console.print(f"[dim]{str(analysis)[:500]}...[/dim]")
    
    # Create summary table
    table = Table(title="Video Analysis Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Video ID", video_id)
    table.add_row("Transcript Length", f"{len(transcript):,} characters")
    table.add_row("Frames Analyzed", str(len(frame_files)))
    table.add_row("Code Snippets Found", str(len(code_snippets)))
    
    console.print("\n")
    console.print(table)
    
    # Save detailed analysis
    analysis_file = Path(f"data/youtube/analysis_{video_id}.md")
    with open(analysis_file, 'w') as f:
        f.write(f"# Video Analysis: {video_id}\n\n")
        f.write(f"## Comprehensive Analysis\n\n{analysis_text}\n\n")
        f.write(f"## Frame Analyses\n\n")
        for frame_info in frame_analyses:
            f.write(f"### Frame {frame_info['frame']}\n\n")
            f.write(f"{frame_info['analysis']}\n\n")
    
    console.print(f"\n[green]✅ Detailed analysis saved to: {analysis_file}[/green]")

if __name__ == "__main__":
    video_id = sys.argv[1] if len(sys.argv) > 1 else "LA-hZDnn5Hc"
    asyncio.run(analyze_video_code(video_id))

