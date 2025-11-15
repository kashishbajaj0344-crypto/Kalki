#!/usr/bin/env python3
"""
Extract code from video frames using vision analysis
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent))

console = Console()

async def extract_code_from_frames(video_id: str = "LA-hZDnn5Hc"):
    """Extract code snippets from video frames"""
    
    console.print(Panel.fit(
        "[bold cyan]💻 Extracting Code from Video Frames[/bold cyan]\n\n"
        f"[yellow]Video ID:[/yellow] {video_id}",
        border_style="cyan"
    ))
    
    frames_dir = Path(f"data/youtube/frames/{video_id}")
    if not frames_dir.exists():
        console.print(f"[red]❌ Frames directory not found[/red]")
        return
    
    frame_files = sorted(frames_dir.glob("*.jpg"))
    console.print(f"\n[cyan]📸 Found {len(frame_files)} frames[/cyan]\n")
    
    # Try to use vision engine directly
    try:
        from modules.llm import LlamaVisionEngine
        from config.models_config import get_model_path
        
        vision_model_path = get_model_path("llama-3.2-11b-vision-instruct")
        if vision_model_path and Path(vision_model_path).exists():
            vision_engine = LlamaVisionEngine(model_path=vision_model_path)
            await vision_engine.initialize()
            
            console.print("[green]✅ Vision engine initialized[/green]\n")
            
            code_extractions = []
            
            for i, frame_file in enumerate(frame_files, 1):
                console.print(f"[dim]Analyzing frame {i}/{len(frame_files)}: {frame_file.name}[/dim]")
                
                prompt = """Extract ALL code visible in this image. 

If you see:
- Python code: Extract it EXACTLY as shown with proper indentation
- Import statements: Include all imports
- Variable assignments: Include variable names and values
- Function definitions: Include complete function code
- Command-line commands: Extract bash/terminal commands
- Configuration: Extract any config code
- Comments: Include all comments

Format the code in proper code blocks. If no code is visible, describe what you see."""
                
                try:
                    analysis = await vision_engine.analyze_image(str(frame_file), prompt)
                    code_extractions.append({
                        "frame": i,
                        "file": frame_file.name,
                        "code": analysis
                    })
                    console.print(f"  [green]✅ Frame {i} analyzed[/green]")
                except Exception as e:
                    console.print(f"  [yellow]⚠️  Frame {i} failed: {e}[/yellow]")
            
            # Display results
            console.print("\n[bold green]📋 Code Extractions:[/bold green]\n")
            
            for extraction in code_extractions:
                console.print(Panel(
                    f"[bold]Frame {extraction['frame']}:[/bold] {extraction['file']}\n\n"
                    f"{extraction['code'][:1000]}...",
                    title=f"Frame {extraction['frame']}",
                    border_style="cyan"
                ))
            
            # Save to file
            output_file = Path(f"data/youtube/code_extractions_{video_id}.md")
            with open(output_file, 'w') as f:
                f.write(f"# Code Extractions from Video: {video_id}\n\n")
                for extraction in code_extractions:
                    f.write(f"## Frame {extraction['frame']}\n\n")
                    f.write(f"**File:** {extraction['file']}\n\n")
                    f.write(f"```\n{extraction['code']}\n```\n\n")
            
            console.print(f"\n[green]✅ Code extractions saved to: {output_file}[/green]")
            
        else:
            console.print("[yellow]⚠️  Vision model not found, using transcript analysis[/yellow]")
            
    except Exception as e:
        console.print(f"[yellow]⚠️  Vision analysis unavailable: {e}[/yellow]")
        console.print("[dim]Using transcript-based code extraction instead...[/dim]")

if __name__ == "__main__":
    video_id = sys.argv[1] if len(sys.argv) > 1 else "LA-hZDnn5Hc"
    asyncio.run(extract_code_from_frames(video_id))

