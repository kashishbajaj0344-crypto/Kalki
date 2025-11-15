#!/usr/bin/env python3
"""
KALKI UNIFIED CHATBOT - Centralized Interface for All Domains
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Single entry point for ALL Kalki capabilities:
- Auto-detects domain from user queries
- Routes to appropriate handler (domain-specific or general)
- Supports all domains: Construction, Game Dev, Robotics, Aerospace, Power Systems, etc.
- Falls back gracefully for general queries

Usage:
    python kalki_unified_chat.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.syntax import Syntax

# Add project root to path
project_root = Path(__file__).parent.parent  # Go up from apps/ to root
sys.path.insert(0, str(project_root))

from modules.domains.domain_registry import DomainRegistry
from modules.supreme_control_hub import SupremeControlHub
from src.kalki_complete import KalkiOrchestrator  # Use the complete orchestrator with process_user_query
from modules.utils.logging_config import setup_logging, get_logger

console = Console()
logger = get_logger("Kalki.UnifiedChat")


class UnifiedKalkiChat:
    """
    Centralized chatbot that handles ALL Kalki capabilities across all domains.
    
    Features:
    - Automatic domain detection from queries
    - Intelligent routing to domain handlers or general orchestrator
    - Multi-domain query support
    - Chat history and context management
    - Beautiful CLI interface
    """
    
    def __init__(self):
        """Initialize the unified chatbot"""
        console.print("[cyan]Initializing Kalki Chatbot...[/cyan]")
        
        # Core systems
        self.domain_registry = DomainRegistry()
        self.supreme_hub = None  # Lazy initialization
        self.orchestrator = None  # Lazy initialization
        
        # Chat state
        self.chat_history: List[Dict[str, Any]] = []
        self.current_project_id: Optional[str] = None
        self.current_domain: Optional[str] = None
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "domain_queries": 0,
            "general_queries": 0,
            "domains_used": set()
        }
        
        console.print("[green]✅ Unified Chatbot ready![/green]")
    
    async def _initialize_systems(self):
        """Lazy initialization of heavy systems"""
        if self.supreme_hub is None:
            console.print("[cyan]Loading Supreme Control Hub...[/cyan]")
            self.supreme_hub = SupremeControlHub()
            console.print("[green]✅ Supreme Control Hub ready[/green]")
        
        if self.orchestrator is None:
            console.print("[cyan]Loading Kalki Orchestrator...[/cyan]")
            self.orchestrator = KalkiOrchestrator()
            # Initialize the system
            success = await self.orchestrator.initialize_system()
            if success:
                console.print("[green]✅ Orchestrator ready[/green]")
            else:
                console.print("[yellow]⚠️  Orchestrator initialization had issues[/yellow]")
    
    async def process_message(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user message with intelligent domain routing.
        
        Args:
            user_input: User's natural language query
            
        Returns:
            Dict with response, domain info, and metadata
        """
        await self._initialize_systems()
        
        self.stats["total_queries"] += 1
        start_time = datetime.now()
        
        # Infer domain(s) from query
        inferred_domains = await self.domain_registry.infer_domain(user_input)
        
        # Route based on domain detection
        if inferred_domains:
            # Domain-specific query
            self.stats["domain_queries"] += 1
            domain_name = inferred_domains[0]
            self.stats["domains_used"].add(domain_name)
            self.current_domain = domain_name
            
            console.print(f"[dim]🔍 Detected domain: {domain_name}[/dim]")
            
            # Get chat context with session management
            chat_context = self._get_chat_context()
            
            # Check if we have an active game dev session
            if domain_name == "game_development":
                # Look for session_id in recent chat history
                for msg in reversed(self.chat_history[-5:]):  # Check last 5 messages
                    if msg.get("session_id"):
                        chat_context["session_id"] = msg["session_id"]
                        break
            
            # Use Supreme Control Hub for domain-aware processing
            # Supreme Hub now automatically uses copilots when available
            result = await self.supreme_hub.process_domain_aware_query(
                query=user_input,
                context=chat_context,
                project_id=self.current_project_id
            )
            
            # Check if copilot was used (for display)
            if result.get("domain", {}).get("copilot_used"):
                console.print("[dim]✨ Enhanced processing with copilot[/dim]")
            
            # Handle game dev copilot workflow
            if domain_name == "game_development":
                # Store session for follow-up questions
                if result.get("session_id"):
                    # Store in metadata for next query
                    if "metadata" not in result:
                        result["metadata"] = {}
                    result["metadata"]["session_id"] = result.get("session_id")
                    result["metadata"]["project_id"] = result.get("project_id")
                
                # If project created, trigger complete build workflow
                if result.get("project_id") and result.get("status") == "project_created":
                    console.print("[cyan]🚀 Building complete game...[/cyan]")
                    # Auto-trigger complete build
                    copilot = self.domain_registry.get_copilot("game_development")
                    if copilot:
                        try:
                            build_result = await copilot.build_complete_game(
                                result.get("session_id"),
                                auto_deploy=True,
                                auto_polish=True,
                                polish_level="standard"
                            )
                            if build_result.get("status") == "completed":
                                result["answer"] += f"\n\n✨ {build_result.get('message', 'Game built successfully!')}"
                                # Add build details
                                for step in build_result.get("steps", []):
                                    step_result = step.get("result", {})
                                    if step_result.get("status") == "success":
                                        result["answer"] += f"\n  ✅ {step.get('step')}: {step_result.get('message', '')}"
                        except Exception as e:
                            logger.exception(f"Complete build workflow failed: {e}")
                            result["answer"] += f"\n\n⚠️  Build workflow encountered an issue: {e}"
            
            # Format response
            if result.get("success"):
                response = {
                    "response": result.get("answer", "Query processed successfully"),
                    "domain": result.get("domain", {}).get("name", domain_name),
                    "confidence": result.get("confidence", 0.8),
                    "metadata": {
                        "domains_detected": inferred_domains,
                        "domain_info": result.get("domain", {}),
                        "project_context": result.get("project_context"),
                        "processing_time": (datetime.now() - start_time).total_seconds()
                    }
                }
            else:
                # Fallback to orchestrator if domain processing fails
                console.print(f"[yellow]⚠️  Domain processing failed, using general orchestrator[/yellow]")
                return await self._process_general_query(user_input, start_time)
        
        else:
            # General query - no specific domain detected
            self.stats["general_queries"] += 1
            self.current_domain = None
            console.print("[dim]💬 Using general Kalki intelligence[/dim]")
            return await self._process_general_query(user_input, start_time)
        
        # Add to chat history
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "response": response,
            "domain": response.get("domain")
        })
        
        return response
    
    async def _process_general_query(self, user_input: str, start_time: datetime) -> Dict[str, Any]:
        """Process query using general Kalki orchestrator"""
        try:
            result = await self.orchestrator.process_user_query(
                user_input,
                context=self._get_chat_context()
            )
            
            # Extract response from various possible result formats
            response_text = (
                result.get("response") or 
                result.get("enhanced_reasoning") or 
                result.get("answer") or 
                result.get("result", {}).get("synthesis", "Query processed")
            )
            
            response = {
                "response": response_text,
                "domain": None,
                "confidence": result.get("confidence", result.get("quality_score", 0.7)),
                "metadata": {
                    "domains_detected": [],
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "status": result.get("status", "completed"),
                    "orchestrator_result": result
                }
            }
            
            # Add to chat history
            self.chat_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "response": response,
                "domain": None
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing general query: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "domain": None,
                "confidence": 0.0,
                "error": str(e),
                "metadata": {
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            }
    
    def _get_chat_context(self) -> Dict[str, Any]:
        """Get recent chat history as context"""
        if not self.chat_history:
            return {}
        
        # Return last 5 exchanges for context
        recent = self.chat_history[-5:]
        return {
            "recent_exchanges": [
                {
                    "user": exchange.get("user"),
                    "response": exchange.get("response", {}).get("response", "")[:200]  # Truncate
                }
                for exchange in recent
            ],
            "current_domain": self.current_domain,
            "project_id": self.current_project_id
        }
    
    def show_domains(self):
        """Display available domains"""
        domains = self.domain_registry.list_domains()
        
        if not domains:
            console.print("[yellow]No domains available[/yellow]")
            return
        
        table = Table(title="Available Kalki Domains", show_header=True, header_style="bold cyan")
        table.add_column("Domain", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Knowledge Items", justify="right")
        
        for domain_name in domains:
            info = self.domain_registry.get_domain_info(domain_name)
            if info:
                status = "✅ Loaded" if info.get("is_loaded") else "⚠️  Not Loaded"
                knowledge = info.get("knowledge_total", 0)
                table.add_row(domain_name, status, str(knowledge))
        
        console.print(table)
    
    def show_stats(self):
        """Display chat statistics"""
        table = Table(title="Chat Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Total Queries", str(self.stats["total_queries"]))
        table.add_row("Domain Queries", str(self.stats["domain_queries"]))
        table.add_row("General Queries", str(self.stats["general_queries"]))
        table.add_row("Domains Used", ", ".join(self.stats["domains_used"]) or "None")
        table.add_row("Chat History Length", str(len(self.chat_history)))
        
        console.print(table)
    
    def show_help(self):
        """Display help information"""
        help_text = """
[bold cyan]Kalki Chatbot - Commands[/bold cyan]

[bold]Chat Commands:[/bold]
  • Just type your question - Kalki will auto-detect the domain!
  • Examples:
    - "What size joists for a 16 foot span?" (Construction)
    - "How do I create a Unity character controller?" (Game Dev)
    - "Design a PID controller for a robot arm" (Robotics)
    - "Calculate battery capacity for a drone" (Power Systems)

[bold]Special Commands:[/bold]
  • /domains - Show available domains
  • /stats - Show chat statistics
  • /history - Show recent chat history
  • /clear - Clear chat history
  • /project <id> - Set current project ID
  • /help - Show this help
  • /exit - Exit chat

[bold]YouTube Ingestion:[/bold]
  • youtube ingest <URL> - Ingest YouTube video and learn from it
  • yt ingest <URL> - Short form
  Example: youtube ingest https://www.youtube.com/watch?v=VIDEO_ID

[bold]Features:[/bold]
  • Automatic domain detection
  • Multi-domain support
  • Context-aware responses
  • Project-aware queries
        """
        console.print(Panel(help_text, border_style="cyan"))
    
    def show_history(self, limit: int = 10):
        """Show recent chat history"""
        if not self.chat_history:
            console.print("[yellow]No chat history[/yellow]")
            return
        
        recent = self.chat_history[-limit:]
        
        for i, exchange in enumerate(recent, 1):
            domain = exchange.get("domain")
            domain_tag = f"[dim][{domain}][/dim] " if domain else ""
            
            console.print(f"\n[bold cyan]Exchange {i}:[/bold cyan]")
            console.print(f"[green]You:[/green] {exchange.get('user')}")
            
            response = exchange.get("response", {})
            response_text = response.get("response", "")
            
            # Truncate long responses
            if len(response_text) > 500:
                response_text = response_text[:500] + "..."
            
            console.print(f"{domain_tag}[cyan]Kalki:[/cyan] {response_text}")
    
    async def _handle_youtube_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Handle YouTube ingestion commands"""
        if command.startswith("youtube ingest ") or command.startswith("yt ingest "):
            url = command.replace("youtube ingest ", "").replace("yt ingest ", "").strip()
            if not url:
                return {"error": "Please provide a YouTube URL", "status": "error"}
            
            try:
                from modules.youtube_ingestion import YouTubeIngestionSystem
                youtube_system = YouTubeIngestionSystem()
                
                console.print(f"[cyan]🎥 Ingesting YouTube video...[/cyan]")
                result = await youtube_system.ingest_youtube_video(url, extract_knowledge=True)
                
                if result.get("status") == "success":
                    metadata = result["result"]["metadata"]
                    console.print(f"[green]✅ Video ingested: {metadata['title']}[/green]")
                    console.print(f"[dim]📝 Transcript: {len(result['result']['transcript'])} characters[/dim]")
                    console.print(f"[dim]🎬 Key frames: {len(result['result']['key_frames'])}[/dim]")
                    if result["result"]["domain"]:
                        console.print(f"[dim]🏷️  Domain: {result['result']['domain']}[/dim]")
                    
                    return {
                        "answer": f"✅ Successfully ingested YouTube video: **{metadata['title']}**\n\n"
                                f"📝 Transcript: {len(result['result']['transcript'])} characters\n"
                                f"🎬 Key frames: {len(result['result']['key_frames'])}\n"
                                f"🏷️  Domain: {result['result']['domain'] or 'Not detected'}\n\n"
                                f"You can now query this video's content!",
                        "status": "success"
                    }
                else:
                    return {"error": result.get("error", "Ingestion failed"), "status": "error"}
            except Exception as e:
                logger.exception(f"YouTube ingestion failed: {e}")
                return {"error": str(e), "status": "error"}
        
        return None
    
    async def start(self):
        """Start the interactive chat session"""
        # Welcome screen
        console.clear()
        welcome = Panel.fit(
            "[bold cyan]🤖 KALKI Chatbot[/bold cyan]\n\n"
            "[white]Your AI assistant for ALL domains[/white]\n"
            "[dim]Construction • Game Dev • Robotics • Aerospace • Power Systems • More...[/dim]\n\n"
            "[yellow]Type your question or /help for commands[/yellow]",
            border_style="cyan"
        )
        console.print(welcome)
        
        # Show available domains
        domains = self.domain_registry.list_domains()
        if domains:
            console.print(f"\n[dim]Available domains: {', '.join(domains)}[/dim]\n")
        
        # Main chat loop
        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("\n[cyan]👋 Goodbye![/cyan]")
                    break
                
                if user_input.lower() == '/help':
                    self.show_help()
                    continue
                
                if user_input.lower() == '/domains':
                    self.show_domains()
                    continue
                
                if user_input.lower() == '/stats':
                    self.show_stats()
                    continue
                
                if user_input.lower() == '/history':
                    self.show_history()
                    continue
                
                if user_input.lower() == '/clear':
                    self.chat_history.clear()
                    console.print("[green]✅ Chat history cleared[/green]")
                    continue
                
                # Handle YouTube ingestion commands
                youtube_result = await self._handle_youtube_command(user_input)
                if youtube_result:
                    if youtube_result.get("status") == "success":
                        console.print(f"[green]{youtube_result.get('answer', 'Video ingested!')}[/green]")
                    else:
                        console.print(f"[red]❌ {youtube_result.get('error', 'Ingestion failed')}[/red]")
                    continue
                
                if user_input.startswith('/project '):
                    self.current_project_id = user_input.split(' ', 1)[1].strip()
                    console.print(f"[green]✅ Project ID set to: {self.current_project_id}[/green]")
                    continue
                
                # Process message
                console.print()  # Blank line for spacing
                result = await self.process_message(user_input)
                
                # Display response
                response_text = result.get("response", "No response")
                domain = result.get("domain")
                confidence = result.get("confidence", 0.0)
                
                # Format response with domain tag
                if domain:
                    domain_tag = f"[dim][{domain}][/dim] "
                else:
                    domain_tag = "[dim][general][/dim] "
                
                # Show response
                console.print(f"{domain_tag}[cyan]Kalki:[/cyan]")
                
                # Try to format as markdown if it looks like markdown
                if "```" in response_text or "#" in response_text[:50]:
                    console.print(Markdown(response_text))
                else:
                    console.print(response_text)
                
                # Show confidence if low
                if confidence < 0.6:
                    console.print(f"[yellow]⚠️  Low confidence ({confidence:.2f})[/yellow]")
                
            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                continue
            except Exception as e:
                logger.exception("Error in chat loop")
                console.print(f"\n[red]Error: {e}[/red]")
                continue


async def main():
    """Main entry point"""
    # Set up logging
    setup_logging(log_level="INFO")
    
    # Create and start chatbot
    chat = UnifiedKalkiChat()
    await chat.start()


if __name__ == "__main__":
    asyncio.run(main())

