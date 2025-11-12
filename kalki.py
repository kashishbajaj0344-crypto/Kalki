#!/usr/bin/env python3
"""
KALKI - Main Entry Point
========================

Single entry point for all KALKI capabilities.

Usage:
    python kalki.py                    # Interactive chat (default)
    python kalki.py --cli              # CLI mode
    python kalki.py --streamlit        # Streamlit app
    python kalki.py --api              # API server
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

async def main():
    parser = argparse.ArgumentParser(
        description="KALKI - Multi-Domain Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python kalki.py                    # Start interactive chat
  python kalki.py --cli chat          # CLI chat mode
  python kalki.py --streamlit         # Launch Streamlit web app
  python kalki.py --api               # Start API server
        """
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use CLI interface (kalki_cli.py)"
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Launch Streamlit web application"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Start API server"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat mode (default)"
    )
    
    args = parser.parse_args()
    
    if args.cli:
        from src.kalki_cli import main as cli_main
        await cli_main()
    elif args.streamlit:
        import subprocess
        import sys
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "apps/kalki_app_enhanced.py"])
        except Exception as e:
            print(f"Error launching Streamlit: {e}")
            print("Make sure Streamlit is installed: pip install streamlit")
    elif args.api:
        try:
            from src.kalki_api_server import main as api_main
            await api_main()
        except Exception as e:
            print(f"Error starting API server: {e}")
    else:
        # Default: Unified chat
        try:
            from apps.kalki_unified_chat import main as chat_main
            await chat_main()
        except Exception as e:
            print(f"Error starting unified chat: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

