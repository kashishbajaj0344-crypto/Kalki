#!/usr/bin/env python3
"""
Kalki Internet Connectivity Setup Script
========================================

This script helps you set up Kalki's internet connectivity features
including web search, API integration, and real-time data retrieval.

Requirements:
- Python 3.8+
- pip
- API keys for search providers (optional but recommended)

Usage:
    python setup_internet.py
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🌐 Kalki Internet Connectivity Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")

    required_packages = [
        "beautifulsoup4==4.12.2",
        "aiohttp==3.9.1",
        "python-dotenv==1.1.1"
    ]

    optional_packages = [
        "lxml==4.9.3"  # Optional for better HTML parsing
    ]

    # Install required packages
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--user"
        ] + required_packages)
        print("✅ Required dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install required dependencies: {e}")
        return False

    # Try to install optional packages
    for package in optional_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--user", package
            ])
            print(f"✅ Optional package {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Optional package {package} failed to install - continuing without it")

    return True

def setup_api_keys():
    """Guide user through API key setup"""
    print("\n🔑 API Key Configuration")
    print("-" * 30)
    print("Kalki can use various search providers. API keys are optional but recommended.")
    print("Without API keys, Kalki will fall back to DuckDuckGo (free, no keys needed).")
    print()

    env_file = Path(".env")
    env_content = []

    # Check if .env already exists
    if env_file.exists():
        print("📄 Found existing .env file")
        with open(env_file, 'r') as f:
            existing_content = f.read()
        env_content.append(existing_content.strip())
        env_content.append("")  # Add blank line

    # Google Custom Search API
    print("1. Google Custom Search API (recommended)")
    print("   - Get API key: https://console.developers.google.com/")
    print("   - Create Custom Search Engine: https://cse.google.com/")
    google_key = input("   Enter Google Search API Key (or press Enter to skip): ").strip()
    google_cse = input("   Enter Google CSE ID (or press Enter to skip): ").strip()

    if google_key and google_cse:
        env_content.append(f"GOOGLE_SEARCH_API_KEY={google_key}")
        env_content.append(f"GOOGLE_CSE_ID={google_cse}")
        print("   ✅ Google Custom Search configured")
    else:
        print("   ⏭️  Google Custom Search skipped")

    print()

    # Bing Search API
    print("2. Bing Search API")
    print("   - Get API key: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api")
    bing_key = input("   Enter Bing Search API Key (or press Enter to skip): ").strip()

    if bing_key:
        env_content.append(f"BING_SEARCH_API_KEY={bing_key}")
        print("   ✅ Bing Search configured")
    else:
        print("   ⏭️  Bing Search skipped")

    print()

    # SerpApi
    print("3. SerpApi (Alternative Google Search)")
    print("   - Get API key: https://serpapi.com/")
    serpapi_key = input("   Enter SerpApi Key (or press Enter to skip): ").strip()

    if serpapi_key:
        env_content.append(f"SERPAPI_KEY={serpapi_key}")
        print("   ✅ SerpApi configured")
    else:
        print("   ⏭️  SerpApi skipped")

    print()

    # OpenAI API (for enhanced LLM features)
    print("4. OpenAI API (for enhanced LLM capabilities)")
    print("   - Get API key: https://platform.openai.com/api-keys")
    openai_key = input("   Enter OpenAI API Key (or press Enter to skip): ").strip()

    if openai_key:
        env_content.append(f"OPENAI_API_KEY={openai_key}")
        print("   ✅ OpenAI API configured")
    else:
        print("   ⏭️  OpenAI API skipped")

    print()

    # HuggingFace API
    print("5. HuggingFace API (for Llama models)")
    print("   - Get API key: https://huggingface.co/settings/tokens")
    hf_key = input("   Enter HuggingFace API Key (or press Enter to skip): ").strip()

    if hf_key:
        env_content.append(f"HUGGINGFACE_API_KEY={hf_key}")
        print("   ✅ HuggingFace API configured")
    else:
        print("   ⏭️  HuggingFace API skipped")

    # Write .env file
    if len(env_content) > 0:
        with open(env_file, 'w') as f:
            f.write('\n'.join(env_content))
        print(f"\n✅ API keys saved to {env_file}")
    else:
        print("\n⚠️  No API keys configured - Kalki will use DuckDuckGo for searches")

def test_connectivity():
    """Test internet connectivity and API keys"""
    print("\n🧪 Testing Connectivity")
    print("-" * 25)

    try:
        # Test basic internet connectivity
        import requests
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code == 200:
            print("✅ Internet connectivity: OK")
        else:
            print("❌ Internet connectivity: FAILED")
            return False

    except ImportError:
        print("❌ requests library not available")
        return False
    except Exception as e:
        print(f"❌ Internet connectivity test failed: {e}")
        return False

    # Test API keys if available
    from dotenv import load_dotenv
    load_dotenv()

    api_tests = [
        ("Google Search", "GOOGLE_SEARCH_API_KEY", "GOOGLE_CSE_ID"),
        ("Bing Search", "BING_SEARCH_API_KEY", None),
        ("SerpApi", "SERPAPI_KEY", None),
        ("OpenAI", "OPENAI_API_KEY", None),
        ("HuggingFace", "HUGGINGFACE_API_KEY", None)
    ]

    for name, key_env, secondary_env in api_tests:
        key = os.getenv(key_env)
        if key:
            if secondary_env:
                secondary = os.getenv(secondary_env)
                if secondary:
                    print(f"✅ {name}: API key configured")
                else:
                    print(f"⚠️  {name}: Primary key set, missing secondary key")
            else:
                print(f"✅ {name}: API key configured")
        else:
            print(f"⏭️  {name}: Not configured")

    return True

def show_usage_examples():
    """Show usage examples"""
    print("\n🚀 Usage Examples")
    print("-" * 20)
    print("Once setup is complete, you can use Kalki with internet connectivity:")
    print()
    print("# Web search")
    print('kalki web search "Call of Duty game mechanics" --results 5')
    print()
    print("# Research a topic")
    print('kalki web research "artificial intelligence trends" --depth comprehensive')
    print()
    print("# Fetch content from a URL")
    print('kalki web fetch "https://en.wikipedia.org/wiki/Call_of_Duty"')
    print()
    print("# Regular queries now use web search when appropriate")
    print('kalki query "What are the latest Call of Duty games?"')
    print()
    print("# Check system status")
    print('kalki status')

def main():
    """Main setup function"""
    print_header()

    # Run setup steps
    check_python_version()

    if not install_dependencies():
        sys.exit(1)

    setup_api_keys()

    if not test_connectivity():
        print("❌ Setup incomplete - connectivity issues detected")
        sys.exit(1)

    show_usage_examples()

    print("\n🎉 Setup Complete!")
    print("Kalki now has internet connectivity capabilities.")
    print("Run 'kalki status' to verify the WebSearchAgent is loaded.")

if __name__ == "__main__":
    main()