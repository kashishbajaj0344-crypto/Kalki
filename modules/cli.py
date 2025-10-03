#!/usr/bin/env python3
"""
Kalki CLI interface
Allows querying Kalki and optionally ingesting PDFs from the command line
"""
import sys
from pathlib import Path
import logging

from .llm import ask_kalki
from .ingest import ingest_pdf_file

logger = logging.getLogger("kalki.cli")

def cli_main():
    print("Kalki CLI v3.0")
    print("Commands:")
    print("  ask <your query>  - Query Kalki")
    print("  ingest <pdf_path> - Ingest a PDF file")
    print("  exit              - Exit CLI")

    while True:
        try:
            inp = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI.")
            break
        if not inp:
            continue
        if inp.lower() == "exit":
            print("Goodbye!")
            break
        parts = inp.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if cmd == "ask":
            if not arg:
                print("Please provide a query.")
                continue
            try:
                answer = ask_kalki(arg)
                print(f"Kalki says:\n{answer}\n")
            except Exception as e:
                logger.exception(f"Query failed: {e}")
                print(f"Error: {e}")
        elif cmd == "ingest":
            if not arg:
                print("Please provide a PDF path to ingest.")
                continue
            pdf_path = Path(arg)
            if not pdf_path.exists():
                print(f"File does not exist: {pdf_path}")
                continue
            try:
                ingest_pdf_file(pdf_path)
                print(f"Ingested PDF: {pdf_path}")
            except Exception as e:
                logger.exception(f"Ingest failed: {e}")
                print(f"Failed to ingest: {e}")
        else:
            print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    cli_main()
