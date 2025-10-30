"""
Live Progress Ingestion Test for Kalki v2.3
Shows per-file ingestion status, chunks, and tags.
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.table import Table
from modules.agents.ingestion_agent import IngestionAgent

console = Console()

async def main():
    pdf_root = Path(__file__).resolve().parent.parent / "pdfs"
    agent = IngestionAgent(vector_db_path=Path(__file__).resolve().parent.parent / "vector_db/chroma.sqlite3")

    files = list(pdf_root.rglob("*.pdf"))
    table = Table(title="Kalki PDF Ingestion Progress")
    table.add_column("File", style="cyan", overflow="fold")
    table.add_column("Status", style="green")
    table.add_column("Chunks", justify="right")
    table.add_column("Tags", style="magenta", overflow="fold")

    for f in files:
        result = await agent.ingest_file(f)
        chunks = result.get("chunks", 0)
        details = result.get("details", [])
        tags_summary = ", ".join({t for d in details for t in d.get("metadata", {}).get("tags", [])})
        status = result.get("status", "error")
        table.add_row(str(f.relative_to(pdf_root)), status, str(chunks), tags_summary)
        console.clear()
        console.print(table)

if __name__ == "__main__":
    asyncio.run(main())
