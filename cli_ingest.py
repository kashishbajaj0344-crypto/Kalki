"""
cli_ingest.py
KALKI v2.3 — Command-line interface for batch document ingestion and embedding.
Usage:
    python cli_ingest.py /path/to/folder1 /path/to/folder2
"""

import sys
from pathlib import Path
from modules.ingest import Ingestor
from modules.logger import get_logger
from modules.utils import safe_execution

logger = get_logger("cli_ingest")

@safe_execution(default=[])
def collect_files(paths):
    """
    Recursively collect PDF and TXT files from given paths.
    Returns a list of file paths.
    """
    files = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            logger.warning("Path does not exist: %s", path)
            continue
        if path.is_file() and path.suffix.lower() in [".pdf", ".txt"]:
            files.append(path)
        elif path.is_dir():
            files.extend([f for f in path.rglob("*") if f.suffix.lower() in [".pdf", ".txt"]])
    return files


def main():
    if len(sys.argv) < 2:
        logger.info("Usage: python cli_ingest.py /path/to/folder_or_file ...")
        return

    paths = sys.argv[1:]
    files_to_ingest = collect_files(paths)

    if not files_to_ingest:
        logger.warning("No PDF or TXT files found in the specified paths.")
        return

    logger.info("Found %d files to ingest.", len(files_to_ingest))

    ingestor = Ingestor()

    for file_path in files_to_ingest:
        result = ingestor.add_file(file_path, async_mode=False)
        if result:
            logger.info("Successfully ingested: %s", file_path)
        else:
            logger.warning("Failed to ingest: %s", file_path)

    logger.info("Ingestion complete.")


if __name__ == "__main__":
    main()
