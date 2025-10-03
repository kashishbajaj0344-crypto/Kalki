#!/usr/bin/env python3
"""
Automation: scheduler tasks for ingestion and backups.
"""
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from pathlib import Path
from datetime import datetime
import zipfile
import shutil

from .ingest import ingest_resources_file
from .config import ROOT_DIR, VECTOR_DB_DIR

logger = logging.getLogger("kalki.automation")
scheduler = BackgroundScheduler()

def backup_vectordb():
    """
    Creates a timestamped backup of the vector DB folder.
    """
    try:
        dst = Path(ROOT_DIR) / "vector_db_backups"
        dst.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        zip_path = dst / f"vector_db_backup_{ts}.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in VECTOR_DB_DIR.rglob('*'):
                zf.write(f, f.relative_to(VECTOR_DB_DIR))
        logger.info(f"Vector DB backup created at {zip_path}")
    except Exception as e:
        logger.exception(f"backup_vectordb failed: {e}")

def start_scheduler():
    """
    Starts background tasks: periodic ingestion and backups.
    """
    # Ingest resources every 6 hours
    scheduler.add_job(ingest_resources_file, 'interval', hours=6, id='ingest_resources', next_run_time=datetime.now())

    # Backup vector DB daily at 03:00 UTC
    scheduler.add_job(backup_vectordb, 'cron', hour=3, minute=0, id='backup_vectordb', next_run_time=datetime.now())

    scheduler.start()
    logger.info("Background scheduler started.")

# ------------------ TEST ------------------
if __name__ == "__main__":
    start_scheduler()
    logger.info("Scheduler test: waiting 10 seconds to see jobs run...")
    import time
    time.sleep(10)
    scheduler.shutdown()
    logger.info("Scheduler test completed.")
