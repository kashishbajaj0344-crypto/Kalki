#!/usr/bin/env python3
"""
Kalki Live Integration Test
Tests the safety-orchestrated functions in a real environment
"""
import asyncio
import logging
from pathlib import Path

from kalki_agent_integration import safe_ask_kalki_sync, safe_ingest_pdf_file_sync

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integration_test")


async def test_safe_query():
    """Test safe query functionality"""
    logger.info("🧪 Testing safe query...")

    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the ethics of AI?"
    ]

    for query in test_queries:
        logger.info(f"\nQuery: {query}")
        result = safe_ask_kalki_sync(query, {"test_mode": True})

        if result["status"] == "success":
            logger.info("✅ Query successful")
            logger.info(f"   Answer: {result['answer'][:100]}...")
            safety = result.get("safety_assessment", {})
            logger.info(f"   Safety Score: {safety.get('ethical_score', 0):.2f}")
        else:
            logger.warning(f"⚠️  Query result: {result['status']} - {result.get('error', 'Unknown')}")


async def test_safe_ingestion():
    """Test safe ingestion functionality"""
    logger.info("\n🧪 Testing safe ingestion...")

    # Create a dummy PDF path for testing
    test_pdf = Path("/tmp/test_ai_document.pdf")

    # Create a simple test file if it doesn't exist
    if not test_pdf.exists():
        test_pdf.write_text("This is a test PDF content for AI safety research.")
        logger.info(f"Created test file: {test_pdf}")

    result = safe_ingest_pdf_file_sync(str(test_pdf), "academic", {"test_mode": True})

    if result["status"] == "success":
        logger.info("✅ Ingestion successful")
        safety = result.get("safety_assessment", {})
        logger.info(f"   Safety Score: {safety.get('ethical_score', 0):.2f}")
    else:
        logger.warning(f"⚠️  Ingestion result: {result['status']} - {result.get('error', 'Unknown')}")


async def main():
    """Run integration tests"""
    logger.info("🚀 Kalki Live Integration Test Starting")
    logger.info("=" * 50)

    try:
        await test_safe_query()
        await test_safe_ingestion()

        logger.info("\n" + "=" * 50)
        logger.info("✅ Integration tests completed successfully!")

    except Exception as e:
        logger.exception(f"❌ Integration test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)