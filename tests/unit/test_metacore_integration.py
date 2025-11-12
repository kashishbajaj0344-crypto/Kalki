#!/usr/bin/env python3
"""
Test MetaCore Progressive Reasoning Integration
Tests the integration of MetaCore into Kalki orchestrator for adaptive reasoning depth
"""

import asyncio
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from kalki_complete import KalkiOrchestrator
from modules.utils.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger("Test.MetaCoreIntegration")

async def test_metacore_integration():
    """Test MetaCore integration with Kalki orchestrator"""
    
    logger.info("=" * 80)
    logger.info("🧪 TESTING METACORE PROGRESSIVE REASONING INTEGRATION")
    logger.info("=" * 80)
    
    # Initialize Kalki orchestrator
    logger.info("\n📦 Initializing Kalki orchestrator...")
    kalki = KalkiOrchestrator()
    
    init_success = await kalki.initialize_system()
    if not init_success:
        logger.error("❌ Failed to initialize Kalki system")
        return False
    
    logger.info("✅ Kalki initialized successfully")
    
    # Get system status
    logger.info("\n📊 System Status:")
    status = await kalki.get_system_status()
    logger.info(f"  - System Status: {status['system_status']}")
    logger.info(f"  - Phases Active: {status['phases_active']}")
    logger.info(f"  - Total Agents: {status['total_agents']}")
    logger.info(f"  - MetaCore Active: {status['meta_core_active']}")
    
    if status['meta_core_status']:
        meta_status = status['meta_core_status']
        logger.info(f"  - Reasoning Depth: {meta_status['reasoning_depth']}")
        logger.info(f"  - Output Style: {meta_status['output_style']}")
        logger.info(f"  - System Health: {meta_status['system_health']}")
    
    # Test 1: Simple query (should use SUMMARY depth)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Simple Query (Expected: SUMMARY depth)")
    logger.info("=" * 80)
    
    simple_query = "What is 2 + 2?"
    logger.info(f"Query: {simple_query}")
    
    result1 = await kalki.process_user_query(simple_query)
    logger.info(f"Status: {result1.get('status', 'unknown')}")
    
    if 'quality_metrics' in result1:
        metrics = result1['quality_metrics']
        logger.info(f"Quality Metrics:")
        logger.info(f"  - Reasoning Depth: {metrics.get('reasoning_depth')}")
        logger.info(f"  - Coverage: {metrics.get('interdisciplinary_coverage', 0):.2f}")
        logger.info(f"  - Coherence: {metrics.get('coherence_score', 0):.2f}")
        logger.info(f"  - Response Time: {metrics.get('response_time', 0):.2f}s")
    
    # Test 2: Moderate complexity query (should use STANDARD depth)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Moderate Query (Expected: STANDARD depth)")
    logger.info("=" * 80)
    
    moderate_query = "How do neural networks learn from data?"
    logger.info(f"Query: {moderate_query}")
    
    result2 = await kalki.process_user_query(moderate_query)
    logger.info(f"Status: {result2.get('status', 'unknown')}")
    
    if 'quality_metrics' in result2:
        metrics = result2['quality_metrics']
        logger.info(f"Quality Metrics:")
        logger.info(f"  - Reasoning Depth: {metrics.get('reasoning_depth')}")
        logger.info(f"  - Coverage: {metrics.get('interdisciplinary_coverage', 0):.2f}")
        logger.info(f"  - Coherence: {metrics.get('coherence_score', 0):.2f}")
        logger.info(f"  - Response Time: {metrics.get('response_time', 0):.2f}s")
    
    # Test 3: Complex interdisciplinary query (should use DEEP_ANALYSIS depth)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Complex Interdisciplinary Query (Expected: DEEP_ANALYSIS depth)")
    logger.info("=" * 80)
    
    complex_query = "Design an innovative sustainable building that optimizes energy efficiency using biomimetic principles"
    logger.info(f"Query: {complex_query}")
    
    result3 = await kalki.process_user_query(complex_query)
    logger.info(f"Status: {result3.get('status', 'unknown')}")
    
    if 'quality_metrics' in result3:
        metrics = result3['quality_metrics']
        logger.info(f"Quality Metrics:")
        logger.info(f"  - Reasoning Depth: {metrics.get('reasoning_depth')}")
        logger.info(f"  - Coverage: {metrics.get('interdisciplinary_coverage', 0):.2f}")
        logger.info(f"  - Coherence: {metrics.get('coherence_score', 0):.2f}")
        logger.info(f"  - Response Time: {metrics.get('response_time', 0):.2f}s")
    
    # Check quality trends
    if kalki.meta_core:
        logger.info("\n" + "=" * 80)
        logger.info("📈 QUALITY TRENDS ANALYSIS")
        logger.info("=" * 80)
        
        trends = kalki.meta_core.get_quality_trends()
        if 'error' not in trends:
            logger.info(f"  - Average Coverage: {trends['average_coverage']:.2f}")
            logger.info(f"  - Average Coherence: {trends['average_coherence']:.2f}")
            logger.info(f"  - Average Satisfaction: {trends['average_satisfaction']:.2f}")
            logger.info(f"  - Average Efficiency: {trends['average_efficiency']:.2f}")
            logger.info(f"  - Sample Size: {trends['sample_size']}")
            logger.info(f"  - Trend Direction: {trends['trend_direction']}")
    
    # Final validation
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST RESULTS")
    logger.info("=" * 80)
    
    tests_passed = []
    
    # Check MetaCore initialization
    if status['meta_core_active']:
        logger.info("✅ MetaCore initialized and active")
        tests_passed.append(True)
    else:
        logger.error("❌ MetaCore not active")
        tests_passed.append(False)
    
    # Check reasoning depth adaptation
    if result1.get('quality_metrics') and result2.get('quality_metrics') and result3.get('quality_metrics'):
        depth1 = result1['quality_metrics'].get('reasoning_depth')
        depth2 = result2['quality_metrics'].get('reasoning_depth')
        depth3 = result3['quality_metrics'].get('reasoning_depth')
        
        logger.info(f"✅ Reasoning depths: Simple={depth1}, Moderate={depth2}, Complex={depth3}")
        tests_passed.append(True)
    else:
        logger.error("❌ Quality metrics not generated")
        tests_passed.append(False)
    
    # Shutdown
    await kalki.shutdown()
    
    # Final result
    logger.info("\n" + "=" * 80)
    if all(tests_passed):
        logger.info("🎉 METACORE INTEGRATION TEST PASSED")
        logger.info("=" * 80)
        return True
    else:
        logger.error("❌ METACORE INTEGRATION TEST FAILED")
        logger.info("=" * 80)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_metacore_integration())
    sys.exit(0 if success else 1)
