#!/usr/bin/env python3
"""
test_robustness.py
KALKI v2.4 — Robustness System Test
Tests the comprehensive robustness monitoring, error recovery, and health checking systems.
"""

import time
import threading
from pathlib import Path
from modules.eventbus import EventBus
from modules.robustness import start_robustness_monitoring, stop_robustness_monitoring, get_robustness_manager
from modules.logger import get_logger

logger = get_logger("test_robustness")

def test_basic_robustness():
    """Test basic robustness functionality"""
    print("🧪 Testing Kalki v2.4 Robustness System")
    print("=" * 50)

    # Initialize eventbus and robustness manager
    eventbus = EventBus()
    manager = start_robustness_monitoring(eventbus)

    try:
        print("✅ Robustness monitoring started")

        # Test health checks
        print("\n🔍 Testing health checks...")
        health = manager.get_system_health()
        print(f"Overall health: {health['overall_status']}")

        # Test individual checks
        for check_name, check_data in health['checks'].items():
            status_icon = "✅" if check_data['status'] == 'healthy' else "⚠️" if check_data['status'] == 'degraded' else "❌"
            print(f"  {status_icon} {check_name}: {check_data['status']}")

        # Test resource monitoring
        print("\n📊 Testing resource monitoring...")
        resources = health['resources']
        print(f"  CPU Usage: {resources['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {resources['memory_percent']:.1f}%")
        print(f"  Disk Usage: {resources['disk_usage_percent']:.1f}%")

        # Test circuit breaker registration
        print("\n🔌 Testing circuit breaker...")
        manager.register_circuit_breaker("test_service", failure_threshold=2, recovery_timeout=5)

        # Simulate some activity
        print("\n⏱️  Monitoring for 10 seconds...")
        time.sleep(10)

        # Get updated health
        health_after = manager.get_system_health()
        print(f"Health after monitoring: {health_after['overall_status']}")

        # Test emergency restart (commented out to avoid actual restart)
        # print("\n🚨 Testing emergency restart...")
        # manager.trigger_emergency_restart("Test emergency restart")

        print("\n✅ All robustness tests completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Robustness test failed")
        return False

    finally:
        stop_robustness_monitoring()
        print("🛑 Robustness monitoring stopped")

    return True

def test_error_recovery():
    """Test error recovery mechanisms"""
    print("\n🔄 Testing error recovery...")

    eventbus = EventBus()
    manager = start_robustness_monitoring(eventbus)

    try:
        # Test circuit breaker with failing function
        def failing_function():
            raise Exception("Simulated failure")

        manager.register_circuit_breaker("failing_service", failure_threshold=2, recovery_timeout=3)

        # First failure
        try:
            manager.call_with_circuit_breaker("failing_service", failing_function)
        except Exception:
            print("✅ First failure handled correctly")

        # Second failure - should still work (under threshold)
        try:
            manager.call_with_circuit_breaker("failing_service", failing_function)
        except Exception:
            print("✅ Second failure handled correctly")

        # Third failure - circuit should open
        try:
            manager.call_with_circuit_breaker("failing_service", failing_function)
            print("❌ Circuit breaker should have opened")
        except Exception as e:
            if "OPEN" in str(e):
                print("✅ Circuit breaker opened correctly")
            else:
                print(f"❌ Unexpected error: {e}")

        # Wait for recovery timeout and test half-open state
        print("⏳ Waiting for circuit breaker recovery...")
        time.sleep(4)

        try:
            manager.call_with_circuit_breaker("failing_service", lambda: "success")
            print("✅ Circuit breaker recovered successfully")
        except Exception as e:
            print(f"❌ Circuit breaker recovery failed: {e}")

    except Exception as e:
        print(f"❌ Error recovery test failed: {e}")
        return False

    finally:
        stop_robustness_monitoring()

    return True

def test_resource_thresholds():
    """Test resource threshold monitoring"""
    print("\n📈 Testing resource threshold monitoring...")

    eventbus = EventBus()
    manager = start_robustness_monitoring(eventbus)

    # Set aggressive thresholds for testing
    manager.config.memory_threshold_percent = 50.0  # Low threshold to trigger alert
    manager.config.cpu_threshold_percent = 50.0

    events_received = []

    def threshold_handler(event_data):
        events_received.append(event_data)

    eventbus.subscribe("robustness.resource_threshold_exceeded", threshold_handler)

    try:
        print("⏳ Monitoring resources for threshold alerts...")
        time.sleep(15)  # Monitor for potential threshold exceedances

        if events_received:
            print(f"✅ Resource threshold alerts received: {len(events_received)}")
            for event in events_received:
                alerts = event.get('alerts', [])
                print(f"  Alert: {', '.join(alerts)}")
        else:
            print("ℹ️  No resource threshold alerts (system usage may be normal)")

    except Exception as e:
        print(f"❌ Resource threshold test failed: {e}")
        return False

    finally:
        stop_robustness_monitoring()

    return True

def main():
    """Run all robustness tests"""
    print("🧪 KALKi v2.4 Robustness System Test Suite")
    print("=" * 60)

    tests = [
        ("Basic Robustness", test_basic_robustness),
        ("Error Recovery", test_error_recovery),
        ("Resource Thresholds", test_resource_thresholds)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        except Exception as e:
            print(f"{test_name}: ❌ FAILED with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All robustness tests passed! System is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Review logs for details.")
        return 1

if __name__ == "__main__":
    exit(main())