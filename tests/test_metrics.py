"""
Unit tests for Phase 20 - Evaluation & Metrics
"""

import unittest
import time

from modules.metrics import (
    MetricsCollector,
    PerformanceMonitor,
    TaskMetrics,
    SystemMetrics
)


class TestMetricsCollector(unittest.TestCase):
    """Tests for MetricsCollector."""
    
    def setUp(self):
        """Create fresh collector for each test."""
        self.collector = MetricsCollector()
    
    def test_start_and_end_task(self):
        """Test basic task tracking."""
        self.collector.start_task("task1", "agent1")
        
        # Simulate work
        time.sleep(0.01)
        
        self.collector.end_task("task1", success=True)
        
        metrics = self.collector.get_task_metrics("task1")
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.task_id, "task1")
        self.assertEqual(metrics.agent_id, "agent1")
        self.assertTrue(metrics.success)
        self.assertGreater(metrics.latency, 0)
    
    def test_record_retry(self):
        """Test recording retries."""
        self.collector.start_task("task1", "agent1")
        self.collector.record_retry("task1")
        self.collector.record_retry("task1")
        
        metrics = self.collector.get_task_metrics("task1")
        
        self.assertEqual(metrics.retries, 2)
    
    def test_record_memory_lookup(self):
        """Test recording memory lookups."""
        self.collector.start_task("task1", "agent1")
        self.collector.record_memory_lookup("task1")
        self.collector.record_memory_lookup("task1")
        self.collector.record_memory_lookup("task1")
        self.collector.end_task("task1", success=True)
        
        metrics = self.collector.get_task_metrics("task1")
        
        self.assertEqual(metrics.memory_lookups, 3)
    
    def test_get_system_metrics(self):
        """Test getting aggregate system metrics."""
        # Create several tasks
        for i in range(5):
            task_id = f"task{i}"
            self.collector.start_task(task_id, f"agent{i % 2}")
            time.sleep(0.001)
            self.collector.end_task(task_id, success=(i % 3 != 0))
        
        system_metrics = self.collector.get_system_metrics()
        
        self.assertEqual(system_metrics.total_tasks, 5)
        self.assertGreater(system_metrics.successful_tasks, 0)
        self.assertGreater(system_metrics.average_latency, 0)
    
    def test_clear_metrics(self):
        """Test clearing all metrics."""
        self.collector.start_task("task1", "agent1")
        self.collector.end_task("task1", success=True)
        
        self.assertEqual(len(self.collector.task_metrics), 1)
        
        self.collector.clear()
        
        self.assertEqual(len(self.collector.task_metrics), 0)


class TestSystemMetrics(unittest.TestCase):
    """Tests for SystemMetrics."""
    
    def test_to_dict(self):
        """Test converting system metrics to dictionary."""
        collector = MetricsCollector()
        
        collector.start_task("task1", "agent1")
        collector.end_task("task1", success=True)
        
        system_metrics = collector.get_system_metrics()
        metrics_dict = system_metrics.to_dict()
        
        self.assertIn('total_tasks', metrics_dict)
        self.assertIn('success_rate', metrics_dict)
        self.assertIn('average_latency', metrics_dict)
        self.assertIn('agent_utilization', metrics_dict)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        collector = MetricsCollector()
        
        # 3 successful, 2 failed
        for i in range(5):
            collector.start_task(f"task{i}", "agent1")
            collector.end_task(f"task{i}", success=(i < 3))
        
        system_metrics = collector.get_system_metrics()
        metrics_dict = system_metrics.to_dict()
        
        self.assertEqual(metrics_dict['success_rate'], 0.6)  # 3/5


class TestPerformanceMonitor(unittest.TestCase):
    """Tests for PerformanceMonitor."""
    
    def setUp(self):
        """Create fresh monitor for each test."""
        self.collector = MetricsCollector()
        self.monitor = PerformanceMonitor(self.collector)
    
    def test_get_performance_report(self):
        """Test generating performance report."""
        # Add some tasks
        for i in range(10):
            self.collector.start_task(f"task{i}", "agent1")
            time.sleep(0.001)
            self.collector.end_task(f"task{i}", success=(i < 9))
        
        report = self.monitor.get_performance_report()
        
        self.assertIn('summary', report)
        self.assertIn('analysis', report)
        self.assertIn('recommendations', report)
    
    def test_performance_analysis(self):
        """Test performance analysis."""
        # Create tasks with high success rate
        for i in range(20):
            self.collector.start_task(f"task{i}", "agent1")
            self.collector.end_task(f"task{i}", success=True)
        
        report = self.monitor.get_performance_report()
        analysis = report['analysis']
        
        # Should be excellent with 100% success
        self.assertEqual(analysis['success_rate_status'], 'excellent')
    
    def test_recommendations_for_poor_performance(self):
        """Test that recommendations are generated for poor performance."""
        # Create tasks with low success rate
        for i in range(10):
            self.collector.start_task(f"task{i}", "agent1")
            self.collector.end_task(f"task{i}", success=(i < 3))
        
        report = self.monitor.get_performance_report()
        recommendations = report['recommendations']
        
        # Should have recommendations for low success rate
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any('success rate' in r.lower() for r in recommendations))
    
    def test_retry_analysis(self):
        """Test retry rate analysis."""
        # Create tasks with retries
        for i in range(5):
            task_id = f"task{i}"
            self.collector.start_task(task_id, "agent1")
            self.collector.record_retry(task_id)
            self.collector.record_retry(task_id)
            self.collector.end_task(task_id, success=True)
        
        report = self.monitor.get_performance_report()
        
        # Should detect high retry rate
        self.assertTrue(any('retry' in r.lower() for r in report['recommendations']))


if __name__ == '__main__':
    unittest.main()
