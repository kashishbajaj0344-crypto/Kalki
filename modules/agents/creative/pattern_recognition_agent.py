#!/usr/bin/env python3
"""
PatternRecognitionAgent - Advanced pattern discovery and analysis

Enhanced with statistical detectors, clustering algorithms, and trend analysis
for improved insights and predictive capabilities.
"""

import asyncio
import logging
import random
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

from ..base_agent import BaseAgent, AgentCapability
from ..memory.memory_agent import MemoryAgent
from ..cognitive.performance_monitor import PerformanceMonitorAgent
from ...eventbus import EventBus

logger = logging.getLogger("kalki.agents.pattern_recognition")


class StatisticalDetector:
    """Statistical pattern detection utilities"""

    @staticmethod
    def detect_trends(data: List[float], window_size: int = 5) -> List[Dict[str, Any]]:
        """Detect trends in numerical data"""
        trends = []

        if len(data) < window_size * 2:
            return trends

        for i in range(window_size, len(data) - window_size + 1):
            prev_window = data[i-window_size:i]
            next_window = data[i:i+window_size]

            prev_mean = np.mean(prev_window)
            next_mean = np.mean(next_window)

            slope = (next_mean - prev_mean) / window_size

            if abs(slope) > np.std(data) * 0.1:  # Significant slope
                trend = {
                    "type": "trend",
                    "direction": "increasing" if slope > 0 else "decreasing",
                    "magnitude": abs(slope),
                    "position": i,
                    "confidence": min(1.0, abs(slope) / (np.std(data) + 0.001))
                }
                trends.append(trend)

        return trends

    @staticmethod
    def detect_outliers(data: List[float], threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect statistical outliers"""
        outliers = []

        if len(data) < 3:
            return outliers

        mean = np.mean(data)
        std = np.std(data)

        for i, value in enumerate(data):
            z_score = abs(value - mean) / (std + 0.001)
            if z_score > threshold:
                outliers.append({
                    "type": "outlier",
                    "position": i,
                    "value": value,
                    "z_score": z_score,
                    "deviation": value - mean
                })

        return outliers

    @staticmethod
    def detect_cycles(data: List[float], min_period: int = 3, max_period: int = 20) -> List[Dict[str, Any]]:
        """Detect cyclic patterns using autocorrelation"""
        cycles = []

        if len(data) < max_period * 2:
            return cycles

        # Compute autocorrelation
        autocorr = []
        for lag in range(1, min(max_period + 1, len(data) // 2)):
            corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            autocorr.append((lag, corr))

        # Find peaks in autocorrelation
        for lag, corr in autocorr:
            if corr > 0.5:  # Strong correlation
                cycles.append({
                    "type": "cycle",
                    "period": lag,
                    "correlation": corr,
                    "strength": "strong" if corr > 0.7 else "moderate"
                })

        return cycles

    @staticmethod
    def calculate_distribution_stats(data: List[float]) -> Dict[str, Any]:
        """Calculate comprehensive distribution statistics"""
        if not data:
            return {}

        data_array = np.array(data)

        return {
            "count": len(data),
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "std": float(np.std(data_array)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "skewness": float(stats.skew(data_array)),
            "kurtosis": float(stats.kurtosis(data_array)),
            "quartiles": [
                float(np.percentile(data_array, 25)),
                float(np.percentile(data_array, 50)),
                float(np.percentile(data_array, 75))
            ]
        }


class ClusteringAnalyzer:
    """Clustering analysis for pattern discovery"""

    def __init__(self):
        self.scaler = StandardScaler()

    def cluster_numerical_data(self, data_points: List[List[float]],
                             n_clusters: int = 3) -> Dict[str, Any]:
        """Cluster numerical data points"""
        try:
            if len(data_points) < n_clusters:
                return {"error": "Insufficient data points for clustering"}

            # Convert to numpy array
            X = np.array(data_points)

            # Scale the data
            X_scaled = self.scaler.fit_transform(X)

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_points = X[clusters == i]
                if len(cluster_points) > 0:
                    centroid = kmeans.cluster_centers_[i]
                    stats = StatisticalDetector.calculate_distribution_stats(
                        cluster_points[:, 0].tolist() if cluster_points.shape[1] == 1
                        else [np.mean(point) for point in cluster_points]
                    )
                    cluster_stats.append({
                        "cluster_id": i,
                        "size": len(cluster_points),
                        "centroid": centroid.tolist(),
                        "statistics": stats
                    })

            return {
                "n_clusters": n_clusters,
                "clusters": cluster_stats,
                "inertia": float(kmeans.inertia_),
                "silhouette_score": self._calculate_silhouette_score(X_scaled, clusters)
            }

        except Exception as e:
            logger.exception(f"Clustering failed: {e}")
            return {"error": str(e)}

    def _calculate_silhouette_score(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(X, labels))
        except:
            return 0.0


class PatternRecognitionAgent(BaseAgent):
    """
    Advanced pattern recognition with statistical analysis and clustering.

    Features:
    - Statistical detectors (trends, outliers, cycles)
    - Clustering algorithms for pattern discovery
    - Multi-dimensional pattern analysis
    - Predictive insights and recommendations
    - Persistence and knowledge integration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="PatternRecognitionAgent",
            capabilities=[
                AgentCapability.PATTERN_RECOGNITION,
                AgentCapability.ANALYTICS,
                AgentCapability.PREDICTIVE_DISCOVERY
            ],
            description="Advanced pattern discovery with statistical analysis and clustering",
            config=config
        )

        # Configuration
        self.enable_statistical_analysis = self.config.get('enable_statistical_analysis', True)
        self.enable_clustering = self.config.get('enable_clustering', True)
        self.max_clusters = self.config.get('max_clusters', 5)
        self.enable_persistence = self.config.get('enable_persistence', True)
        self.enable_events = self.config.get('enable_events', True)

        # Components
        self.statistical_detector = StatisticalDetector()
        self.clustering_analyzer = ClusteringAnalyzer()

        # State
        self.patterns = []
        self.analysis_history = []
        self.domain_patterns = defaultdict(list)

        # Dependencies
        self.knowledge_agent = None
        self.event_bus = None
        self.memory_agent = None
        self.performance_monitor = None

    def set_knowledge_agent(self, knowledge_agent):
        """Set the knowledge agent for persistence"""
        self.knowledge_agent = knowledge_agent

    def set_event_bus(self, event_bus):
        """Set the event bus for broadcasting"""
        self.event_bus = event_bus

    def set_memory_agent(self, memory_agent: MemoryAgent):
        """Set the memory agent for episodic storage"""
        self.memory_agent = memory_agent

    def set_performance_monitor(self, performance_monitor: PerformanceMonitorAgent):
        """Set the performance monitor for metrics tracking"""
        self.performance_monitor = performance_monitor

    async def initialize(self) -> bool:
        """Initialize the pattern recognition agent"""
        try:
            logger.info("PatternRecognitionAgent initializing")
            logger.info("PatternRecognitionAgent initialized successfully")
            return True

        except Exception as e:
            logger.exception(f"PatternRecognitionAgent initialization failed: {e}")
            return False

    async def analyze_patterns(self, data: List[Dict[str, Any]],
                             analysis_type: str = "comprehensive",
                             domain: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive pattern analysis"""
        start_time = time.time()
        try:
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            patterns_found = []

            # Extract numerical data for statistical analysis
            numerical_data = self._extract_numerical_data(data)

            if self.enable_statistical_analysis and numerical_data:
                # Statistical pattern detection
                stat_patterns = await self._perform_statistical_analysis(numerical_data)
                patterns_found.extend(stat_patterns)

            # Clustering analysis
            if self.enable_clustering and len(data) >= 3:
                cluster_patterns = await self._perform_clustering_analysis(data)
                patterns_found.extend(cluster_patterns)

            # Categorical pattern analysis
            categorical_patterns = self._analyze_categorical_patterns(data)
            patterns_found.extend(categorical_patterns)

            # Frequency and correlation analysis
            frequency_patterns = self._analyze_frequencies(data)
            patterns_found.extend(frequency_patterns)

            # Create analysis result
            analysis_result = {
                "analysis_id": analysis_id,
                "analysis_type": analysis_type,
                "domain": domain,
                "data_points": len(data),
                "patterns_found": len(patterns_found),
                "patterns": patterns_found,
                "insights": await self._generate_insights(patterns_found, domain),
                "recommendations": self._generate_recommendations(patterns_found),
                "created_at": datetime.now().isoformat()
            }

            # Store patterns
            for pattern in patterns_found:
                pattern["analysis_id"] = analysis_id
                pattern["detected_at"] = datetime.now().isoformat()
                self.patterns.append(pattern)

                if domain:
                    self.domain_patterns[domain].append(pattern)

            # Store analysis
            self.analysis_history.append(analysis_result)

            # Persist if enabled
            if self.enable_persistence and self.knowledge_agent:
                meta = {
                    "source_agent": "PatternRecognitionAgent",
                    "analysis_type": analysis_type,
                    "domain": domain,
                    "patterns_found": len(patterns_found)
                }
                await self.knowledge_agent.create_version(
                    knowledge_id=analysis_id,
                    content=analysis_result,
                    metadata=meta
                )

            # Store in memory agent
            if self.memory_agent:
                await self.memory_agent.store_episodic({
                    "event_type": "pattern_analysis_completed",
                    "analysis_id": analysis_id,
                    "data_points": len(data),
                    "patterns_found": len(patterns_found),
                    "domain": domain,
                    "analysis_type": analysis_type,
                    "timestamp": datetime.now().isoformat()
                })

                # Semantic memory for pattern insights
                for pattern in patterns_found:
                    await self.memory_agent.store_semantic(
                        concept=f"pattern_{pattern.get('type', 'unknown')}",
                        knowledge={
                            "pattern_id": pattern.get("pattern_id"),
                            "type": pattern.get("type"),
                            "description": pattern.get("description"),
                            "confidence": pattern.get("confidence"),
                            "domain": domain,
                            "analysis_id": analysis_id,
                            "created_at": datetime.now().isoformat()
                        }
                    )

            # Record performance metrics
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_metric(
                    "pattern_analysis_duration",
                    duration,
                    {
                        "data_points": len(data),
                        "patterns_found": len(patterns_found),
                        "domain": domain,
                        "analysis_type": analysis_type
                    }
                )

            # Emit event
            if self.enable_events and self.event_bus:
                await self.event_bus.publish("patterns.detected", {
                    "agent": "PatternRecognitionAgent",
                    "analysis_id": analysis_id,
                    "patterns_count": len(patterns_found),
                    "domain": domain
                })

            logger.info(f"Completed pattern analysis {analysis_id}: {len(patterns_found)} patterns found")
            return {
                "status": "success",
                "analysis": analysis_result
            }

        except Exception as e:
            logger.exception(f"Pattern analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _extract_numerical_data(self, data: List[Dict[str, Any]]) -> List[float]:
        """Extract numerical values from data for statistical analysis"""
        numerical_values = []

        for item in data:
            # Look for numerical fields
            for key, value in item.items():
                if isinstance(value, (int, float)):
                    numerical_values.append(float(value))
                elif isinstance(value, str):
                    # Try to extract numbers from strings
                    try:
                        import re
                        numbers = re.findall(r'\d+\.?\d*', value)
                        numerical_values.extend(float(n) for n in numbers)
                    except:
                        pass

        return numerical_values

    async def _perform_statistical_analysis(self, data: List[float]) -> List[Dict[str, Any]]:
        """Perform statistical pattern detection"""
        patterns = []

        try:
            # Trend detection
            trends = self.statistical_detector.detect_trends(data)
            for trend in trends:
                trend["category"] = "statistical"
                trend["subtype"] = "trend"
                patterns.append(trend)

            # Outlier detection
            outliers = self.statistical_detector.detect_outliers(data)
            for outlier in outliers:
                outlier["category"] = "statistical"
                outlier["subtype"] = "outlier"
                patterns.append(outlier)

            # Cycle detection
            cycles = self.statistical_detector.detect_cycles(data)
            for cycle in cycles:
                cycle["category"] = "statistical"
                cycle["subtype"] = "cycle"
                patterns.append(cycle)

            # Distribution analysis
            if data:
                dist_stats = self.statistical_detector.calculate_distribution_stats(data)
                patterns.append({
                    "category": "statistical",
                    "subtype": "distribution",
                    "description": f"Data distribution analysis: mean={dist_stats.get('mean', 0):.2f}, std={dist_stats.get('std', 0):.2f}",
                    "statistics": dist_stats,
                    "confidence": 0.9
                })

        except Exception as e:
            logger.exception(f"Statistical analysis failed: {e}")

        return patterns

    async def _perform_clustering_analysis(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform clustering analysis on data"""
        patterns = []

        try:
            # Extract feature vectors from data
            feature_vectors = []
            feature_names = []

            # Find common numerical fields
            all_keys = set()
            for item in data:
                all_keys.update(item.keys())

            numerical_keys = []
            for key in all_keys:
                values = [item.get(key) for item in data]
                if all(isinstance(v, (int, float)) or v is None for v in values):
                    numerical_keys.append(key)

            if len(numerical_keys) >= 2:  # Need at least 2 features for meaningful clustering
                for item in data:
                    vector = []
                    for key in numerical_keys:
                        value = item.get(key, 0)
                        vector.append(float(value) if value is not None else 0.0)
                    feature_vectors.append(vector)

                feature_names = numerical_keys

                # Perform clustering
                n_clusters = min(self.max_clusters, len(feature_vectors) // 2, 5)
                if n_clusters >= 2:
                    cluster_result = self.clustering_analyzer.cluster_numerical_data(
                        feature_vectors, n_clusters
                    )

                    if "error" not in cluster_result:
                        patterns.append({
                            "category": "clustering",
                            "subtype": "kmeans",
                            "description": f"Identified {n_clusters} clusters in {len(feature_vectors)} data points",
                            "clusters": cluster_result["clusters"],
                            "features": feature_names,
                            "quality_score": cluster_result.get("silhouette_score", 0),
                            "confidence": 0.8
                        })

        except Exception as e:
            logger.exception(f"Clustering analysis failed: {e}")

        return patterns

    def _analyze_categorical_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze patterns in categorical data"""
        patterns = []

        try:
            # Find categorical fields
            categorical_fields = {}
            for item in data:
                for key, value in item.items():
                    if isinstance(value, str) and key not in categorical_fields:
                        categorical_fields[key] = []

                    if isinstance(value, str):
                        categorical_fields[key].append(value)

            # Analyze each categorical field
            for field, values in categorical_fields.items():
                if len(values) > 1:
                    # Frequency analysis
                    freq_counter = Counter(values)
                    total = len(values)

                    # Find dominant categories
                    dominant = freq_counter.most_common(3)
                    if dominant and dominant[0][1] / total > 0.5:  # More than 50% dominance
                        patterns.append({
                            "category": "categorical",
                            "subtype": "dominant_category",
                            "field": field,
                            "description": f"Field '{field}' shows dominance of '{dominant[0][0]}' ({dominant[0][1]}/{total})",
                            "dominant_values": dominant,
                            "confidence": min(1.0, dominant[0][1] / total)
                        })

                    # Entropy analysis (diversity)
                    entropy = self._calculate_entropy(values)
                    if entropy < 0.5:  # Low diversity
                        patterns.append({
                            "category": "categorical",
                            "subtype": "low_diversity",
                            "field": field,
                            "description": f"Field '{field}' shows low diversity (entropy: {entropy:.3f})",
                            "unique_values": len(set(values)),
                            "total_values": total,
                            "confidence": 0.7
                        })

        except Exception as e:
            logger.exception(f"Categorical analysis failed: {e}")

        return patterns

    def _analyze_frequencies(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze frequency patterns and correlations"""
        patterns = []

        try:
            # Simple frequency analysis across all fields
            field_frequencies = defaultdict(Counter)

            for item in data:
                for key, value in item.items():
                    field_frequencies[key][str(value)] += 1

            # Find fields with interesting frequency distributions
            for field, freq_dict in field_frequencies.items():
                total = sum(freq_dict.values())
                if total > 1:
                    # Calculate concentration (Herfindahl index)
                    concentrations = [(count/total)**2 for count in freq_dict.values()]
                    herfindahl = sum(concentrations)

                    if herfindahl > 0.6:  # High concentration
                        patterns.append({
                            "category": "frequency",
                            "subtype": "high_concentration",
                            "field": field,
                            "description": f"Field '{field}' shows high value concentration (Herfindahl: {herfindahl:.3f})",
                            "top_values": freq_dict.most_common(3),
                            "unique_values": len(freq_dict),
                            "confidence": 0.8
                        })

        except Exception as e:
            logger.exception(f"Frequency analysis failed: {e}")

        return patterns

    def _calculate_entropy(self, values: List[str]) -> float:
        """Calculate Shannon entropy for categorical data"""
        if not values:
            return 0.0

        total = len(values)
        freq_counter = Counter(values)

        entropy = 0.0
        for count in freq_counter.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(freq_counter)) if freq_counter else 0
        return entropy / max_entropy if max_entropy > 0 else 0

    async def _generate_insights(self, patterns: List[Dict[str, Any]],
                               domain: Optional[str]) -> List[str]:
        """Generate insights from detected patterns"""
        insights = []

        try:
            # Group patterns by category
            categories = defaultdict(list)
            for pattern in patterns:
                categories[pattern.get("category", "unknown")].append(pattern)

            # Generate insights for each category
            if "statistical" in categories:
                stat_patterns = categories["statistical"]
                trend_count = sum(1 for p in stat_patterns if p.get("subtype") == "trend")
                outlier_count = sum(1 for p in stat_patterns if p.get("subtype") == "outlier")

                if trend_count > 0:
                    insights.append(f"Detected {trend_count} significant trends in the data")
                if outlier_count > 0:
                    insights.append(f"Found {outlier_count} statistical outliers that may indicate anomalies")

            if "clustering" in categories:
                cluster_patterns = categories["clustering"]
                total_clusters = sum(len(p.get("clusters", [])) for p in cluster_patterns)
                insights.append(f"Data naturally groups into {total_clusters} distinct clusters")

            if "categorical" in categories:
                cat_patterns = categories["categorical"]
                dominant_count = sum(1 for p in cat_patterns if p.get("subtype") == "dominant_category")
                if dominant_count > 0:
                    insights.append(f"Identified {dominant_count} fields with dominant categorical values")

            # Domain-specific insights
            if domain:
                insights.append(f"Patterns detected in {domain} domain may inform strategic decisions")

        except Exception as e:
            logger.exception(f"Insight generation failed: {e}")
            insights.append("Pattern analysis completed but insight generation encountered issues")

        return insights

    def _generate_recommendations(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on patterns"""
        recommendations = []

        try:
            # Analyze pattern implications
            has_outliers = any(p.get("subtype") == "outlier" for p in patterns)
            has_trends = any(p.get("subtype") == "trend" for p in patterns)
            has_clusters = any(p.get("category") == "clustering" for p in patterns)

            if has_outliers:
                recommendations.append("Investigate outliers to understand anomalies or opportunities")
                recommendations.append("Consider robust statistical methods that handle outliers")

            if has_trends:
                recommendations.append("Monitor trend directions for predictive insights")
                recommendations.append("Validate trend significance with additional data")

            if has_clusters:
                recommendations.append("Use cluster analysis for segmentation strategies")
                recommendations.append("Validate cluster stability with cross-validation")

            if len(patterns) > 10:
                recommendations.append("Consider dimensionality reduction for complex pattern analysis")

            recommendations.append("Schedule regular pattern analysis to track changes over time")

        except Exception as e:
            logger.exception(f"Recommendation generation failed: {e}")

        return recommendations

    async def get_patterns_by_domain(self, domain: str) -> Dict[str, Any]:
        """Get patterns detected in a specific domain"""
        try:
            patterns = self.domain_patterns.get(domain, [])

            return {
                "status": "success",
                "domain": domain,
                "patterns": patterns,
                "total_patterns": len(patterns)
            }

        except Exception as e:
            logger.exception(f"Failed to get domain patterns: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern recognition tasks"""
        action = task.get("action")

        try:
            if action == "analyze_patterns":
                return await self.analyze_patterns(
                    task["data"],
                    task.get("analysis_type", "comprehensive"),
                    task.get("domain")
                )
            elif action == "get_domain_patterns":
                return await self.get_patterns_by_domain(task["domain"])
            elif action == "get_insights":
                # Generate insights from existing patterns
                domain = task.get("domain")
                patterns = self.domain_patterns.get(domain, []) if domain else self.patterns
                insights = await self._generate_insights(patterns, domain)
                recommendations = self._generate_recommendations(patterns)

                return {
                    "status": "success",
                    "insights": insights,
                    "recommendations": recommendations,
                    "patterns_analyzed": len(patterns)
                }
            elif action == "list_patterns":
                return {
                    "status": "success",
                    "patterns": self.patterns,
                    "total_count": len(self.patterns)
                }
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            return {"status": "error", "message": str(e)}

    async def shutdown(self) -> bool:
        """Shutdown the pattern recognition agent"""
        try:
            logger.info("PatternRecognitionAgent shutting down")

            # Clear state
            self.patterns.clear()
            self.analysis_history.clear()
            self.domain_patterns.clear()

            logger.info("PatternRecognitionAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"PatternRecognitionAgent shutdown failed: {e}")
            return False