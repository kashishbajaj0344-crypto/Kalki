"""
Real-World Telemetry Integration
Connects KALKI to deployed designs in the physical world to learn from
real performance data and continuously improve.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
from enum import Enum

logger = logging.getLogger(__name__)


class TelemetryType(Enum):
    """Types of telemetry data"""
    PERFORMANCE = "performance"
    STRUCTURAL = "structural"
    THERMAL = "thermal"
    ACOUSTIC = "acoustic"
    USER_FEEDBACK = "user_feedback"
    FAILURE_MODE = "failure_mode"
    ENVIRONMENTAL = "environmental"


@dataclass
class TelemetryDataPoint:
    """Single telemetry measurement from a deployed design"""
    design_id: str
    telemetry_type: TelemetryType
    timestamp: datetime
    measurements: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    anomaly_detected: bool = False
    
    
@dataclass
class DeployedDesign:
    """Tracking info for a design deployed in the real world"""
    design_id: str
    project_id: str
    deployment_timestamp: datetime
    location: str
    telemetry_endpoints: List[str]
    expected_performance: Dict[str, float]
    actual_performance: Dict[str, float] = field(default_factory=dict)
    issues_detected: List[str] = field(default_factory=list)
    total_data_points: int = 0
    last_update: Optional[datetime] = None
    
    
@dataclass
class LearningInsight:
    """Insights learned from real-world telemetry"""
    insight_id: str
    source_designs: List[str]
    insight_type: str  # 'design_improvement', 'failure_pattern', 'optimization'
    description: str
    confidence: float
    recommended_actions: List[str]
    supporting_data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    applied_to_system: bool = False


class RealWorldTelemetryIntegration:
    """
    Integrates real-world telemetry from deployed designs to enable
    continuous learning and improvement.
    
    Key Features:
    - Real-time telemetry ingestion from deployed designs
    - Anomaly detection and alerting
    - Performance vs. prediction comparison
    - Automated insight extraction
    - Feedback loop into design process
    """
    
    def __init__(self):
        self.deployed_designs: Dict[str, DeployedDesign] = {}
        self.telemetry_buffer: List[TelemetryDataPoint] = []
        self.learning_insights: List[LearningInsight] = []
        self.is_running = False
        self.telemetry_check_interval = 300  # 5 minutes
        
    async def initialize(self):
        """Initialize telemetry integration"""
        logger.info("📡 Initializing Real-World Telemetry Integration")
        
        # Load deployed designs registry
        await self._load_deployed_designs()
        
        # Load learning insights
        await self._load_learning_insights()
        
        logger.info(f"✅ Tracking {len(self.deployed_designs)} deployed designs")
        logger.info(f"📚 Loaded {len(self.learning_insights)} learning insights")
        
    async def start(self):
        """Start telemetry collection"""
        if self.is_running:
            logger.warning("Telemetry integration already running")
            return
            
        self.is_running = True
        logger.info("🔄 Starting telemetry collection")
        
        while self.is_running:
            try:
                # Collect telemetry from all deployed designs
                await self._collect_telemetry()
                
                # Process and analyze telemetry
                await self._process_telemetry()
                
                # Extract learning insights
                await self._extract_insights()
                
                # Wait for next cycle
                await asyncio.sleep(self.telemetry_check_interval)
                
            except Exception as e:
                logger.error(f"Telemetry collection error: {e}", exc_info=True)
                await asyncio.sleep(60)
                
    async def stop(self):
        """Stop telemetry collection"""
        self.is_running = False
        logger.info("⏸️ Telemetry collection stopped")
        
    async def register_deployment(self, design_id: str, project_id: str, 
                                  location: str, telemetry_endpoints: List[str],
                                  expected_performance: Dict[str, float]):
        """Register a new deployed design for telemetry tracking"""
        deployment = DeployedDesign(
            design_id=design_id,
            project_id=project_id,
            deployment_timestamp=datetime.now(),
            location=location,
            telemetry_endpoints=telemetry_endpoints,
            expected_performance=expected_performance
        )
        
        self.deployed_designs[design_id] = deployment
        await self._save_deployed_designs()
        
        logger.info(f"✅ Registered deployment: {design_id} at {location}")
        
    async def ingest_telemetry(self, design_id: str, telemetry_type: TelemetryType,
                               measurements: Dict[str, float], metadata: Optional[Dict] = None):
        """Ingest a telemetry data point"""
        if design_id not in self.deployed_designs:
            logger.warning(f"Unknown design_id: {design_id}")
            return
            
        data_point = TelemetryDataPoint(
            design_id=design_id,
            telemetry_type=telemetry_type,
            timestamp=datetime.now(),
            measurements=measurements,
            metadata=metadata or {}
        )
        
        # Check for anomalies
        data_point.anomaly_detected = await self._detect_anomaly(data_point)
        
        # Add to buffer
        self.telemetry_buffer.append(data_point)
        
        # Update deployment tracking
        deployment = self.deployed_designs[design_id]
        deployment.total_data_points += 1
        deployment.last_update = datetime.now()
        
        # Update actual performance metrics
        for metric, value in measurements.items():
            if metric not in deployment.actual_performance:
                deployment.actual_performance[metric] = value
            else:
                # Running average
                n = deployment.total_data_points
                deployment.actual_performance[metric] = (
                    deployment.actual_performance[metric] * (n - 1) / n + value / n
                )
                
        if data_point.anomaly_detected:
            logger.warning(f"⚠️ Anomaly detected in {design_id}: {telemetry_type.value}")
            deployment.issues_detected.append(
                f"{telemetry_type.value}_anomaly_{datetime.now().isoformat()}"
            )
            
    async def _collect_telemetry(self):
        """Collect telemetry from all deployed designs"""
        for design_id, deployment in self.deployed_designs.items():
            try:
                # In production, this would poll actual telemetry endpoints
                # For now, simulate with example data
                
                # Simulate performance telemetry
                if deployment.telemetry_endpoints:
                    # Example: simulate stress sensor data
                    await self.ingest_telemetry(
                        design_id=design_id,
                        telemetry_type=TelemetryType.STRUCTURAL,
                        measurements={
                            'max_stress_mpa': 150.0 + (hash(design_id) % 50),
                            'deflection_mm': 2.5 + (hash(design_id) % 10) / 10,
                            'safety_factor': 3.2
                        },
                        metadata={'sensor_id': 'strain_gauge_001'}
                    )
                    
            except Exception as e:
                logger.error(f"Error collecting from {design_id}: {e}")
                
    async def _process_telemetry(self):
        """Process buffered telemetry data"""
        if not self.telemetry_buffer:
            return
            
        logger.info(f"📊 Processing {len(self.telemetry_buffer)} telemetry data points")
        
        # Group by design
        by_design: Dict[str, List[TelemetryDataPoint]] = {}
        for data_point in self.telemetry_buffer:
            if data_point.design_id not in by_design:
                by_design[data_point.design_id] = []
            by_design[data_point.design_id].append(data_point)
            
        # Analyze each design's performance
        for design_id, data_points in by_design.items():
            await self._analyze_design_performance(design_id, data_points)
            
        # Clear processed buffer
        self.telemetry_buffer.clear()
        
    async def _analyze_design_performance(self, design_id: str, 
                                         data_points: List[TelemetryDataPoint]):
        """Analyze performance for a specific design"""
        deployment = self.deployed_designs.get(design_id)
        if not deployment:
            return
            
        # Compare actual vs expected performance
        deviations = {}
        for metric, expected in deployment.expected_performance.items():
            actual = deployment.actual_performance.get(metric)
            if actual is not None:
                deviation = abs(actual - expected) / expected if expected != 0 else abs(actual)
                deviations[metric] = deviation
                
                # Flag significant deviations
                if deviation > 0.2:  # 20% deviation threshold
                    logger.warning(
                        f"⚠️ Performance deviation in {design_id}:"
                        f" {metric} expected={expected:.2f}, actual={actual:.2f}"
                        f" (deviation={deviation:.1%})"
                    )
                    
        # Log performance summary
        anomaly_count = sum(1 for dp in data_points if dp.anomaly_detected)
        if anomaly_count > 0:
            logger.info(f"🔍 {design_id}: {anomaly_count} anomalies in {len(data_points)} readings")
            
    async def _detect_anomaly(self, data_point: TelemetryDataPoint) -> bool:
        """Detect if a data point represents an anomaly"""
        deployment = self.deployed_designs.get(data_point.design_id)
        if not deployment:
            return False
            
        # Check against expected performance
        for metric, value in data_point.measurements.items():
            expected = deployment.expected_performance.get(metric)
            if expected is not None:
                # Flag if value deviates significantly
                deviation = abs(value - expected) / expected if expected != 0 else abs(value)
                if deviation > 0.5:  # 50% deviation = anomaly
                    return True
                    
        # Check for extreme values based on type
        if data_point.telemetry_type == TelemetryType.STRUCTURAL:
            max_stress = data_point.measurements.get('max_stress_mpa', 0)
            if max_stress > 500:  # Extreme stress
                return True
            safety_factor = data_point.measurements.get('safety_factor', 10)
            if safety_factor < 1.5:  # Low safety factor
                return True
                
        elif data_point.telemetry_type == TelemetryType.THERMAL:
            max_temp = data_point.measurements.get('max_temperature_c', 0)
            if max_temp > 100:  # Overheating
                return True
                
        return False
        
    async def _extract_insights(self):
        """Extract learning insights from telemetry data"""
        # Analyze deployed designs for patterns
        successful_designs = []
        problematic_designs = []
        
        for design_id, deployment in self.deployed_designs.items():
            # Categorize based on performance
            deviations = []
            for metric, expected in deployment.expected_performance.items():
                actual = deployment.actual_performance.get(metric)
                if actual is not None and expected != 0:
                    deviation = abs(actual - expected) / expected
                    deviations.append(deviation)
                    
            avg_deviation = sum(deviations) / len(deviations) if deviations else 0
            
            if avg_deviation < 0.1 and not deployment.issues_detected:
                successful_designs.append(design_id)
            elif avg_deviation > 0.2 or deployment.issues_detected:
                problematic_designs.append(design_id)
                
        # Generate insights from successful patterns
        if len(successful_designs) >= 3:
            insight = LearningInsight(
                insight_id=f"insight_{datetime.now().timestamp()}",
                source_designs=successful_designs,
                insight_type='design_improvement',
                description=f"Identified {len(successful_designs)} designs performing within 10% of predictions",
                confidence=0.85,
                recommended_actions=[
                    "Extract common design patterns from successful designs",
                    "Reinforce validation methods that produced accurate predictions",
                    "Apply successful parameter ranges to new designs"
                ],
                supporting_data={
                    'successful_count': len(successful_designs),
                    'avg_accuracy': 0.95
                }
            )
            self.learning_insights.append(insight)
            logger.info(f"💡 New insight: {insight.description}")
            
        # Generate insights from failures
        if problematic_designs:
            insight = LearningInsight(
                insight_id=f"insight_{datetime.now().timestamp()}_failure",
                source_designs=problematic_designs,
                insight_type='failure_pattern',
                description=f"Identified {len(problematic_designs)} designs with performance issues",
                confidence=0.90,
                recommended_actions=[
                    "Analyze common failure modes",
                    "Improve validation thresholds",
                    "Enhance design constraints for affected parameters"
                ],
                supporting_data={
                    'problematic_count': len(problematic_designs),
                    'common_issues': list(set(
                        issue for d in problematic_designs
                        for issue in self.deployed_designs[d].issues_detected
                    ))
                }
            )
            self.learning_insights.append(insight)
            logger.warning(f"⚠️ Failure pattern: {insight.description}")
            
        await self._save_learning_insights()
        
    async def _load_deployed_designs(self):
        """Load deployed designs registry"""
        try:
            registry_path = Path("data/deployed_designs.json")
            if registry_path.exists():
                with open(registry_path) as f:
                    data = json.load(f)
                    for item in data:
                        deployment = DeployedDesign(
                            design_id=item['design_id'],
                            project_id=item['project_id'],
                            deployment_timestamp=datetime.fromisoformat(item['deployment_timestamp']),
                            location=item['location'],
                            telemetry_endpoints=item['telemetry_endpoints'],
                            expected_performance=item['expected_performance'],
                            actual_performance=item.get('actual_performance', {}),
                            issues_detected=item.get('issues_detected', []),
                            total_data_points=item.get('total_data_points', 0)
                        )
                        self.deployed_designs[deployment.design_id] = deployment
        except Exception as e:
            logger.error(f"Error loading deployed designs: {e}")
            
    async def _save_deployed_designs(self):
        """Save deployed designs registry"""
        try:
            registry_path = Path("data/deployed_designs.json")
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    'design_id': d.design_id,
                    'project_id': d.project_id,
                    'deployment_timestamp': d.deployment_timestamp.isoformat(),
                    'location': d.location,
                    'telemetry_endpoints': d.telemetry_endpoints,
                    'expected_performance': d.expected_performance,
                    'actual_performance': d.actual_performance,
                    'issues_detected': d.issues_detected,
                    'total_data_points': d.total_data_points
                }
                for d in self.deployed_designs.values()
            ]
            
            with open(registry_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving deployed designs: {e}")
            
    async def _load_learning_insights(self):
        """Load learning insights"""
        try:
            insights_path = Path("data/learning_insights.json")
            if insights_path.exists():
                with open(insights_path) as f:
                    data = json.load(f)
                    for item in data:
                        insight = LearningInsight(
                            insight_id=item['insight_id'],
                            source_designs=item['source_designs'],
                            insight_type=item['insight_type'],
                            description=item['description'],
                            confidence=item['confidence'],
                            recommended_actions=item['recommended_actions'],
                            supporting_data=item['supporting_data'],
                            created_at=datetime.fromisoformat(item['created_at']),
                            applied_to_system=item.get('applied_to_system', False)
                        )
                        self.learning_insights.append(insight)
        except Exception as e:
            logger.error(f"Error loading learning insights: {e}")
            
    async def _save_learning_insights(self):
        """Save learning insights"""
        try:
            insights_path = Path("data/learning_insights.json")
            insights_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    'insight_id': i.insight_id,
                    'source_designs': i.source_designs,
                    'insight_type': i.insight_type,
                    'description': i.description,
                    'confidence': i.confidence,
                    'recommended_actions': i.recommended_actions,
                    'supporting_data': i.supporting_data,
                    'created_at': i.created_at.isoformat(),
                    'applied_to_system': i.applied_to_system
                }
                for i in self.learning_insights
            ]
            
            with open(insights_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving learning insights: {e}")
            
    def get_telemetry_status(self) -> Dict[str, Any]:
        """Get current telemetry integration status"""
        total_data_points = sum(d.total_data_points for d in self.deployed_designs.values())
        designs_with_issues = sum(1 for d in self.deployed_designs.values() if d.issues_detected)
        
        return {
            'is_running': self.is_running,
            'deployed_designs': len(self.deployed_designs),
            'total_data_points_collected': total_data_points,
            'designs_with_issues': designs_with_issues,
            'learning_insights': len(self.learning_insights),
            'unapplied_insights': len([i for i in self.learning_insights if not i.applied_to_system]),
            'buffer_size': len(self.telemetry_buffer)
        }


# Singleton instance
_telemetry_integration = None

def get_telemetry_integration() -> RealWorldTelemetryIntegration:
    """Get the global telemetry integration instance"""
    global _telemetry_integration
    if _telemetry_integration is None:
        _telemetry_integration = RealWorldTelemetryIntegration()
    return _telemetry_integration
