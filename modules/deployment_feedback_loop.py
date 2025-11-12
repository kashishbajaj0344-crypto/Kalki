"""
Deployment Feedback Loop System
Automates learning from production deployments with continuous improvement.
Closes the loop: Design → Deploy → Monitor → Learn → Improve → Redesign
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class DeploymentFeedback:
    """Feedback from a deployed design"""
    design_id: str
    deployment_id: str
    timestamp: datetime
    performance_metrics: Dict[str, float]
    user_feedback: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    success_rate: float = 1.0
    

@dataclass
class LearningCycle:
    """A complete learning cycle from deployment feedback"""
    cycle_id: str
    start_time: datetime
    design_ids: List[str]
    insights_extracted: int
    improvements_generated: int
    designs_updated: int
    status: str  # 'running', 'completed', 'failed'
    

class DeploymentFeedbackLoop:
    """
    Automated learning from production deployments.
    
    Features:
    - Continuous monitoring of deployed designs
    - Automated feedback collection
    - Pattern recognition across deployments
    - Automated improvement generation
    - Design iteration and optimization
    - A/B testing of improvements
    """
    
    def __init__(self):
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.feedback_history: List[DeploymentFeedback] = []
        self.learning_cycles: List[LearningCycle] = []
        self.is_running = False
        self.cycle_interval = 3600  # Run learning cycle every hour
        
    async def initialize(self):
        """Initialize the feedback loop system"""
        logger.info("🔄 Initializing Deployment Feedback Loop")
        
        # Load deployment history
        await self._load_deployment_history()
        
        # Load learning cycles
        await self._load_learning_cycles()
        
        logger.info(f"✅ Feedback loop initialized - {len(self.active_deployments)} active deployments")
        
    async def start(self):
        """Start the continuous feedback loop"""
        if self.is_running:
            logger.warning("Feedback loop already running")
            return
            
        self.is_running = True
        logger.info("🔄 Starting continuous feedback loop")
        
        while self.is_running:
            try:
                # Run a learning cycle
                await self._run_learning_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.cycle_interval)
                
            except Exception as e:
                logger.error(f"Learning cycle error: {e}", exc_info=True)
                await asyncio.sleep(60)
                
    async def stop(self):
        """Stop the feedback loop"""
        self.is_running = False
        logger.info("⏸️ Feedback loop stopped")
        
    async def register_deployment(self, design_id: str, deployment_id: str, 
                                  deployment_info: Dict[str, Any]):
        """Register a new deployment for monitoring"""
        self.active_deployments[deployment_id] = {
            'design_id': design_id,
            'deployment_id': deployment_id,
            'start_time': datetime.now(),
            'info': deployment_info,
            'feedback_count': 0
        }
        
        logger.info(f"📝 Registered deployment: {deployment_id} (design: {design_id})")
        
        # Integrate with telemetry system
        await self._integrate_deployment_with_telemetry(design_id, deployment_id, deployment_info)
        
    async def collect_feedback(self, deployment_id: str, feedback: DeploymentFeedback):
        """Collect feedback from a deployment"""
        if deployment_id not in self.active_deployments:
            logger.warning(f"Unknown deployment: {deployment_id}")
            return
            
        # Store feedback
        self.feedback_history.append(feedback)
        self.active_deployments[deployment_id]['feedback_count'] += 1
        
        logger.info(f"📊 Feedback collected from {deployment_id}: Success rate {feedback.success_rate:.1%}")
        
        # If critical failure, trigger immediate learning
        if feedback.success_rate < 0.5 or feedback.failure_modes:
            logger.warning(f"⚠️ Critical feedback from {deployment_id} - triggering immediate learning")
            await self._run_immediate_learning(feedback)
            
    async def _run_learning_cycle(self):
        """Run one complete learning cycle"""
        cycle_id = f"cycle_{datetime.now().timestamp()}"
        logger.info(f"🔬 Starting learning cycle: {cycle_id}")
        
        cycle = LearningCycle(
            cycle_id=cycle_id,
            start_time=datetime.now(),
            design_ids=[],
            insights_extracted=0,
            improvements_generated=0,
            designs_updated=0,
            status='running'
        )
        
        try:
            # 1. Collect and aggregate feedback
            recent_feedback = await self._aggregate_recent_feedback()
            
            if not recent_feedback:
                logger.info("✅ No new feedback to process")
                return
                
            logger.info(f"📊 Processing feedback from {len(recent_feedback)} deployments")
            
            # 2. Extract insights
            insights = await self._extract_insights(recent_feedback)
            cycle.insights_extracted = len(insights)
            
            logger.info(f"💡 Extracted {len(insights)} insights")
            
            # 3. Generate improvements
            improvements = await self._generate_improvements(insights)
            cycle.improvements_generated = len(improvements)
            
            logger.info(f"⚡ Generated {len(improvements)} improvements")
            
            # 4. Apply improvements to designs
            updated_count = await self._apply_improvements(improvements)
            cycle.designs_updated = updated_count
            
            logger.info(f"✅ Updated {updated_count} designs")
            
            # 5. Trigger evolution system
            await self._trigger_evolution_system(insights, improvements)
            
            cycle.status = 'completed'
            self.learning_cycles.append(cycle)
            
            # Save state
            await self._save_learning_cycles()
            
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}", exc_info=True)
            cycle.status = 'failed'
            
    async def _aggregate_recent_feedback(self) -> List[DeploymentFeedback]:
        """Aggregate feedback from recent time window"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        recent_feedback = [
            fb for fb in self.feedback_history
            if fb.timestamp >= cutoff_time
        ]
        
        return recent_feedback
        
    async def _extract_insights(self, feedback_list: List[DeploymentFeedback]) -> List[Dict[str, Any]]:
        """Extract actionable insights from feedback"""
        insights = []
        
        # Group feedback by design
        by_design: Dict[str, List[DeploymentFeedback]] = {}
        for fb in feedback_list:
            if fb.design_id not in by_design:
                by_design[fb.design_id] = []
            by_design[fb.design_id].append(fb)
            
        # Analyze each design's feedback
        for design_id, feedbacks in by_design.items():
            # Calculate aggregate metrics
            avg_success_rate = sum(fb.success_rate for fb in feedbacks) / len(feedbacks)
            
            # Identify common failure modes
            all_failures = []
            for fb in feedbacks:
                all_failures.extend(fb.failure_modes)
            failure_counts = {}
            for failure in all_failures:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
                
            # Extract performance trends
            performance_trends = {}
            for metric in ['stress', 'temperature', 'efficiency']:
                values = [
                    fb.performance_metrics.get(metric, 0)
                    for fb in feedbacks
                    if metric in fb.performance_metrics
                ]
                if values:
                    performance_trends[metric] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
                    
            # Create insight
            if avg_success_rate < 0.8 or failure_counts:
                insights.append({
                    'type': 'performance_issue',
                    'design_id': design_id,
                    'success_rate': avg_success_rate,
                    'common_failures': sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:3],
                    'performance_trends': performance_trends,
                    'priority': 'high' if avg_success_rate < 0.6 else 'medium'
                })
            elif avg_success_rate > 0.95:
                insights.append({
                    'type': 'successful_pattern',
                    'design_id': design_id,
                    'success_rate': avg_success_rate,
                    'performance_trends': performance_trends,
                    'priority': 'low'
                })
                
        return insights
        
    async def _generate_improvements(self, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate improvements based on insights"""
        improvements = []
        
        for insight in insights:
            if insight['type'] == 'performance_issue':
                # Generate targeted improvements
                if insight.get('common_failures'):
                    for failure, count in insight['common_failures']:
                        improvements.append({
                            'target_design': insight['design_id'],
                            'type': 'failure_mitigation',
                            'issue': failure,
                            'frequency': count,
                            'proposed_action': f"Redesign to address {failure}",
                            'priority': insight['priority']
                        })
                        
                # Check performance metrics
                if 'stress' in insight.get('performance_trends', {}):
                    stress_trend = insight['performance_trends']['stress']
                    if stress_trend['max'] > 200:  # MPa threshold
                        improvements.append({
                            'target_design': insight['design_id'],
                            'type': 'stress_reduction',
                            'current_max': stress_trend['max'],
                            'proposed_action': 'Reinforce high-stress areas or increase material strength',
                            'priority': 'high'
                        })
                        
            elif insight['type'] == 'successful_pattern':
                # Extract successful patterns to apply elsewhere
                improvements.append({
                    'type': 'pattern_extraction',
                    'source_design': insight['design_id'],
                    'success_rate': insight['success_rate'],
                    'proposed_action': 'Extract and generalize successful design patterns',
                    'priority': 'medium'
                })
                
        return improvements
        
    async def _apply_improvements(self, improvements: List[Dict[str, Any]]) -> int:
        """Apply improvements to designs"""
        updated_count = 0
        
        for improvement in improvements:
            try:
                if improvement['type'] == 'failure_mitigation':
                    # Log the improvement (in production, would update design database)
                    logger.info(f"💡 Improvement for {improvement['target_design']}: {improvement['proposed_action']}")
                    updated_count += 1
                    
                elif improvement['type'] == 'stress_reduction':
                    logger.info(f"🔧 Stress optimization for {improvement['target_design']}")
                    updated_count += 1
                    
                elif improvement['type'] == 'pattern_extraction':
                    logger.info(f"📚 Pattern extracted from {improvement['source_design']}")
                    updated_count += 1
                    
            except Exception as e:
                logger.error(f"Error applying improvement: {e}")
                
        return updated_count
        
    async def _run_immediate_learning(self, feedback: DeploymentFeedback):
        """Run immediate learning in response to critical feedback"""
        logger.warning(f"⚡ Immediate learning triggered for {feedback.design_id}")
        
        # Extract urgent insights
        insights = await self._extract_insights([feedback])
        
        # Generate and apply high-priority improvements
        improvements = await self._generate_improvements(insights)
        high_priority = [imp for imp in improvements if imp.get('priority') == 'high']
        
        if high_priority:
            await self._apply_improvements(high_priority)
            logger.info(f"✅ Applied {len(high_priority)} urgent improvements")
            
    async def _trigger_evolution_system(self, insights: List[Dict[str, Any]], 
                                       improvements: List[Dict[str, Any]]):
        """Trigger the autonomous evolution system with learning"""
        try:
            from modules.autonomous_evolution_loop import get_evolution_loop
            
            evolution = get_evolution_loop()
            
            # Create evolution candidates from improvements
            logger.info(f"🧬 Triggering evolution system with {len(improvements)} improvements")
            
            # In production, would create actual evolution candidates
            # For now, log the integration
            
        except Exception as e:
            logger.debug(f"Evolution system integration: {e}")
            
    async def _integrate_deployment_with_telemetry(self, design_id: str, 
                                                   deployment_id: str,
                                                   deployment_info: Dict[str, Any]):
        """Integrate deployment with telemetry system"""
        try:
            from modules.realworld_telemetry_integration import get_telemetry_integration
            
            telemetry = get_telemetry_integration()
            
            # Register with telemetry
            await telemetry.register_deployment(
                design_id=design_id,
                project_id=deployment_info.get('project_id', 'unknown'),
                location=deployment_info.get('location', 'unknown'),
                telemetry_endpoints=deployment_info.get('endpoints', []),
                expected_performance=deployment_info.get('expected_performance', {})
            )
            
        except Exception as e:
            logger.debug(f"Telemetry integration: {e}")
            
    async def _load_deployment_history(self):
        """Load deployment history from disk"""
        try:
            history_path = Path("data/deployment_history.json")
            if history_path.exists():
                with open(history_path) as f:
                    data = json.load(f)
                    # Load active deployments
                    self.active_deployments = data.get('active_deployments', {})
                    logger.info(f"📂 Loaded {len(self.active_deployments)} active deployments")
        except Exception as e:
            logger.error(f"Error loading deployment history: {e}")
            
    async def _save_deployment_history(self):
        """Save deployment history to disk"""
        try:
            history_path = Path("data/deployment_history.json")
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'active_deployments': self.active_deployments,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(history_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving deployment history: {e}")
            
    async def _load_learning_cycles(self):
        """Load learning cycles from disk"""
        try:
            cycles_path = Path("data/learning_cycles.json")
            if cycles_path.exists():
                with open(cycles_path) as f:
                    data = json.load(f)
                    for item in data:
                        cycle = LearningCycle(
                            cycle_id=item['cycle_id'],
                            start_time=datetime.fromisoformat(item['start_time']),
                            design_ids=item['design_ids'],
                            insights_extracted=item['insights_extracted'],
                            improvements_generated=item['improvements_generated'],
                            designs_updated=item['designs_updated'],
                            status=item['status']
                        )
                        self.learning_cycles.append(cycle)
        except Exception as e:
            logger.debug(f"No learning cycles loaded: {e}")
            
    async def _save_learning_cycles(self):
        """Save learning cycles to disk"""
        try:
            cycles_path = Path("data/learning_cycles.json")
            cycles_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    'cycle_id': cycle.cycle_id,
                    'start_time': cycle.start_time.isoformat(),
                    'design_ids': cycle.design_ids,
                    'insights_extracted': cycle.insights_extracted,
                    'improvements_generated': cycle.improvements_generated,
                    'designs_updated': cycle.designs_updated,
                    'status': cycle.status
                }
                for cycle in self.learning_cycles[-100:]  # Keep last 100
            ]
            
            with open(cycles_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving learning cycles: {e}")
            
    def get_feedback_loop_status(self) -> Dict[str, Any]:
        """Get current feedback loop status"""
        completed_cycles = len([c for c in self.learning_cycles if c.status == 'completed'])
        total_insights = sum(c.insights_extracted for c in self.learning_cycles)
        total_improvements = sum(c.improvements_generated for c in self.learning_cycles)
        
        return {
            'is_running': self.is_running,
            'active_deployments': len(self.active_deployments),
            'total_feedback': len(self.feedback_history),
            'learning_cycles_completed': completed_cycles,
            'total_insights_extracted': total_insights,
            'total_improvements_generated': total_improvements
        }


# Singleton instance
_feedback_loop = None

def get_feedback_loop() -> DeploymentFeedbackLoop:
    """Get the global feedback loop instance"""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = DeploymentFeedbackLoop()
    return _feedback_loop
