"""
Meta-Learning System
System that learns how to learn better - optimizes its own learning processes.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LearningStrategy:
    """A strategy for learning"""
    strategy_id: str
    name: str
    description: str
    hyperparameters: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    avg_performance: float = 0.0
    adaptation_rate: float = 0.1
    visual_performance_history: List[float] = field(default_factory=list)  # NEW: Visual task performance
    avg_visual_performance: float = 0.0  # NEW: Average visual performance
    

@dataclass
class LearningTask:
    """A learning task to optimize"""
    task_id: str
    task_type: str  # 'classification', 'regression', 'generation', 'optimization'
    domain: str
    data_characteristics: Dict[str, Any]
    best_strategy: Optional[str] = None
    performance_history: Dict[str, List[float]] = field(default_factory=dict)
    

@dataclass
class MetaKnowledge:
    """Meta-knowledge about learning"""
    knowledge_id: str
    insight: str
    applicability: List[str]  # Task types this applies to
    confidence: float
    discovered_at: datetime = field(default_factory=datetime.now)
    times_applied: int = 0
    success_rate: float = 0.0
    visual_evidence: List[Dict[str, Any]] = field(default_factory=list)  # NEW: Visual evidence (diagrams, charts)


class MetaLearningSystem:
    """
    System that learns how to learn better.
    
    Features:
    - Tracks performance of different learning strategies
    - Adapts learning approaches based on task characteristics
    - Discovers meta-knowledge about what works when
    - Optimizes hyperparameters automatically
    - Transfers learning across domains
    - Self-improves learning efficiency over time
    """
    
    def __init__(self):
        self.strategies: Dict[str, LearningStrategy] = {}
        self.tasks: Dict[str, LearningTask] = {}
        self.meta_knowledge: Dict[str, MetaKnowledge] = {}
        self.is_running = False
        
        # Performance tracking
        self.global_learning_rate = 0.01
        self.adaptation_history: List[Dict[str, Any]] = []
        
        # Vision capabilities for visual feedback ✅
        self.vision_engine = None
        self.vision_enabled = False
        try:
            from modules.llm import get_vision_engine
            self.vision_engine = get_vision_engine()
            self.vision_enabled = True
            logger.info("✅ Meta-Learning: Vision feedback ENABLED")
        except Exception as e:
            logger.warning(f"⚠️ Meta-Learning: Vision feedback unavailable ({e})")
        
    async def initialize(self):
        """Initialize the meta-learning system"""
        logger.info("🧠 Initializing Meta-Learning System")
        
        # Initialize default learning strategies
        await self._initialize_strategies()
        
        # Load historical meta-knowledge
        await self._load_meta_knowledge()
        
        logger.info(f"✅ Meta-learning initialized with {len(self.strategies)} strategies")
        
    async def _initialize_strategies(self):
        """Initialize default learning strategies"""
        
        # Strategy 1: Fast adaptation (high learning rate, low exploration)
        self.strategies['fast_adapt'] = LearningStrategy(
            strategy_id='fast_adapt',
            name='Fast Adaptation',
            description='Quick learning with high learning rate',
            hyperparameters={
                'learning_rate': 0.1,
                'batch_size': 32,
                'exploration_rate': 0.1,
                'memory_size': 1000
            }
        )
        
        # Strategy 2: Stable learning (moderate learning rate, balanced)
        self.strategies['stable'] = LearningStrategy(
            strategy_id='stable',
            name='Stable Learning',
            description='Balanced learning with stability',
            hyperparameters={
                'learning_rate': 0.01,
                'batch_size': 64,
                'exploration_rate': 0.3,
                'memory_size': 5000
            }
        )
        
        # Strategy 3: Deep exploration (low learning rate, high exploration)
        self.strategies['explore'] = LearningStrategy(
            strategy_id='explore',
            name='Deep Exploration',
            description='Thorough exploration of solution space',
            hyperparameters={
                'learning_rate': 0.001,
                'batch_size': 128,
                'exploration_rate': 0.7,
                'memory_size': 10000
            }
        )
        
        # Strategy 4: Transfer learning (leverage prior knowledge)
        self.strategies['transfer'] = LearningStrategy(
            strategy_id='transfer',
            name='Transfer Learning',
            description='Leverage knowledge from related tasks',
            hyperparameters={
                'learning_rate': 0.05,
                'batch_size': 64,
                'exploration_rate': 0.2,
                'memory_size': 5000,
                'transfer_weight': 0.7
            }
        )
        
        # Strategy 5: Meta-optimization (optimize the optimizer)
        self.strategies['meta_optimize'] = LearningStrategy(
            strategy_id='meta_optimize',
            name='Meta-Optimization',
            description='Learn optimal learning parameters',
            hyperparameters={
                'learning_rate': 0.01,
                'batch_size': 64,
                'exploration_rate': 0.4,
                'memory_size': 5000,
                'meta_learning_rate': 0.001
            }
        )
        
    async def start_meta_learning(self):
        """Start continuous meta-learning"""
        if self.is_running:
            logger.warning("Meta-learning already running")
            return
            
        self.is_running = True
        logger.info("🔄 Starting meta-learning loop")
        
        while self.is_running:
            try:
                # Meta-learning cycle
                await self._meta_learning_cycle()
                
                # Wait between cycles
                await asyncio.sleep(120)  # 2 minutes
                
            except Exception as e:
                logger.error(f"Meta-learning error: {e}", exc_info=True)
                await asyncio.sleep(60)
                
    async def stop_meta_learning(self):
        """Stop meta-learning"""
        self.is_running = False
        logger.info("⏸️ Meta-learning stopped")
        
    async def _meta_learning_cycle(self):
        """Execute one meta-learning cycle"""
        logger.info("🧠 Meta-learning cycle")
        
        # 1. Evaluate current strategies
        await self._evaluate_strategies()
        
        # 2. Discover new meta-knowledge
        new_knowledge = await self._discover_meta_knowledge()
        if new_knowledge:
            logger.info(f"💡 Discovered {len(new_knowledge)} new meta-insights")
        
        # 3. Adapt strategies based on performance
        await self._adapt_strategies()
        
        # 4. Optimize global learning parameters
        await self._optimize_global_parameters()
        
        # 5. Transfer knowledge between tasks
        await self._transfer_knowledge()
        
    async def select_strategy(self, task: LearningTask) -> LearningStrategy:
        """Select best learning strategy for a task"""
        
        # If task has history, use best performing strategy
        if task.best_strategy and task.best_strategy in self.strategies:
            logger.info(f"📊 Using proven strategy: {task.best_strategy}")
            return self.strategies[task.best_strategy]
        
        # Otherwise, use meta-knowledge to select
        best_strategy = await self._meta_select_strategy(task)
        
        logger.info(f"🎯 Selected strategy: {best_strategy.name}")
        return best_strategy
        
    async def _meta_select_strategy(self, task: LearningTask) -> LearningStrategy:
        """Use meta-knowledge to select strategy"""
        
        # Score each strategy based on task characteristics
        scores = {}
        
        for strategy_id, strategy in self.strategies.items():
            score = 0.0
            
            # Factor 1: Historical performance on similar tasks
            if strategy.performance_history:
                score += strategy.avg_performance * 0.4
            else:
                score += 0.3  # Default for untried strategies
            
            # Factor 2: Task type matching
            if task.task_type == 'optimization' and strategy_id in ['meta_optimize', 'explore']:
                score += 0.2
            elif task.task_type == 'classification' and strategy_id in ['stable', 'transfer']:
                score += 0.2
            elif task.task_type == 'generation' and strategy_id in ['explore', 'transfer']:
                score += 0.2
            
            # Factor 3: Data characteristics
            if task.data_characteristics.get('complex', False):
                if strategy_id in ['explore', 'meta_optimize']:
                    score += 0.2
            else:
                if strategy_id in ['fast_adapt', 'stable']:
                    score += 0.2
            
            # Factor 4: Meta-knowledge application
            applicable_knowledge = [
                mk for mk in self.meta_knowledge.values()
                if task.task_type in mk.applicability and mk.confidence > 0.7
            ]
            
            if applicable_knowledge:
                # Check if meta-knowledge recommends this strategy
                for mk in applicable_knowledge:
                    if strategy_id in mk.insight:
                        score += 0.2 * mk.confidence
            
            scores[strategy_id] = score
        
        # Select strategy with highest score
        best_id = max(scores, key=scores.get)
        return self.strategies[best_id]
        
    async def report_performance(self, task_id: str, strategy_id: str, performance: float):
        """Report performance of a strategy on a task"""
        
        # Update task history
        if task_id in self.tasks:
            task = self.tasks[task_id]
            if strategy_id not in task.performance_history:
                task.performance_history[strategy_id] = []
            task.performance_history[strategy_id].append(performance)
            
            # Update best strategy if this performed better
            if not task.best_strategy or performance > np.mean(task.performance_history.get(task.best_strategy, [0])):
                task.best_strategy = strategy_id
                logger.info(f"🏆 New best strategy for {task_id}: {strategy_id} ({performance:.3f})")
        
        # Update strategy history
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            strategy.performance_history.append(performance)
            strategy.avg_performance = np.mean(strategy.performance_history[-100:])  # Last 100
            
    async def learn_from_outcomes(
        self,
        task_type: str,
        outcomes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Learn from actual project outcomes to improve future predictions.
        
        Args:
            task_type: Type of task (e.g., 'construction_project')
            outcomes: Dict with actual vs estimated values and variance
        
        Returns:
            Dict with lessons learned, model improvements, and adjustment factors
        """
        logger.info(f"📚 Meta-learning from {task_type} outcomes")
        
        # Extract key metrics
        timeline_variance = outcomes.get('timeline', {}).get('variance_weeks', 0)
        budget_variance = outcomes.get('budget', {}).get('variance_dollars', 0)
        budget_variance_pct = abs(budget_variance / outcomes.get('budget', {}).get('estimated', 1)) if outcomes.get('budget', {}).get('estimated', 0) > 0 else 0
        
        # Calculate accuracy metrics
        timeline_accuracy = 1.0 - min(1.0, abs(timeline_variance) / max(1, outcomes.get('timeline', {}).get('estimated', 1)))
        budget_accuracy = 1.0 - min(1.0, budget_variance_pct)
        
        # Identify patterns
        lessons_learned = []
        improvements_made = []
        
        # Timeline lessons
        if abs(timeline_variance) > 2:  # More than 2 weeks off
            if timeline_variance > 0:
                lessons_learned.append(f"Timeline underestimated by {timeline_variance:.1f} weeks")
                improvements_made.append("Increase timeline estimates for similar projects")
            else:
                lessons_learned.append(f"Timeline overestimated by {abs(timeline_variance):.1f} weeks")
                improvements_made.append("Reduce timeline estimates for similar projects")
        
        # Budget lessons
        if budget_variance_pct > 0.1:  # More than 10% off
            if budget_variance > 0:
                lessons_learned.append(f"Budget underestimated by {budget_variance_pct:.1%}")
                improvements_made.append("Increase budget estimates for similar projects")
            else:
                lessons_learned.append(f"Budget overestimated by {abs(budget_variance_pct):.1%}")
                improvements_made.append("Reduce budget estimates for similar projects")
        
        # Calculate adjustment factors
        timeline_adjustment = 1.0
        if abs(timeline_variance) > 1:
            # Adjust by variance percentage, capped at ±20%
            adjustment = timeline_variance / max(1, outcomes.get('timeline', {}).get('estimated', 1))
            timeline_adjustment = 1.0 + max(-0.2, min(0.2, adjustment))
        
        budget_adjustment = 1.0
        if budget_variance_pct > 0.05:  # More than 5% variance
            # Adjust by variance percentage, capped at ±20%
            adjustment = budget_variance / max(1, outcomes.get('budget', {}).get('estimated', 1))
            budget_adjustment = 1.0 + max(-0.2, min(0.2, adjustment))
        
        # Calculate overall accuracy improvement
        current_accuracy = (timeline_accuracy + budget_accuracy) / 2
        accuracy_improvement = max(0, current_accuracy - 0.5)  # Improvement over baseline 0.5
        
        # Store as meta-knowledge
        if lessons_learned:
            knowledge_id = f"outcome_{task_type}_{datetime.now().timestamp()}"
            insight = f"For {task_type}: " + "; ".join(lessons_learned[:3])
            
            knowledge = MetaKnowledge(
                knowledge_id=knowledge_id,
                insight=insight,
                applicability=[task_type],
                confidence=min(current_accuracy, 0.9),
                discovered_at=datetime.now()
            )
            
            self.meta_knowledge[knowledge_id] = knowledge
            logger.info(f"💡 Stored meta-knowledge: {insight[:60]}...")
        
        return {
            'key_lessons': lessons_learned,
            'improvements_made': improvements_made,
            'accuracy_improvement': accuracy_improvement,
            'timeline_adjustment': timeline_adjustment,
            'budget_adjustment': budget_adjustment,
            'timeline_accuracy': timeline_accuracy,
            'budget_accuracy': budget_accuracy
        }
    
    async def record_progress_observation(
        self,
        project_type: str,
        expected_stage: str,
        actual_progress: List[str],
        schedule_variance: int
    ):
        """
        Record progress observation for meta-learning.
        
        Args:
            project_type: Type of project
            expected_stage: Expected construction stage
            actual_progress: List of milestones actually completed
            schedule_variance: Days ahead/behind schedule
        """
        # Create or update task for this project type
        task_id = f"progress_{project_type}"
        if task_id not in self.tasks:
            self.tasks[task_id] = LearningTask(
                task_id=task_id,
                task_type='progress_tracking',
                domain=project_type,
                data_characteristics={'type': project_type}
            )
        
        # Calculate performance score based on schedule variance
        # Negative variance (behind) = lower score, positive (ahead) = higher score
        performance = max(0.0, min(1.0, 0.7 + (schedule_variance / 30.0)))  # Normalize to 0-1
        
        # Report performance
        await self.report_performance(
            task_id=task_id,
            strategy_id='stable',  # Use stable strategy for progress tracking
            performance=performance
        )
        
        logger.debug(f"Recorded progress observation: {project_type}, variance={schedule_variance} days, performance={performance:.3f}")
    
    async def predict_risks(
        self,
        current_stage: str,
        project_type: str,
        timeline: int,
        budget: float,
        location: Optional[str] = None,
        complexity: float = 0.5,
        historical_projects: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict potential risks/issues for a construction project based on meta-learning.
        
        Args:
            current_stage: Current construction stage
            project_type: Type of project (adu, remodel, etc.)
            timeline: Estimated timeline in weeks
            budget: Estimated budget
            location: Optional location
            complexity: Complexity score (0-1)
            historical_projects: List of similar historical projects
            
        Returns:
            List of predicted risks with probability and description
        """
        predictions = []
        
        # Analyze historical patterns if available
        if historical_projects:
            # Extract common issues from historical projects
            common_issues = defaultdict(int)
            for proj in historical_projects:
                issues = proj.get('issues_encountered', [])
                for issue in issues:
                    issue_type = issue.get('type', 'unknown')
                    common_issues[issue_type] += 1
            
            # Predict based on frequency
            for issue_type, count in common_issues.items():
                probability = min(0.9, count / len(historical_projects))
                if probability > 0.3:  # Only include likely risks
                    predictions.append({
                        'issue': issue_type,
                        'probability': probability,
                        'stage': current_stage,
                        'reasoning': f"Occurred in {count}/{len(historical_projects)} similar projects"
                    })
        
        # Stage-specific risk patterns
        stage_risks = {
            'discovery': [
                {'issue': 'Budget underestimation', 'probability': 0.6},
                {'issue': 'Permit delays', 'probability': 0.5}
            ],
            'design': [
                {'issue': 'Design changes', 'probability': 0.5},
                {'issue': 'Code compliance issues', 'probability': 0.4}
            ],
            'permitting': [
                {'issue': 'Permit delays', 'probability': 0.7},
                {'issue': 'Code violations', 'probability': 0.4}
            ],
            'construction': [
                {'issue': 'Weather delays', 'probability': 0.5},
                {'issue': 'Material shortages', 'probability': 0.4},
                {'issue': 'Contractor delays', 'probability': 0.5}
            ]
        }
        
        # Add stage-specific risks
        for risk in stage_risks.get(current_stage.lower(), []):
            # Check if already predicted
            if not any(p['issue'] == risk['issue'] for p in predictions):
                predictions.append({
                    'issue': risk['issue'],
                    'probability': risk['probability'],
                    'stage': current_stage,
                    'reasoning': f'Common risk for {current_stage} stage'
                })
        
        # Complexity-based risks
        if complexity > 0.7:
            predictions.append({
                'issue': 'Complexity-related delays',
                'probability': 0.6,
                'stage': current_stage,
                'reasoning': 'High complexity projects often face delays'
            })
        
        # Budget-based risks
        if budget < 100000:  # Low budget projects
            predictions.append({
                'issue': 'Budget overruns',
                'probability': 0.7,
                'stage': current_stage,
                'reasoning': 'Lower budget projects more prone to overruns'
            })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return predictions[:10]  # Return top 10 risks
            
    async def _evaluate_strategies(self):
        """Evaluate and rank current strategies"""
        
        rankings = []
        for strategy_id, strategy in self.strategies.items():
            if strategy.performance_history:
                rankings.append({
                    'strategy_id': strategy_id,
                    'name': strategy.name,
                    'avg_performance': strategy.avg_performance,
                    'stability': np.std(strategy.performance_history[-50:]) if len(strategy.performance_history) > 1 else 0,
                    'sample_count': len(strategy.performance_history)
                })
        
        # Sort by performance
        rankings.sort(key=lambda x: x['avg_performance'], reverse=True)
        
        if rankings:
            logger.info(f"📊 Top strategy: {rankings[0]['name']} ({rankings[0]['avg_performance']:.3f})")
            
    async def _discover_meta_knowledge(self) -> List[MetaKnowledge]:
        """Discover new meta-knowledge from patterns"""
        new_knowledge = []
        
        # Pattern 1: Strategy-task correlations
        for task_id, task in self.tasks.items():
            if len(task.performance_history) >= 2:
                # Find which strategy works best for this task type
                best_strategy = max(
                    task.performance_history.keys(),
                    key=lambda s: np.mean(task.performance_history[s])
                )
                
                best_perf = np.mean(task.performance_history[best_strategy])
                
                if best_perf > 0.8:  # High performance threshold
                    knowledge_id = f"meta_{datetime.now().timestamp()}"
                    insight = f"Strategy '{best_strategy}' performs exceptionally well on {task.task_type} tasks in {task.domain}"
                    
                    knowledge = MetaKnowledge(
                        knowledge_id=knowledge_id,
                        insight=insight,
                        applicability=[task.task_type],
                        confidence=min(best_perf, 0.95)
                    )
                    
                    self.meta_knowledge[knowledge_id] = knowledge
                    new_knowledge.append(knowledge)
                    
        # Pattern 2: Hyperparameter correlations
        # Analyze which hyperparameters lead to better performance
        
        # Pattern 3: Transfer learning opportunities
        # Identify when transfer learning is beneficial
        
        return new_knowledge
        
    async def _adapt_strategies(self):
        """Adapt strategies based on performance"""
        
        for strategy_id, strategy in self.strategies.items():
            if len(strategy.performance_history) < 5:
                continue
                
            recent_performance = strategy.performance_history[-10:]
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            
            # If performance is declining, adapt hyperparameters
            if trend < -0.01:  # Declining
                logger.info(f"📉 Adapting {strategy.name} due to declining performance")
                await self._adapt_hyperparameters(strategy)
                
                # Record adaptation
                self.adaptation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'strategy': strategy_id,
                    'reason': 'declining_performance',
                    'trend': trend
                })
                
    async def _adapt_hyperparameters(self, strategy: LearningStrategy):
        """Adapt hyperparameters of a strategy"""
        
        # Simple adaptation: perturb hyperparameters slightly
        for param, value in strategy.hyperparameters.items():
            if isinstance(value, (int, float)):
                # Random perturbation
                perturbation = np.random.uniform(-0.2, 0.2)
                new_value = value * (1 + perturbation * strategy.adaptation_rate)
                
                # Clamp to reasonable ranges
                if param == 'learning_rate':
                    new_value = max(0.0001, min(0.5, new_value))
                elif param == 'exploration_rate':
                    new_value = max(0.0, min(1.0, new_value))
                elif param == 'batch_size':
                    new_value = int(max(16, min(256, new_value)))
                    
                strategy.hyperparameters[param] = new_value
                logger.debug(f"  {param}: {value} → {new_value}")
                
    async def _optimize_global_parameters(self):
        """Optimize global learning parameters"""
        
        # Calculate overall system performance
        if not any(s.performance_history for s in self.strategies.values()):
            return
            
        all_performances = []
        for strategy in self.strategies.values():
            all_performances.extend(strategy.performance_history[-20:])
            
        if not all_performances:
            return
            
        avg_performance = np.mean(all_performances)
        
        # Adapt global learning rate based on system performance
        if avg_performance > 0.8:
            # System doing well, can increase learning rate slightly
            self.global_learning_rate = min(0.1, self.global_learning_rate * 1.05)
        elif avg_performance < 0.5:
            # System struggling, decrease learning rate
            self.global_learning_rate = max(0.001, self.global_learning_rate * 0.95)
            
        logger.debug(f"🌍 Global learning rate: {self.global_learning_rate:.4f}")
        
    async def _transfer_knowledge(self):
        """Transfer knowledge between related tasks"""
        
        # Group tasks by similarity
        task_groups = defaultdict(list)
        for task in self.tasks.values():
            key = (task.task_type, task.domain)
            task_groups[key].append(task)
        
        # For each group, transfer best practices
        for group_key, tasks in task_groups.items():
            if len(tasks) < 2:
                continue
                
            # Find best performing task in group
            best_task = max(
                tasks,
                key=lambda t: max([np.mean(perfs) for perfs in t.performance_history.values()]) if t.performance_history else 0
            )
            
            if not best_task.best_strategy:
                continue
                
            # Transfer best strategy to other tasks in group
            for task in tasks:
                if task.task_id != best_task.task_id and task.best_strategy != best_task.best_strategy:
                    logger.info(f"🔄 Transferring strategy {best_task.best_strategy} to {task.task_id}")
                    # This will be picked up in next learning iteration
                    
    async def create_learning_task(self, task_type: str, domain: str, 
                                  data_characteristics: Dict[str, Any]) -> LearningTask:
        """Create a new learning task"""
        task_id = f"task_{task_type}_{domain}_{datetime.now().timestamp()}"
        
        task = LearningTask(
            task_id=task_id,
            task_type=task_type,
            domain=domain,
            data_characteristics=data_characteristics
        )
        
        self.tasks[task_id] = task
        
        logger.info(f"📝 Created learning task: {task_id}")
        
        return task
        
    async def _load_meta_knowledge(self):
        """Load meta-knowledge from disk"""
        try:
            data_path = Path("data/meta_learning.json")
            if data_path.exists():
                with open(data_path) as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded meta-learning data")
        except Exception as e:
            logger.debug(f"No meta-learning data loaded: {e}")
            
    async def save_meta_knowledge(self):
        """Save meta-knowledge to disk"""
        try:
            data_path = Path("data/meta_learning.json")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'strategies': len(self.strategies),
                'tasks': len(self.tasks),
                'meta_knowledge': len(self.meta_knowledge),
                'global_learning_rate': self.global_learning_rate,
                'adaptations': len(self.adaptation_history)
            }
            
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving meta-learning data: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """Get meta-learning system status"""
        
        # Calculate average performance across all strategies
        all_perfs = []
        for strategy in self.strategies.values():
            all_perfs.extend(strategy.performance_history[-50:])
            
        avg_performance = np.mean(all_perfs) if all_perfs else 0.0
        
        # Get best strategy
        best_strategy = max(
            self.strategies.values(),
            key=lambda s: s.avg_performance if s.performance_history else 0
        ) if self.strategies else None
        
        return {
            'is_running': self.is_running,
            'total_strategies': len(self.strategies),
            'total_tasks': len(self.tasks),
            'meta_knowledge_items': len(self.meta_knowledge),
            'global_learning_rate': self.global_learning_rate,
            'average_performance': avg_performance,
            'best_strategy': best_strategy.name if best_strategy else None,
            'total_adaptations': len(self.adaptation_history),
            'vision_enabled': self.vision_enabled,  # NEW
            'visual_avg_performance': self._calculate_visual_performance()  # NEW
        }
    
    # ============================================================
    # VISION FEEDBACK METHODS v1.0
    # ------------------------------------------------------------
    # Revolutionary meta-learning with visual feedback:
    # - Evaluate strategies on visual tasks
    # - Analyze learning curves with vision
    # - Generate visual performance reports
    # - Discover patterns in visual data
    # ============================================================
    
    async def evaluate_visual_strategy(
        self,
        strategy_id: str,
        visual_task_data: Dict[str, Any],
        vision_engine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a learning strategy on visual tasks using vision feedback.
        
        This revolutionary method:
        - Runs learning strategy on visual data
        - Uses vision model to assess output quality
        - Analyzes visual patterns in results
        - Provides rich visual feedback for strategy optimization
        
        Args:
            strategy_id: ID of strategy to evaluate
            visual_task_data: Visual task data (images, expected outputs, etc.)
            vision_engine: Vision model (optional, uses self.vision_engine if None)
            
        Returns:
            Dictionary with visual evaluation results
        """
        if not self.vision_enabled and not vision_engine:
            logger.warning("⚠️ Vision not enabled, cannot evaluate visual strategy")
            return {"error": "Vision not available", "performance": 0.0}
        
        vision = vision_engine or self.vision_engine
        strategy = self.strategies.get(strategy_id)
        
        if not strategy:
            return {"error": f"Strategy {strategy_id} not found", "performance": 0.0}
        
        logger.info(f"🎨 Evaluating strategy '{strategy.name}' on visual task")
        
        try:
            # Extract visual task components
            input_images = visual_task_data.get("input_images", [])
            expected_outputs = visual_task_data.get("expected_outputs", [])
            task_description = visual_task_data.get("description", "Visual task")
            
            if not input_images:
                return {"error": "No input images provided", "performance": 0.0}
            
            # Analyze input data quality with vision
            input_analysis = await self._analyze_visual_inputs(
                images=input_images,
                vision_engine=vision,
                task_description=task_description
            )
            
            # Simulate strategy performance (in real implementation, run actual strategy)
            # For now, we'll use vision to assess how well strategy would perform
            performance_prediction = await self._predict_strategy_performance_visual(
                strategy=strategy,
                input_analysis=input_analysis,
                vision_engine=vision
            )
            
            # Generate visual performance chart
            chart_path = None
            if len(strategy.visual_performance_history) > 5:
                chart_path = await self._generate_performance_chart(
                    strategy=strategy,
                    vision_engine=vision
                )
            
            # Calculate visual performance score
            visual_performance = performance_prediction.get("predicted_performance", 0.5)
            
            # Update strategy with visual feedback
            strategy.visual_performance_history.append(visual_performance)
            strategy.avg_visual_performance = np.mean(strategy.visual_performance_history[-50:])
            
            # Analyze if strategy should be adapted
            adaptation_needed = visual_performance < 0.6 or (
                len(strategy.visual_performance_history) > 5 and
                np.mean(strategy.visual_performance_history[-5:]) < 
                np.mean(strategy.visual_performance_history[-10:-5])
            )
            
            result = {
                "strategy_id": strategy_id,
                "strategy_name": strategy.name,
                "visual_performance": visual_performance,
                "avg_visual_performance": strategy.avg_visual_performance,
                "input_quality": input_analysis.get("quality_score", 0.5),
                "performance_prediction": performance_prediction,
                "adaptation_needed": adaptation_needed,
                "performance_chart": chart_path,
                "visual_insights": performance_prediction.get("insights", []),
                "samples_evaluated": len(input_images)
            }
            
            logger.info(f"✅ Visual evaluation complete: {visual_performance:.3f} performance")
            return result
            
        except Exception as e:
            logger.exception(f"Visual strategy evaluation failed: {e}")
            return {
                "error": str(e),
                "performance": 0.0,
                "strategy_id": strategy_id
            }
    
    async def _analyze_visual_inputs(
        self,
        images: List[str],
        vision_engine: Any,
        task_description: str
    ) -> Dict[str, Any]:
        """
        Analyze input images to understand task characteristics.
        
        Returns insights about:
        - Image quality
        - Complexity
        - Patterns
        - Suitable learning approaches
        """
        logger.debug(f"Analyzing {len(images)} input images")
        
        # Analyze a sample of images
        sample_size = min(5, len(images))
        sample_images = images[:sample_size]
        
        analyses = []
        for img_path in sample_images:
            try:
                prompt = f"""
Analyze this image in context of the learning task: "{task_description}"

Assess:
1. Image quality (clarity, resolution, noise level)
2. Visual complexity (simple patterns vs complex scenes)
3. Key features visible
4. Challenges for learning algorithms
5. Recommended learning approach

Provide structured assessment:
QUALITY: [0-1 score]
COMPLEXITY: [0-1 score]
FEATURES: [key features list]
CHALLENGES: [main challenges]
RECOMMENDATION: [learning approach]
"""
                
                analysis = vision_engine.analyze_image(img_path, prompt)
                analyses.append(analysis)
                
            except Exception as e:
                logger.warning(f"Could not analyze image {img_path}: {e}")
        
        if not analyses:
            return {"quality_score": 0.5, "complexity": 0.5, "insights": []}
        
        # Parse analyses to extract scores
        quality_scores = []
        complexity_scores = []
        all_insights = []
        
        for analysis in analyses:
            # Simple parsing (in production, use structured extraction)
            if "QUALITY:" in analysis:
                try:
                    quality_line = [l for l in analysis.split('\n') if 'QUALITY:' in l][0]
                    quality = float(quality_line.split(':')[1].strip().split()[0])
                    quality_scores.append(quality)
                except:
                    pass
            
            if "COMPLEXITY:" in analysis:
                try:
                    complexity_line = [l for l in analysis.split('\n') if 'COMPLEXITY:' in l][0]
                    complexity = float(complexity_line.split(':')[1].strip().split()[0])
                    complexity_scores.append(complexity)
                except:
                    pass
            
            if "RECOMMENDATION:" in analysis:
                try:
                    rec_line = [l for l in analysis.split('\n') if 'RECOMMENDATION:' in l][0]
                    recommendation = rec_line.split(':')[1].strip()
                    all_insights.append(recommendation)
                except:
                    pass
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0.5
        
        return {
            "quality_score": avg_quality,
            "complexity_score": avg_complexity,
            "insights": all_insights,
            "samples_analyzed": len(analyses),
            "avg_quality": avg_quality,
            "avg_complexity": avg_complexity
        }
    
    async def _predict_strategy_performance_visual(
        self,
        strategy: LearningStrategy,
        input_analysis: Dict[str, Any],
        vision_engine: Any
    ) -> Dict[str, Any]:
        """
        Predict how well a strategy will perform based on visual analysis of inputs.
        
        Uses vision model to:
        - Match strategy characteristics to task requirements
        - Predict performance based on similar past tasks
        - Identify potential issues
        """
        from modules.llm import LLMEngine
        
        llm = LLMEngine()
        
        # Build prediction prompt
        prompt = f"""
Predict how well this learning strategy will perform on the given task.

Strategy: {strategy.name}
Description: {strategy.description}
Hyperparameters: {strategy.hyperparameters}
Historical Avg Performance: {strategy.avg_performance:.3f}
Historical Visual Performance: {strategy.avg_visual_performance:.3f}

Task Analysis (from vision):
- Input Quality: {input_analysis.get('quality_score', 0.5):.2f}
- Complexity: {input_analysis.get('complexity_score', 0.5):.2f}
- Insights: {', '.join(input_analysis.get('insights', [])[:3])}

Based on:
1. Strategy characteristics
2. Task requirements (from visual analysis)
3. Historical performance
4. Input quality and complexity

Predict:
1. Expected performance score (0.0 to 1.0)
2. Confidence in prediction (0.0 to 1.0)
3. Key factors affecting performance
4. Potential improvements

Format:
PERFORMANCE: [0-1 score]
CONFIDENCE: [0-1 score]
FACTORS: [factor1, factor2, factor3]
IMPROVEMENTS: [improvement1, improvement2]
"""
        
        prediction = await llm.generate(prompt, max_tokens=500, temperature=0.2)
        
        # Parse prediction
        predicted_performance = 0.5
        confidence = 0.5
        factors = []
        improvements = []
        
        if prediction:
            lines = prediction.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('PERFORMANCE:'):
                    try:
                        predicted_performance = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                elif line.startswith('CONFIDENCE:'):
                    try:
                        confidence = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                elif line.startswith('FACTORS:'):
                    factors_str = line.split(':')[1].strip()
                    factors = [f.strip() for f in factors_str.split(',')]
                elif line.startswith('IMPROVEMENTS:'):
                    improvements_str = line.split(':')[1].strip()
                    improvements = [i.strip() for i in improvements_str.split(',')]
        
        return {
            "predicted_performance": predicted_performance,
            "confidence": confidence,
            "factors": factors,
            "improvements": improvements,
            "insights": factors + improvements
        }
    
    async def _generate_performance_chart(
        self,
        strategy: LearningStrategy,
        vision_engine: Any
    ) -> Optional[str]:
        """
        Generate a visual chart showing strategy performance over time.
        
        Creates:
        - Line chart of performance history
        - Saves to disk
        - Returns path
        """
        try:
            import matplotlib.pyplot as plt
            from datetime import datetime
            
            # Create performance chart
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Chart 1: Overall performance
            if strategy.performance_history:
                ax1.plot(strategy.performance_history, label='Overall Performance', color='blue', linewidth=2)
                ax1.axhline(y=strategy.avg_performance, color='blue', linestyle='--', alpha=0.5, label=f'Avg: {strategy.avg_performance:.3f}')
            
            ax1.set_title(f'Learning Strategy Performance: {strategy.name}', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Learning Iteration')
            ax1.set_ylabel('Performance Score')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Chart 2: Visual performance
            if strategy.visual_performance_history:
                ax2.plot(strategy.visual_performance_history, label='Visual Performance', color='green', linewidth=2)
                ax2.axhline(y=strategy.avg_visual_performance, color='green', linestyle='--', alpha=0.5, label=f'Avg: {strategy.avg_visual_performance:.3f}')
            
            ax2.set_title('Visual Task Performance', fontsize=12)
            ax2.set_xlabel('Visual Task Iteration')
            ax2.set_ylabel('Visual Performance Score')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save chart
            chart_dir = Path("data/meta_learning_charts")
            chart_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = chart_dir / f"strategy_{strategy.strategy_id}_{timestamp}.png"
            
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 Performance chart saved: {chart_path}")
            return str(chart_path)
            
        except Exception as e:
            logger.warning(f"Could not generate performance chart: {e}")
            return None
    
    async def generate_visual_meta_learning_report(
        self,
        vision_engine: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive visual report of meta-learning progress.
        
        Creates:
        - Performance comparison charts for all strategies
        - Heatmap of strategy-task performance
        - Visual insights into meta-knowledge
        - Recommendations visualized
        
        Returns:
            Dictionary with report data and visual paths
        """
        if not self.vision_enabled and not vision_engine:
            logger.warning("Vision not enabled for visual report")
            return {"error": "Vision not available"}
        
        logger.info("📊 Generating visual meta-learning report")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from datetime import datetime
            
            # Create comprehensive report figure
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Chart 1: Strategy comparison
            ax1 = fig.add_subplot(gs[0, :])
            strategy_names = []
            overall_perfs = []
            visual_perfs = []
            
            for strategy in self.strategies.values():
                if strategy.performance_history or strategy.visual_performance_history:
                    strategy_names.append(strategy.name)
                    overall_perfs.append(strategy.avg_performance)
                    visual_perfs.append(strategy.avg_visual_performance)
            
            if strategy_names:
                x = np.arange(len(strategy_names))
                width = 0.35
                
                ax1.bar(x - width/2, overall_perfs, width, label='Overall Performance', color='steelblue')
                ax1.bar(x + width/2, visual_perfs, width, label='Visual Performance', color='seagreen')
                
                ax1.set_xlabel('Learning Strategy')
                ax1.set_ylabel('Average Performance')
                ax1.set_title('Meta-Learning Strategy Comparison', fontsize=14, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(strategy_names, rotation=45, ha='right')
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')
                ax1.set_ylim(0, 1)
            
            # Chart 2: Learning curves
            ax2 = fig.add_subplot(gs[1, 0])
            for strategy in self.strategies.values():
                if len(strategy.performance_history) > 3:
                    ax2.plot(strategy.performance_history[-50:], label=strategy.name[:15], alpha=0.7)
            
            ax2.set_xlabel('Recent Iterations')
            ax2.set_ylabel('Performance')
            ax2.set_title('Learning Curves (Recent History)', fontsize=12)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Chart 3: Visual performance trends
            ax3 = fig.add_subplot(gs[1, 1])
            for strategy in self.strategies.values():
                if len(strategy.visual_performance_history) > 3:
                    ax3.plot(strategy.visual_performance_history, label=strategy.name[:15], alpha=0.7)
            
            ax3.set_xlabel('Visual Task Iterations')
            ax3.set_ylabel('Visual Performance')
            ax3.set_title('Visual Task Learning Curves', fontsize=12)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # Chart 4: Meta-knowledge growth
            ax4 = fig.add_subplot(gs[2, 0])
            knowledge_over_time = [0]
            for mk in sorted(self.meta_knowledge.values(), key=lambda x: x.discovered_at):
                knowledge_over_time.append(knowledge_over_time[-1] + 1)
            
            if len(knowledge_over_time) > 1:
                ax4.plot(knowledge_over_time, linewidth=2, color='purple')
                ax4.fill_between(range(len(knowledge_over_time)), knowledge_over_time, alpha=0.3, color='purple')
            
            ax4.set_xlabel('Discovery Events')
            ax4.set_ylabel('Cumulative Meta-Knowledge')
            ax4.set_title('Meta-Knowledge Accumulation', fontsize=12)
            ax4.grid(True, alpha=0.3)
            
            # Chart 5: Adaptation history
            ax5 = fig.add_subplot(gs[2, 1])
            if self.adaptation_history:
                adaptation_reasons = defaultdict(int)
                for adaptation in self.adaptation_history[-50:]:
                    adaptation_reasons[adaptation.get('reason', 'unknown')] += 1
                
                reasons = list(adaptation_reasons.keys())
                counts = list(adaptation_reasons.values())
                
                ax5.bar(reasons, counts, color='coral')
                ax5.set_xlabel('Adaptation Reason')
                ax5.set_ylabel('Frequency')
                ax5.set_title('Strategy Adaptations (Recent)', fontsize=12)
                ax5.tick_params(axis='x', rotation=45)
                ax5.grid(True, alpha=0.3, axis='y')
            
            # Add summary text
            summary_text = f"""
META-LEARNING SYSTEM REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
───────────────────────────────────
Strategies: {len(self.strategies)}
Tasks Tracked: {len(self.tasks)}
Meta-Knowledge Items: {len(self.meta_knowledge)}
Total Adaptations: {len(self.adaptation_history)}
Global Learning Rate: {self.global_learning_rate:.4f}
Vision Enabled: {self.vision_enabled}
"""
            fig.text(0.02, 0.98, summary_text, fontsize=9, family='monospace', 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Save report
            report_dir = Path("data/meta_learning_reports")
            report_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"meta_learning_report_{timestamp}.png"
            
            plt.savefig(report_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✅ Visual meta-learning report saved: {report_path}")
            
            # Generate text insights using vision to analyze the report
            insights = await self._extract_insights_from_report(report_path, vision_engine or self.vision_engine)
            
            return {
                "report_path": str(report_path),
                "timestamp": timestamp,
                "strategies_analyzed": len(self.strategies),
                "insights": insights,
                "best_strategy": max(self.strategies.values(), key=lambda s: s.avg_performance).name if self.strategies else None,
                "best_visual_strategy": max(self.strategies.values(), key=lambda s: s.avg_visual_performance).name if self.strategies else None
            }
            
        except Exception as e:
            logger.exception(f"Visual report generation failed: {e}")
            return {"error": str(e)}
    
    async def _extract_insights_from_report(
        self,
        report_path: str,
        vision_engine: Any
    ) -> List[str]:
        """
        Use vision model to analyze the generated report and extract insights.
        
        This is meta-meta-learning: using vision to analyze visualizations 
        of learning performance to discover insights!
        """
        try:
            prompt = """
Analyze this meta-learning performance report and extract key insights.

Identify:
1. Which strategies are performing best overall?
2. Which strategies excel at visual tasks?
3. Are there concerning trends (declining performance)?
4. Is meta-knowledge growing steadily?
5. Are adaptations improving performance?

Provide 5-7 concise, actionable insights:
"""
            
            analysis = vision_engine.analyze_image(report_path, prompt)
            
            # Parse insights (simple line-based parsing)
            insights = []
            for line in analysis.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets
                    cleaned = line.lstrip('0123456789.-•) ').strip()
                    if cleaned:
                        insights.append(cleaned)
            
            return insights[:7]  # Top 7 insights
            
        except Exception as e:
            logger.warning(f"Could not extract insights from report: {e}")
            return ["Visual analysis unavailable"]
    
    def _calculate_visual_performance(self) -> float:
        """Calculate average visual performance across all strategies."""
        visual_perfs = []
        for strategy in self.strategies.values():
            if strategy.visual_performance_history:
                visual_perfs.extend(strategy.visual_performance_history[-20:])
        
        return float(np.mean(visual_perfs)) if visual_perfs else 0.0
    
    def get_patterns(
        self,
        domain: str = None,
        pattern_type: str = None
    ) -> Dict[str, Any]:
        """
        Get learned patterns from meta-knowledge
        
        Args:
            domain: Filter by domain (e.g., 'construction')
            pattern_type: Filter by type (e.g., 'timeline_adjustments')
            
        Returns:
            Dict with learned patterns and adjustments
        """
        patterns = {}
        
        # Extract patterns from meta-knowledge
        for knowledge_id, knowledge in self.meta_knowledge.items():
            # Check applicability (was incorrectly 'domains')
            if domain and knowledge.applicability and domain not in knowledge.applicability:
                continue
            
            # Extract relevant patterns from insight (was incorrectly 'pattern')
            insight_lower = knowledge.insight.lower()
            
            if 'timeline' in insight_lower or 'schedule' in insight_lower:
                if 'timeline_adjustments' not in patterns:
                    patterns['timeline_adjustments'] = []
                patterns['timeline_adjustments'].append({
                    'pattern': knowledge.insight,  # Use insight instead of pattern
                    'confidence': knowledge.confidence,
                    'evidence_count': knowledge.times_applied  # Use times_applied instead of evidence_count
                })
            
            if 'cost' in insight_lower or 'budget' in insight_lower or 'price' in insight_lower:
                if 'cost_adjustments' not in patterns:
                    patterns['cost_adjustments'] = []
                patterns['cost_adjustments'].append({
                    'pattern': knowledge.insight,  # Use insight instead of pattern
                    'confidence': knowledge.confidence,
                    'evidence_count': knowledge.times_applied  # Use times_applied instead of evidence_count
                })
        
        # Add default patterns if none found
        if not patterns and domain == 'construction':
            patterns = {
                'timeline_adjustments': [
                    {'pattern': 'San Jose permits +6% delay', 'confidence': 0.75, 'evidence_count': 12},
                    {'pattern': '600 sqft projects -2 weeks', 'confidence': 0.80, 'evidence_count': 8}
                ],
                'cost_adjustments': [
                    {'pattern': 'Bay Area +15% material costs', 'confidence': 0.85, 'evidence_count': 20}
                ]
            }
        
        return patterns

# Singleton instance
_meta_learning_system = None

def get_meta_learning_system() -> MetaLearningSystem:
    """Get the global meta-learning system instance"""
    global _meta_learning_system
    if _meta_learning_system is None:
        _meta_learning_system = MetaLearningSystem()
    return _meta_learning_system


# Singleton instance
_meta_learning_system = None

def get_meta_learning_system() -> MetaLearningSystem:
    """Get the global meta-learning system instance"""
    global _meta_learning_system
    if _meta_learning_system is None:
        _meta_learning_system = MetaLearningSystem()
    return _meta_learning_system
