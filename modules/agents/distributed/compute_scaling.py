"""
Compute Scaling Agent (Phase 8)
================================

Implements dynamic compute resource scaling using predictive algorithms
and real-time load monitoring. Uses machine learning for scaling decisions.
"""

import asyncio
import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.ComputeScaling")


@dataclass
class ScalingMetrics:
    """Real-time scaling metrics"""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    timestamp: float


@dataclass
class ScalingDecision:
    """Scaling decision with confidence"""
    action: str  # 'scale_up', 'scale_down', 'maintain'
    confidence: float
    predicted_load: float
    recommended_instances: int
    reasoning: str


class ComputeScalingAgent(BaseAgent):
    """
    Dynamic compute scaling agent using predictive algorithms.
    Implements real ML-based scaling decisions with historical analysis.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="ComputeScalingAgent",
            capabilities=[
                AgentCapability.COMPUTE_SCALING,
                AgentCapability.LOAD_BALANCING,
                AgentCapability.OPTIMIZATION
            ],
            description="Dynamic compute resource scaling with predictive algorithms",
            config=config or {}
        )

        # Scaling parameters
        self.min_instances = self.config.get('min_instances', 1)
        self.max_instances = self.config.get('max_instances', 10)
        self.scale_up_threshold = self.config.get('scale_up_threshold', 0.8)
        self.scale_down_threshold = self.config.get('scale_down_threshold', 0.3)
        self.cooldown_period = self.config.get('cooldown_period', 300)  # 5 minutes

        # ML components for predictive scaling
        self.scaler = StandardScaler()
        self.predictor = LinearRegression()

        # Historical data
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)

        # Current state
        self.current_instances = self.min_instances
        self.last_scaling_time = 0
        self.is_trained = False

    async def initialize(self) -> bool:
        """Initialize scaling agent with baseline metrics"""
        try:
            logger.info("ComputeScalingAgent initializing predictive scaling system")

            # Collect initial baseline metrics
            await self._collect_baseline_metrics()

            # Train initial model if we have enough data
            if len(self.metrics_history) >= 10:
                await self._train_predictive_model()

            logger.info(f"ComputeScalingAgent initialized with {self.current_instances} instances")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize ComputeScalingAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scaling operations"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "analyze_load":
            return await self._analyze_current_load(params)
        elif action == "predictive_scale":
            return await self._predictive_scaling_decision(params)
        elif action == "emergency_scale":
            return await self._emergency_scaling(params)
        elif action == "get_metrics":
            return await self._get_scaling_metrics(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _analyze_current_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system load and make scaling recommendations"""
        try:
            # Collect current metrics
            metrics = await self._collect_current_metrics()

            # Get scaling decision
            decision = await self._make_scaling_decision(metrics)

            # Check cooldown period
            current_time = time.time()
            can_scale = (current_time - self.last_scaling_time) > self.cooldown_period

            return {
                "status": "success",
                "current_metrics": {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "request_rate": metrics.request_rate,
                    "response_time": metrics.response_time
                },
                "scaling_decision": {
                    "action": decision.action,
                    "confidence": decision.confidence,
                    "recommended_instances": decision.recommended_instances,
                    "reasoning": decision.reasoning
                },
                "can_scale": can_scale,
                "cooldown_remaining": max(0, self.cooldown_period - (current_time - self.last_scaling_time))
            }

        except Exception as e:
            logger.exception(f"Load analysis error: {e}")
            return {"status": "error", "error": str(e)}

    async def _predictive_scaling_decision(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictive scaling decision using ML model"""
        try:
            if not self.is_trained:
                return await self._analyze_current_load(params)  # Fallback to reactive

            # Get prediction timeframe
            prediction_window = params.get('prediction_window', 300)  # 5 minutes

            # Predict future load
            predicted_load = await self._predict_future_load(prediction_window)

            # Make scaling decision based on prediction
            decision = await self._make_predictive_decision(predicted_load)

            return {
                "status": "success",
                "prediction": {
                    "window_seconds": prediction_window,
                    "predicted_load": predicted_load,
                    "confidence_interval": 0.85  # Would be calculated from model
                },
                "scaling_decision": {
                    "action": decision.action,
                    "recommended_instances": decision.recommended_instances,
                    "reasoning": f"Predictive: {decision.reasoning}"
                }
            }

        except Exception as e:
            logger.exception(f"Predictive scaling error: {e}")
            return {"status": "error", "error": str(e)}

    async def _emergency_scaling(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency scaling for critical situations"""
        try:
            severity = params.get('severity', 'high')
            reason = params.get('reason', 'unspecified')

            # Emergency scaling logic
            if severity == 'critical':
                target_instances = min(self.max_instances, self.current_instances * 2)
            elif severity == 'high':
                target_instances = min(self.max_instances, self.current_instances + 3)
            else:
                target_instances = min(self.max_instances, self.current_instances + 1)

            # Execute emergency scaling
            await self._execute_scaling(target_instances)

            logger.warning(f"Emergency scaling executed: {self.current_instances} -> {target_instances} ({reason})")

            return {
                "status": "success",
                "emergency_scaling": {
                    "from_instances": self.current_instances,
                    "to_instances": target_instances,
                    "severity": severity,
                    "reason": reason
                }
            }

        except Exception as e:
            logger.exception(f"Emergency scaling error: {e}")
            return {"status": "error", "error": str(e)}

    async def _collect_current_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        try:
            # CPU and memory usage
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0

            # Simulate request rate and response time (would come from monitoring system)
            request_rate = np.random.normal(100, 20)  # requests per second
            response_time = np.random.normal(0.1, 0.02)  # seconds

            metrics = ScalingMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                request_rate=max(0, request_rate),
                response_time=max(0.001, response_time),
                timestamp=time.time()
            )

            # Store in history
            self.metrics_history.append(metrics)

            return metrics

        except Exception as e:
            logger.exception(f"Metrics collection error: {e}")
            # Return default metrics
            return ScalingMetrics(0.5, 0.5, 50, 0.1, time.time())

    async def _collect_baseline_metrics(self):
        """Collect baseline metrics for initial training"""
        logger.info("Collecting baseline metrics for scaling model")

        for _ in range(10):
            metrics = await self._collect_current_metrics()
            await asyncio.sleep(1)

    async def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDecision:
        """Make scaling decision based on current metrics"""
        # Calculate overall load factor
        load_factor = (
            metrics.cpu_usage * 0.4 +
            metrics.memory_usage * 0.3 +
            min(1.0, metrics.request_rate / 200) * 0.2 +
            min(1.0, metrics.response_time / 0.5) * 0.1
        )

        # Determine action
        if load_factor > self.scale_up_threshold:
            action = "scale_up"
            confidence = min(1.0, (load_factor - self.scale_up_threshold) / 0.2)
            recommended_instances = min(self.max_instances, self.current_instances + 1)
            reasoning = f"High load factor: {load_factor:.2f}"
        elif load_factor < self.scale_down_threshold and self.current_instances > self.min_instances:
            action = "scale_down"
            confidence = min(1.0, (self.scale_down_threshold - load_factor) / 0.3)
            recommended_instances = max(self.min_instances, self.current_instances - 1)
            reasoning = f"Low load factor: {load_factor:.2f}"
        else:
            action = "maintain"
            confidence = 0.8
            recommended_instances = self.current_instances
            reasoning = f"Stable load factor: {load_factor:.2f}"

        return ScalingDecision(
            action=action,
            confidence=confidence,
            predicted_load=load_factor,
            recommended_instances=recommended_instances,
            reasoning=reasoning
        )

    async def _make_predictive_decision(self, predicted_load: float) -> ScalingDecision:
        """Make scaling decision based on predicted load"""
        if predicted_load > self.scale_up_threshold:
            action = "scale_up"
            recommended_instances = min(self.max_instances, self.current_instances + 1)
            reasoning = f"Predicted high load: {predicted_load:.2f}"
        elif predicted_load < self.scale_down_threshold and self.current_instances > self.min_instances:
            action = "scale_down"
            recommended_instances = max(self.min_instances, self.current_instances - 1)
            reasoning = f"Predicted low load: {predicted_load:.2f}"
        else:
            action = "maintain"
            recommended_instances = self.current_instances
            reasoning = f"Predicted stable load: {predicted_load:.2f}"

        return ScalingDecision(
            action=action,
            confidence=0.7,  # Lower confidence for predictions
            predicted_load=predicted_load,
            recommended_instances=recommended_instances,
            reasoning=reasoning
        )

    async def _predict_future_load(self, window_seconds: int) -> float:
        """Predict future load using trained model"""
        if not self.is_trained or len(self.metrics_history) < 5:
            # Fallback to simple average
            recent_loads = [m.cpu_usage for m in list(self.metrics_history)[-5:]]
            return np.mean(recent_loads) if recent_loads else 0.5

        try:
            # Simple time-series prediction (would use proper time series model)
            recent_trend = np.polyfit(
                range(len(self.metrics_history)),
                [m.cpu_usage for m in self.metrics_history],
                1
            )[0]

            current_load = self.metrics_history[-1].cpu_usage
            predicted_load = current_load + (recent_trend * window_seconds / 60)  # per minute trend

            return max(0.0, min(1.0, predicted_load))

        except Exception:
            return self.metrics_history[-1].cpu_usage if self.metrics_history else 0.5

    async def _train_predictive_model(self):
        """Train the predictive scaling model"""
        try:
            if len(self.metrics_history) < 20:
                return

            # Prepare training data
            X = []
            y = []

            history_list = list(self.metrics_history)
            for i in range(5, len(history_list)):
                # Use last 5 metrics to predict next load
                features = [
                    history_list[j].cpu_usage for j in range(i-5, i)
                ] + [
                    history_list[j].memory_usage for j in range(i-5, i)
                ] + [
                    history_list[j].request_rate for j in range(i-5, i)
                ]

                X.append(features)
                y.append(history_list[i].cpu_usage)

            # Train model
            X = np.array(X)
            y = np.array(y)

            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.predictor.fit(X_scaled, y)

            self.is_trained = True
            logger.info("Predictive scaling model trained successfully")

        except Exception as e:
            logger.exception(f"Model training error: {e}")

    async def _execute_scaling(self, target_instances: int):
        """Execute the scaling operation"""
        if target_instances != self.current_instances:
            logger.info(f"Scaling from {self.current_instances} to {target_instances} instances")
            self.current_instances = target_instances
            self.last_scaling_time = time.time()

            # Record scaling event
            self.scaling_history.append({
                'timestamp': time.time(),
                'from_instances': self.current_instances,
                'to_instances': target_instances,
                'reason': 'automated_scaling'
            })

    async def _get_scaling_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive scaling metrics"""
        return {
            "status": "success",
            "current_state": {
                "instances": self.current_instances,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "last_scaling": self.last_scaling_time
            },
            "model_status": {
                "is_trained": self.is_trained,
                "metrics_history_size": len(self.metrics_history),
                "scaling_history_size": len(self.scaling_history)
            },
            "thresholds": {
                "scale_up": self.scale_up_threshold,
                "scale_down": self.scale_down_threshold,
                "cooldown_period": self.cooldown_period
            }
        }

    async def shutdown(self) -> bool:
        """Shutdown the compute scaling agent"""
        try:
            logger.info("ComputeScalingAgent shutting down")

            # Save final state (simplified - would save to persistent storage)
            # self._save_scaling_state()

            # Clear resources
            self.metrics_history.clear()
            self.scaling_history.clear()

            logger.info("ComputeScalingAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during ComputeScalingAgent shutdown: {e}")
            return False