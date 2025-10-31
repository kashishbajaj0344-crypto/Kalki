"""
Predictive Discovery Agent (Phase 14)
-------------------------------------
Implements predictive modeling for technology trends and future scenarios.
Uses time series analysis, trend extrapolation, and scenario planning.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import json
from modules.logging_config import get_logger

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.PredictiveDiscovery")


@dataclass
class TrendData:
    """Represents trend data with timestamps and values"""
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "values": self.values,
            "metadata": self.metadata
        }


@dataclass
class PredictionResult:
    """Represents prediction results with confidence intervals"""
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    trend_strength: float
    prediction_horizon: int
    model_accuracy: float


class PredictiveDiscoveryAgent(BaseAgent):
    """
    Predictive discovery agent for technology trends and future scenarios.
    Uses statistical modeling, trend analysis, and scenario planning.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="PredictiveDiscoveryAgent",
            capabilities=[
                AgentCapability.PREDICTIVE_DISCOVERY,
                AgentCapability.SIMULATION,
                AgentCapability.EXPERIMENTATION
            ],
            description="Predictive modeling and trend forecasting for technology discovery",
            config=config or {}
        )

        # Prediction parameters
        self.max_prediction_horizon = self.config.get('max_horizon', 365)  # days
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.min_data_points = self.config.get('min_data_points', 10)

        # Technology trend database
        self.trend_database = self._initialize_trend_database()

    async def initialize(self) -> bool:
        """Initialize predictive modeling environment"""
        try:
            # Test trend analysis
            test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            trend = self._calculate_trend_strength(test_data)
            logger.info(f"PredictiveDiscoveryAgent initialized with trend analysis capability")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize PredictiveDiscoveryAgent: {e}")
            return False

    def _initialize_trend_database(self) -> Dict[str, TrendData]:
        """Initialize database with known technology trends"""
        # Sample technology trends for demonstration
        base_date = datetime.now() - timedelta(days=365*2)

        trends = {}

        # AI/ML adoption trend
        ai_timestamps = [base_date + timedelta(days=i*30) for i in range(24)]
        ai_values = [0.1 + 0.8 * (1 - np.exp(-i/12)) + np.random.normal(0, 0.05) for i in range(24)]
        trends["ai_adoption"] = TrendData(ai_timestamps, ai_values,
                                        {"category": "AI", "description": "AI/ML technology adoption rate"})

        # Quantum computing progress
        qc_timestamps = [base_date + timedelta(days=i*45) for i in range(16)]
        qc_values = [0.01 + 0.15 * (1 - np.exp(-i/8)) + np.random.normal(0, 0.02) for i in range(16)]
        trends["quantum_computing"] = TrendData(qc_timestamps, qc_values,
                                              {"category": "Quantum", "description": "Quantum computing capability"})

        # Renewable energy adoption
        re_timestamps = [base_date + timedelta(days=i*20) for i in range(36)]
        re_values = [0.2 + 0.6 * (1 - np.exp(-i/18)) + np.random.normal(0, 0.03) for i in range(36)]
        trends["renewable_energy"] = TrendData(re_timestamps, re_values,
                                             {"category": "Energy", "description": "Renewable energy market share"})

        return trends

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute predictive discovery tasks"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "forecast":
            return await self._forecast_trend(params)
        elif action == "analyze_trends":
            return await self._analyze_trends(params)
        elif action == "scenario_planning":
            return await self._scenario_planning(params)
        elif action == "discover_opportunities":
            return await self._discover_opportunities(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _forecast_trend(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast future values for a technology trend"""
        try:
            trend_name = params.get("trend_name")
            horizon_days = params.get("horizon_days", 90)
            model_type = params.get("model_type", "polynomial")

            if trend_name not in self.trend_database:
                return {"status": "error", "error": f"Trend '{trend_name}' not found"}

            trend_data = self.trend_database[trend_name]

            if len(trend_data.values) < self.min_data_points:
                return {"status": "error", "error": "Insufficient data points for forecasting"}

            # Perform forecasting
            prediction = self._forecast_technology_trend(
                trend_data, horizon_days, model_type
            )

            return {
                "status": "success",
                "trend_name": trend_name,
                "prediction": {
                    "values": prediction.predicted_values,
                    "confidence_intervals": prediction.confidence_intervals,
                    "trend_strength": prediction.trend_strength,
                    "horizon_days": prediction.prediction_horizon,
                    "model_accuracy": prediction.model_accuracy
                },
                "metadata": trend_data.metadata
            }
        except Exception as e:
            logger.exception(f"Trend forecasting error: {e}")
            return {"status": "error", "error": str(e)}

    def _forecast_technology_trend(self, trend_data: TrendData, horizon_days: int,
                                 model_type: str) -> PredictionResult:
        """Forecast technology trend using statistical models"""
        values = np.array(trend_data.values)
        n_points = len(values)

        # Convert timestamps to days since start
        start_date = trend_data.timestamps[0]
        x_train = np.array([(ts - start_date).days for ts in trend_data.timestamps]).reshape(-1, 1)

        # Create prediction timeline
        x_pred = np.array([x_train[-1][0] + i for i in range(1, horizon_days + 1)]).reshape(-1, 1)

        predicted_values = []
        confidence_intervals = []

        if model_type == "polynomial":
            # Polynomial regression for non-linear trends
            poly_features = PolynomialFeatures(degree=2)
            x_train_poly = poly_features.fit_transform(x_train)
            x_pred_poly = poly_features.transform(x_pred)

            model = LinearRegression()
            model.fit(x_train_poly, values)

            predictions = model.predict(x_pred_poly)
            predicted_values = predictions.tolist()

            # Calculate confidence intervals (simplified)
            residuals = values - model.predict(x_train_poly)
            std_error = np.std(residuals)
            confidence_margin = 1.96 * std_error  # 95% confidence

            for pred in predictions:
                confidence_intervals.append((pred - confidence_margin, pred + confidence_margin))

        elif model_type == "exponential":
            # Exponential growth model for technology adoption
            # Use log transform for exponential fitting
            try:
                log_values = np.log(np.maximum(values, 0.001))  # Avoid log(0)
                model = LinearRegression()
                model.fit(x_train, log_values)

                log_predictions = model.predict(x_pred)
                predictions = np.exp(log_predictions)
                predicted_values = predictions.tolist()

                # Confidence intervals for exponential model
                residuals = log_values - model.predict(x_train)
                std_error = np.std(residuals)
                confidence_margin = 1.96 * std_error

                for log_pred in log_predictions:
                    log_lower = log_pred - confidence_margin
                    log_upper = log_pred + confidence_margin
                    confidence_intervals.append((np.exp(log_lower), np.exp(log_upper)))

            except Exception:
                # Fallback to linear if exponential fails
                model = LinearRegression()
                model.fit(x_train, values)
                predictions = model.predict(x_pred)
                predicted_values = predictions.tolist()
                std_error = np.std(values - model.predict(x_train))
                confidence_margin = 1.96 * std_error
                for pred in predictions:
                    confidence_intervals.append((pred - confidence_margin, pred + confidence_margin))

        else:
            # Default linear regression
            model = LinearRegression()
            model.fit(x_train, values)
            predictions = model.predict(x_pred)
            predicted_values = predictions.tolist()

            residuals = values - model.predict(x_train)
            std_error = np.std(residuals)
            confidence_margin = 1.96 * std_error

            for pred in predictions:
                confidence_intervals.append((pred - confidence_margin, pred + confidence_margin))

        # Calculate trend strength (slope of linear trend)
        linear_model = LinearRegression()
        linear_model.fit(x_train, values)
        trend_strength = linear_model.coef_[0]

        # Calculate model accuracy (R-squared on training data)
        r_squared = linear_model.score(x_train, values)

        return PredictionResult(
            predicted_values=predicted_values,
            confidence_intervals=confidence_intervals,
            trend_strength=trend_strength,
            prediction_horizon=horizon_days,
            model_accuracy=r_squared
        )

    async def _analyze_trends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze multiple technology trends for patterns and correlations"""
        try:
            trend_names = params.get("trend_names", list(self.trend_database.keys()))
            analysis_type = params.get("analysis_type", "correlation")

            available_trends = [name for name in trend_names if name in self.trend_database]

            if not available_trends:
                return {"status": "error", "error": "No valid trends specified"}

            # Perform trend analysis
            analysis_results = self._analyze_technology_trends(available_trends, analysis_type)

            return {
                "status": "success",
                "analysis_results": analysis_results,
                "analyzed_trends": available_trends,
                "analysis_type": analysis_type
            }
        except Exception as e:
            logger.exception(f"Trend analysis error: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_technology_trends(self, trend_names: List[str],
                                 analysis_type: str) -> Dict[str, Any]:
        """Analyze relationships between technology trends"""
        results = {}

        if analysis_type == "correlation":
            # Calculate correlation matrix
            trend_data = {}
            for name in trend_names:
                if name in self.trend_database:
                    values = self.trend_database[name].values
                    # Normalize to same length by interpolation
                    trend_data[name] = self._normalize_trend_length(values, 50)

            if len(trend_data) > 1:
                correlation_matrix = self._calculate_trend_correlations(trend_data)
                results["correlation_matrix"] = correlation_matrix

                # Find strongest correlations
                strong_correlations = self._find_strong_correlations(correlation_matrix, threshold=0.7)
                results["strong_correlations"] = strong_correlations

        elif analysis_type == "growth_rates":
            growth_analysis = {}
            for name in trend_names:
                if name in self.trend_database:
                    trend_data = self.trend_database[name]
                    growth_rate = self._calculate_growth_rate(trend_data.values)
                    growth_analysis[name] = {
                        "growth_rate": growth_rate,
                        "trend_strength": self._calculate_trend_strength(trend_data.values),
                        "volatility": np.std(trend_data.values) / np.mean(trend_data.values)
                    }
            results["growth_analysis"] = growth_analysis

        elif analysis_type == "convergence":
            # Analyze if trends are converging or diverging
            convergence_analysis = {}
            for name in trend_names:
                if name in self.trend_database:
                    trend_data = self.trend_database[name]
                    convergence_score = self._calculate_convergence_score(trend_data.values)
                    convergence_analysis[name] = convergence_score
            results["convergence_analysis"] = convergence_analysis

        return results

    def _normalize_trend_length(self, values: List[float], target_length: int) -> List[float]:
        """Normalize trend to target length using interpolation"""
        if len(values) == target_length:
            return values

        x_original = np.linspace(0, 1, len(values))
        x_target = np.linspace(0, 1, target_length)

        return np.interp(x_target, x_original, values).tolist()

    def _calculate_trend_correlations(self, trend_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between trends"""
        trend_names = list(trend_data.keys())
        correlation_matrix = {}

        for i, name1 in enumerate(trend_names):
            correlation_matrix[name1] = {}
            for j, name2 in enumerate(trend_names):
                if i == j:
                    correlation_matrix[name1][name2] = 1.0
                else:
                    corr = np.corrcoef(trend_data[name1], trend_data[name2])[0, 1]
                    correlation_matrix[name1][name2] = float(corr)

        return correlation_matrix

    def _find_strong_correlations(self, correlation_matrix: Dict[str, Dict[str, float]],
                                threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find strongly correlated trend pairs"""
        strong_pairs = []

        trend_names = list(correlation_matrix.keys())
        for i, name1 in enumerate(trend_names):
            for j, name2 in enumerate(trend_names):
                if i < j:  # Avoid duplicates
                    corr = abs(correlation_matrix[name1][name2])
                    if corr >= threshold:
                        strong_pairs.append({
                            "trend1": name1,
                            "trend2": name2,
                            "correlation": correlation_matrix[name1][name2],
                            "strength": "strong" if corr >= 0.8 else "moderate"
                        })

        return strong_pairs

    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate compound annual growth rate"""
        if len(values) < 2:
            return 0.0

        initial_value = values[0]
        final_value = values[-1]

        if initial_value == 0:
            return float('inf') if final_value > 0 else 0.0

        # Assume monthly data, calculate CAGR
        periods = len(values) - 1
        return (final_value / initial_value) ** (1 / periods) - 1

    def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return slope

    def _calculate_convergence_score(self, values: List[float]) -> float:
        """Calculate convergence score (how much the trend is accelerating/decelerating)"""
        if len(values) < 3:
            return 0.0

        # Calculate second derivative (acceleration)
        first_derivative = np.diff(values)
        second_derivative = np.diff(first_derivative)

        # Average acceleration as convergence score
        return np.mean(second_derivative)

    async def _scenario_planning(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate future scenarios based on trend analysis"""
        try:
            focus_area = params.get("focus_area", "technology")
            time_horizon = params.get("time_horizon", 5)  # years
            num_scenarios = params.get("num_scenarios", 3)

            # Generate technology scenarios
            scenarios = self._generate_technology_scenarios(
                focus_area, time_horizon, num_scenarios
            )

            return {
                "status": "success",
                "scenarios": scenarios,
                "focus_area": focus_area,
                "time_horizon_years": time_horizon,
                "num_scenarios": num_scenarios
            }
        except Exception as e:
            logger.exception(f"Scenario planning error: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_technology_scenarios(self, focus_area: str, time_horizon: int,
                                     num_scenarios: int) -> List[Dict[str, Any]]:
        """Generate plausible future technology scenarios"""
        scenarios = []

        # Base scenario templates
        scenario_templates = [
            {
                "name": "Accelerated Adoption",
                "description": f"Rapid adoption and integration of {focus_area} technologies",
                "drivers": ["Strong investment", "Regulatory support", "Technical breakthroughs"],
                "probability": 0.4,
                "impact": "High"
            },
            {
                "name": "Gradual Evolution",
                "description": f"Steady, incremental progress in {focus_area} development",
                "drivers": ["Market forces", "Incremental innovation", "Infrastructure development"],
                "probability": 0.5,
                "impact": "Medium"
            },
            {
                "name": "Disruptive Breakthrough",
                "description": f"Unexpected breakthrough revolutionizes {focus_area} landscape",
                "drivers": ["Scientific discovery", "Paradigm shift", "Cross-domain innovation"],
                "probability": 0.1,
                "impact": "Very High"
            }
        ]

        # Generate scenarios based on current trends
        relevant_trends = [name for name, data in self.trend_database.items()
                          if focus_area.lower() in data.metadata.get("category", "").lower()]

        for i, template in enumerate(scenario_templates[:num_scenarios]):
            scenario = template.copy()

            # Customize based on trend analysis
            if relevant_trends:
                trend_projections = []
                for trend_name in relevant_trends[:3]:  # Limit to 3 trends
                    trend_data = self.trend_database[trend_name]
                    projection = self._forecast_technology_trend(
                        trend_data, time_horizon * 365, "polynomial"
                    )
                    final_value = projection.predicted_values[-1] if projection.predicted_values else 0
                    trend_projections.append({
                        "trend": trend_name,
                        "projected_value": final_value,
                        "confidence": projection.model_accuracy
                    })

                scenario["trend_projections"] = trend_projections

            # Add timeline milestones
            scenario["milestones"] = self._generate_scenario_milestones(
                focus_area, time_horizon, scenario["name"]
            )

            scenarios.append(scenario)

        return scenarios

    def _generate_scenario_milestones(self, focus_area: str, time_horizon: int,
                                    scenario_name: str) -> List[Dict[str, Any]]:
        """Generate milestone projections for a scenario"""
        milestones = []

        # Generate yearly milestones
        for year in range(1, time_horizon + 1):
            if scenario_name == "Accelerated Adoption":
                milestone_desc = f"Widespread adoption of advanced {focus_area} solutions"
            elif scenario_name == "Gradual Evolution":
                milestone_desc = f"Incremental improvements in {focus_area} capabilities"
            else:  # Disruptive Breakthrough
                milestone_desc = f"Breakthrough innovation transforms {focus_area} landscape"

            milestones.append({
                "year": year,
                "description": milestone_desc,
                "probability": 0.8 - (year - 1) * 0.1  # Decreasing probability over time
            })

        return milestones

    async def _discover_opportunities(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Discover investment/research opportunities based on trend analysis"""
        try:
            criteria = params.get("criteria", {})
            max_opportunities = params.get("max_opportunities", 5)

            # Analyze trends for opportunities
            opportunities = self._analyze_trend_opportunities(criteria, max_opportunities)

            return {
                "status": "success",
                "opportunities": opportunities,
                "criteria": criteria,
                "total_found": len(opportunities)
            }
        except Exception as e:
            logger.exception(f"Opportunity discovery error: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_trend_opportunities(self, criteria: Dict[str, Any],
                                   max_opportunities: int) -> List[Dict[str, Any]]:
        """Analyze trends to identify investment/research opportunities"""
        opportunities = []

        for trend_name, trend_data in self.trend_database.items():
            # Calculate opportunity score based on multiple factors
            growth_rate = self._calculate_growth_rate(trend_data.values)
            trend_strength = self._calculate_trend_strength(trend_data.values)
            volatility = np.std(trend_data.values) / np.mean(trend_data.values)

            # Opportunity scoring algorithm
            opportunity_score = (
                growth_rate * 0.4 +  # Growth potential
                trend_strength * 0.3 +  # Trend momentum
                (1 / (1 + volatility)) * 0.3  # Stability (inverse of volatility)
            )

            # Apply criteria filters
            meets_criteria = True
            if "min_growth_rate" in criteria:
                meets_criteria &= growth_rate >= criteria["min_growth_rate"]
            if "max_volatility" in criteria:
                meets_criteria &= volatility <= criteria["max_volatility"]
            if "categories" in criteria:
                meets_criteria &= trend_data.metadata.get("category") in criteria["categories"]

            if meets_criteria and opportunity_score > 0:
                opportunities.append({
                    "trend_name": trend_name,
                    "opportunity_score": opportunity_score,
                    "growth_rate": growth_rate,
                    "trend_strength": trend_strength,
                    "volatility": volatility,
                    "category": trend_data.metadata.get("category"),
                    "description": trend_data.metadata.get("description"),
                    "recommendation": self._generate_investment_recommendation(
                        growth_rate, trend_strength, volatility
                    )
                })

        # Sort by opportunity score and return top results
        opportunities.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return opportunities[:max_opportunities]

    def _generate_investment_recommendation(self, growth_rate: float,
                                          trend_strength: float,
                                          volatility: float) -> str:
        """Generate investment recommendation based on trend metrics"""
        if growth_rate > 0.5 and volatility < 0.3:
            return "High potential - Strong growth with low risk"
        elif growth_rate > 0.3 and trend_strength > 0.1:
            return "Moderate potential - Steady growth trajectory"
        elif volatility > 0.5:
            return "High risk - Volatile but potentially rewarding"
        else:
            return "Conservative - Stable but slow growth"

    async def shutdown(self) -> bool:
        """Clean up predictive modeling resources"""
        logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True