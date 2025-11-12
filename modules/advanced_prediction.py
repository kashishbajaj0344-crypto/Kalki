"""
Advanced Prediction System
Comprehensive predictive analytics for Kalki.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from modules.llm import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A prediction result"""
    prediction_type: str  # "outcome", "timeline", "cost", "issue"
    value: Any
    confidence: float  # 0-1
    timeframe: Optional[str] = None
    reasoning: str = ""
    factors: List[str] = field(default_factory=list)


@dataclass
class RiskPrediction:
    """A risk prediction"""
    risk_name: str
    probability: float  # 0-1
    impact: str  # "low", "medium", "high", "critical"
    timeframe_days: int
    mitigation_strategies: List[str] = field(default_factory=list)


class AdvancedPredictionSystem:
    """
    Advanced prediction system that enables Kalki to:
    - Predict project outcomes
    - Forecast timeline risks and delays
    - Predict budget overruns
    - Predict issues before they occur
    - Predict market trends
    """
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.prediction_history: List[Dict[str, Any]] = []
        self.accuracy_tracking: Dict[str, List[float]] = {}  # Track prediction accuracy
    
    async def predict_project_outcome(
        self,
        project: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict project success/failure with confidence.
        
        Args:
            project: Project data
        
        Returns:
            Prediction with confidence and factors
        """
        logger.info(f"🔮 Predicting project outcome...")
        
        prompt = f"""Analyze this project and predict its outcome:

Project Data:
{json.dumps(project, indent=2, default=str)}

Predict:
1. Success probability (0-100%)
2. Likely outcome (success/failure/partial)
3. Key risk factors
4. Confidence in prediction (0-1)
5. Timeline to completion estimate

Provide structured prediction."""
        
        prediction_result = await self.llm_engine.generate(
            prompt=prompt,
            context={"task": "project_outcome_prediction"},
            use_advanced_reasoning=True,
            reasoning_method="tot"  # Use Tree-of-Thought for complex prediction
        )
        
        # Parse prediction
        prediction_text = str(prediction_result)
        
        # Extract values (simplified - would parse properly)
        success_prob = self._extract_percentage(prediction_text, "success")
        confidence = self._extract_confidence(prediction_text)
        outcome = "success" if success_prob > 50 else "failure" if success_prob < 30 else "partial"
        
        prediction = {
            "outcome": outcome,
            "success_probability": success_prob / 100.0,
            "confidence": confidence,
            "reasoning": prediction_text,
            "factors": self._extract_factors(prediction_text),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store for accuracy tracking
        self.prediction_history.append({
            "type": "outcome",
            "prediction": prediction,
            "project": project.get("project_id", "unknown")
        })
        
        return prediction
    
    async def predict_timeline_risks(
        self,
        project: Dict[str, Any],
        horizon_days: int = 30
    ) -> List[RiskPrediction]:
        """
        Predict timeline risks and delays.
        
        Args:
            project: Project data
            horizon_days: How far ahead to predict
        
        Returns:
            List of risk predictions
        """
        logger.info(f"⏰ Predicting timeline risks (next {horizon_days} days)...")
        
        prompt = f"""Analyze this project and predict timeline risks:

Project: {json.dumps(project, indent=2, default=str)}
Prediction Horizon: {horizon_days} days

Predict:
1. Potential delays
2. Bottlenecks
3. Resource constraints
4. External factors

For each risk, provide:
- Risk name
- Probability (0-1)
- Impact (low/medium/high/critical)
- Timeframe (days from now)
- Mitigation strategies"""
        
        prediction_result = await self.llm_engine.generate(
            prompt=prompt,
            context={"task": "timeline_risk_prediction"},
            use_advanced_reasoning=True
        )
        
        # Parse risks (simplified)
        risks = self._parse_risks(str(prediction_result))
        
        return risks
    
    async def predict_issues(
        self,
        project: Dict[str, Any],
        horizon_days: int = 7
    ) -> List[RiskPrediction]:
        """
        Predict issues N days ahead.
        
        Args:
            project: Project data
            horizon_days: Days ahead to predict
        
        Returns:
            List of issue predictions
        """
        logger.info(f"⚠️ Predicting issues (next {horizon_days} days)...")
        
        prompt = f"""Predict potential issues for this project in the next {horizon_days} days:

Project: {json.dumps(project, indent=2, default=str)}

Current Status:
- Stage: {project.get('current_stage', 'unknown')}
- Progress: {project.get('completion_percentage', 0)}%
- Timeline: {project.get('timeline_estimate_weeks', 0)} weeks

Predict:
1. Technical issues
2. Resource issues
3. Quality issues
4. Schedule issues

For each issue, provide:
- Issue name
- Probability (0-1)
- Impact (low/medium/high/critical)
- When it might occur (days from now)
- Prevention strategies"""
        
        prediction_result = await self.llm_engine.generate(
            prompt=prompt,
            context={"task": "issue_prediction"},
            use_advanced_reasoning=True
        )
        
        issues = self._parse_risks(str(prediction_result))
        
        return issues
    
    async def predict_cost_overrun(
        self,
        project: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict budget overruns.
        
        Args:
            project: Project data
        
        Returns:
            Cost prediction
        """
        logger.info(f"💰 Predicting cost overrun...")
        
        prompt = f"""Predict cost overrun risk for this project:

Project: {json.dumps(project, indent=2, default=str)}

Current:
- Budget: ${project.get('budget', 0):,.2f}
- Spent: ${project.get('spent', 0):,.2f}
- Remaining: ${project.get('budget', 0) - project.get('spent', 0):,.2f}

Predict:
1. Overrun probability (0-1)
2. Expected overrun amount (%)
3. Risk factors
4. Confidence"""
        
        prediction_result = await self.llm_engine.generate(
            prompt=prompt,
            context={"task": "cost_prediction"},
            use_advanced_reasoning=True
        )
        
        prediction_text = str(prediction_result)
        
        overrun_prob = self._extract_percentage(prediction_text, "overrun") / 100.0
        overrun_amount = self._extract_percentage(prediction_text, "amount")
        
        return {
            "overrun_probability": overrun_prob,
            "expected_overrun_percent": overrun_amount,
            "expected_overrun_amount": (project.get('budget', 0) * overrun_amount / 100.0),
            "confidence": self._extract_confidence(prediction_text),
            "reasoning": prediction_text,
            "risk_factors": self._extract_factors(prediction_text)
        }
    
    # Helper methods
    
    def _extract_percentage(self, text: str, keyword: str) -> float:
        """Extract percentage from text"""
        import re
        patterns = [
            rf"{keyword}.*?(\d+(?:\.\d+)?)\s*%",
            rf"(\d+(?:\.\d+)?)\s*%.*?{keyword}"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 50.0  # Default
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from text"""
        import re
        patterns = [
            r"confidence[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*confidence"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return 0.7  # Default
    
    def _extract_factors(self, text: str) -> List[str]:
        """Extract risk factors from text"""
        import re
        factors = []
        # Look for bullet points or numbered lists
        factor_patterns = [
            r"(?:^|\n)[\d\-•]\s*(.+?)(?=\n|$)",
            r"factor[:\s]+(.+?)(?=\n|\.|$)"
        ]
        for pattern in factor_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            factors.extend(matches)
        return factors[:5]  # Top 5
    
    def _parse_risks(self, text: str) -> List[RiskPrediction]:
        """Parse risk predictions from text"""
        risks = []
        # Simplified parsing - would be more sophisticated
        lines = text.split('\n')
        current_risk = None
        
        for line in lines:
            if any(word in line.lower() for word in ['risk', 'issue', 'problem']):
                if current_risk:
                    risks.append(current_risk)
                
                # Extract risk name
                risk_name = line.strip()
                current_risk = RiskPrediction(
                    risk_name=risk_name,
                    probability=0.5,
                    impact="medium",
                    timeframe_days=7
                )
            elif current_risk:
                if "probability" in line.lower() or "chance" in line.lower():
                    current_risk.probability = self._extract_percentage(line, "probability") / 100.0
                elif "impact" in line.lower():
                    impact = "low"
                    if "high" in line.lower() or "critical" in line.lower():
                        impact = "high"
                    elif "medium" in line.lower():
                        impact = "medium"
                    current_risk.impact = impact
                elif "day" in line.lower():
                    import re
                    day_match = re.search(r'(\d+)\s*days?', line, re.IGNORECASE)
                    if day_match:
                        current_risk.timeframe_days = int(day_match.group(1))
        
        if current_risk:
            risks.append(current_risk)
        
        return risks if risks else [
            RiskPrediction(
                risk_name="General risk",
                probability=0.5,
                impact="medium",
                timeframe_days=7
            )
        ]

