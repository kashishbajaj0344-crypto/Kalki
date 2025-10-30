"""Compatibility shim: re-export safety agents from `modules.agents.safety`."""
from modules.agents.safety import (
    EthicsAgent,
    RiskAssessmentAgent,
    SimulationVerifierAgent,
)

__all__ = ["EthicsAgent", "RiskAssessmentAgent", "SimulationVerifierAgent"]
