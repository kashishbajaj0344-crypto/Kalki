"""
Distributed Computing Agents (Phase 8)
=======================================

Agents for distributed computing, scaling, load balancing, and self-healing.
Implements real distributed algorithms with cross-agent cooperation and persistent state.
"""

from .compute_scaling import ComputeScalingAgent
from .load_balancing import LoadBalancingAgent
from .self_healing import SelfHealingAgent
from .compute_cluster_agent import ComputeClusterAgent
from .consensus_agent import ConsensusAgent
from .observability_agent import ObservabilityAgent

__all__ = [
    'ComputeScalingAgent',
    'LoadBalancingAgent',
    'SelfHealingAgent',
    'ComputeClusterAgent',
    'ConsensusAgent',
    'ObservabilityAgent'
]