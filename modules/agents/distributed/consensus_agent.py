"""
Consensus Agent (Phase 8)
=========================

Implements distributed consensus protocols for fault-tolerant coordination
among distributed agents. Provides foundation for Byzantine fault tolerance
and distributed decision making.
"""

import asyncio
import hashlib
import time
import random
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.Consensus")


class ConsensusAlgorithm(Enum):
    """Supported consensus algorithms"""
    PAXOS = "paxos"
    RAFT = "raft"
    PBFT = "pbft"  # Practical Byzantine Fault Tolerance
    PROOF_OF_WORK = "pow"


class ConsensusState(Enum):
    """Consensus protocol states"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"
    PREPARING = "preparing"
    COMMITTING = "committing"


@dataclass
class ConsensusProposal:
    """A proposal for consensus"""
    proposal_id: str
    proposer_id: str
    value: Any
    timestamp: float
    term: int = 0
    votes: Set[str] = field(default_factory=set)
    accepted: bool = False


@dataclass
class ConsensusNode:
    """A node participating in consensus"""
    node_id: str
    address: str
    last_heartbeat: float
    term: int = 0
    state: ConsensusState = ConsensusState.FOLLOWER
    voted_for: Optional[str] = None


class ConsensusAgent(BaseAgent):
    """
    Distributed consensus agent implementing multiple consensus algorithms.
    Provides fault-tolerant coordination for distributed systems.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="ConsensusAgent",
            capabilities=[
                AgentCapability.CONSENSUS,
                AgentCapability.COORDINATION,
                AgentCapability.FAULT_TOLERANCE
            ],
            description="Distributed consensus for fault-tolerant coordination",
            config=config or {}
        )

        # Consensus configuration
        self.algorithm = ConsensusAlgorithm(self.config.get('algorithm', 'raft'))
        self.node_id = self.config.get('node_id', f"node_{random.randint(1000, 9999)}")
        self.cluster_size = self.config.get('cluster_size', 3)
        self.heartbeat_interval = self.config.get('heartbeat_interval', 1.0)
        self.election_timeout_min = self.config.get('election_timeout_min', 2.0)
        self.election_timeout_max = self.config.get('election_timeout_max', 4.0)

        # Consensus state
        self.current_term = 0
        self.state = ConsensusState.FOLLOWER
        self.voted_for: Optional[str] = None
        self.commit_index = 0
        self.last_applied = 0

        # Node management
        self.nodes: Dict[str, ConsensusNode] = {}
        self.leader_id: Optional[str] = None

        # Consensus data structures
        self.log: List[ConsensusProposal] = []
        self.pending_proposals: Dict[str, ConsensusProposal] = {}
        self.accepted_proposals: Dict[str, ConsensusProposal] = {}

        # Timing
        self.last_heartbeat = time.time()
        self.election_timeout = self._generate_election_timeout()

        # Byzantine fault tolerance (for PBFT)
        self.fault_tolerance = (self.cluster_size - 1) // 3
        self.sequence_number = 0
        self.view_number = 0
        self.primary_node = None

        # Node discovery
        self.discovery_enabled = True
        self.discovery_interval = 5.0
        self.known_addresses: Set[str] = set()

    async def initialize(self) -> bool:
        """Initialize consensus protocol"""
        try:
            logger.info(f"ConsensusAgent initializing with {self.algorithm.value} algorithm")

            # Initialize self as first node
            self.nodes[self.node_id] = ConsensusNode(
                node_id=self.node_id,
                address=self.config.get('address', 'localhost:8000'),
                last_heartbeat=time.time(),
                term=self.current_term,
                state=self.state
            )

            # Start consensus protocol
            if self.algorithm == ConsensusAlgorithm.RAFT:
                asyncio.create_task(self._raft_consensus_loop())
            elif self.algorithm == ConsensusAlgorithm.PBFT:
                asyncio.create_task(self._pbft_consensus_loop())
            elif self.algorithm == ConsensusAlgorithm.PAXOS:
                asyncio.create_task(self._paxos_consensus_loop())

            # Start node discovery
            if self.discovery_enabled:
                asyncio.create_task(self._node_discovery_loop())

            logger.info(f"ConsensusAgent initialized as {self.algorithm.value} node: {self.node_id}")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize ConsensusAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute consensus operations"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "propose_value":
            return await self._propose_value(params)
        elif action == "get_consensus_status":
            return await self._get_consensus_status(params)
        elif action == "add_node":
            return await self._add_cluster_node(params)
        elif action == "remove_node":
            return await self._remove_cluster_node(params)
        elif action == "force_election":
            return await self._force_election(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _propose_value(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Propose a value for consensus"""
        try:
            value = params.get("value")
            proposal_id = params.get("proposal_id", f"prop_{int(time.time() * 1000)}")

            if not value:
                return {"status": "error", "error": "No value provided for consensus"}

            # Create proposal
            proposal = ConsensusProposal(
                proposal_id=proposal_id,
                proposer_id=self.node_id,
                value=value,
                timestamp=time.time(),
                term=self.current_term
            )

            # Start consensus protocol based on algorithm
            if self.algorithm == ConsensusAlgorithm.RAFT:
                success = await self._raft_propose(proposal)
            elif self.algorithm == ConsensusAlgorithm.PBFT:
                success = await self._pbft_propose(proposal)
            elif self.algorithm == ConsensusAlgorithm.PAXOS:
                success = await self._paxos_propose(proposal)
            else:
                return {"status": "error", "error": "Unsupported consensus algorithm"}

            if success:
                return {
                    "status": "success",
                    "proposal_id": proposal_id,
                    "consensus_achieved": True,
                    "leader": self.leader_id,
                    "term": self.current_term
                }
            else:
                return {
                    "status": "error",
                    "proposal_id": proposal_id,
                    "consensus_achieved": False,
                    "error": "Consensus not reached"
                }

        except Exception as e:
            logger.exception(f"Failed to propose value: {e}")
            return {"status": "error", "error": str(e)}

    # RAFT Consensus Implementation

    async def _raft_consensus_loop(self):
        """Main RAFT consensus loop"""
        while True:
            try:
                current_time = time.time()

                if self.state == ConsensusState.LEADER:
                    # Send heartbeats
                    await self._send_heartbeats()
                    await asyncio.sleep(self.heartbeat_interval)

                elif self.state in [ConsensusState.FOLLOWER, ConsensusState.CANDIDATE]:
                    # Check for election timeout
                    if current_time - self.last_heartbeat > self.election_timeout:
                        await self._start_election()

                    await asyncio.sleep(0.1)  # Short sleep to prevent busy waiting

            except Exception as e:
                logger.exception(f"RAFT consensus loop error: {e}")
                await asyncio.sleep(1.0)

    async def _start_election(self):
        """Start leader election"""
        try:
            self.current_term += 1
            self.state = ConsensusState.CANDIDATE
            self.voted_for = self.node_id
            self.election_timeout = self._generate_election_timeout()

            logger.info(f"Starting election for term {self.current_term}")

            # Request votes from other nodes
            votes_received = 1  # Vote for self
            nodes_to_contact = [nid for nid in self.nodes.keys() if nid != self.node_id]

            for node_id in nodes_to_contact:
                try:
                    vote_granted = await self._request_vote(node_id)
                    if vote_granted:
                        votes_received += 1
                except Exception as e:
                    logger.warning(f"Failed to get vote from {node_id}: {e}")

            # Check if we have majority
            if votes_received > len(self.nodes) // 2:
                await self._become_leader()
            else:
                # Election failed, become follower
                self.state = ConsensusState.FOLLOWER
                self.voted_for = None

        except Exception as e:
            logger.exception(f"Election failed: {e}")

    async def _become_leader(self):
        """Become the leader"""
        try:
            self.state = ConsensusState.LEADER
            self.leader_id = self.node_id

            logger.info(f"Became leader for term {self.current_term}")

            # Send initial heartbeats
            await self._send_heartbeats()

        except Exception as e:
            logger.exception(f"Failed to become leader: {e}")

    async def _send_heartbeats(self):
        """Send heartbeats to all followers"""
        try:
            for node_id, node in self.nodes.items():
                if node_id != self.node_id:
                    try:
                        await self._send_heartbeat(node_id)
                    except Exception as e:
                        logger.warning(f"Failed to send heartbeat to {node_id}: {e}")

        except Exception as e:
            logger.exception(f"Heartbeat sending failed: {e}")

    async def _send_heartbeat(self, node_id: str):
        """Send heartbeat to a specific node"""
        # Placeholder for actual network communication
        # In a real implementation, this would send RPC calls
        pass

    async def _request_vote(self, node_id: str) -> bool:
        """Request vote from a node"""
        # Placeholder for actual network communication
        # In a real implementation, this would send RPC calls
        return random.choice([True, False])  # Simulate vote

    async def _raft_propose(self, proposal: ConsensusProposal) -> bool:
        """Propose value using RAFT"""
        try:
            if self.state != ConsensusState.LEADER:
                return False

            # Add to log
            self.log.append(proposal)

            # Replicate to followers (simplified)
            replication_success = await self._replicate_log_entries()

            if replication_success:
                self.commit_index += 1
                proposal.accepted = True
                return True

            return False

        except Exception as e:
            logger.exception(f"RAFT proposal failed: {e}")
            return False

    async def _replicate_log_entries(self) -> bool:
        """Replicate log entries to followers"""
        # Placeholder for log replication
        # In a real implementation, this would ensure majority replication
        return True

    # PBFT Consensus Implementation

    async def _pbft_consensus_loop(self):
        """Main PBFT consensus loop"""
        while True:
            try:
                # PBFT-specific logic would go here
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.exception(f"PBFT consensus loop error: {e}")
                await asyncio.sleep(1.0)

    async def _pbft_propose(self, proposal: ConsensusProposal) -> bool:
        """Propose value using PBFT"""
        try:
            # PBFT three-phase protocol: Pre-prepare, Prepare, Commit
            self.sequence_number += 1

            # Phase 1: Pre-prepare (only primary can do this)
            if self.primary_node == self.node_id:
                await self._pbft_pre_prepare(proposal)

            # Phase 2: Prepare
            await self._pbft_prepare(proposal)

            # Phase 3: Commit
            await self._pbft_commit(proposal)

            # Check if consensus reached
            if len(proposal.votes) >= (2 * self.fault_tolerance + 1):
                proposal.accepted = True
                return True

            return False

        except Exception as e:
            logger.exception(f"PBFT proposal failed: {e}")
            return False

    async def _pbft_pre_prepare(self, proposal: ConsensusProposal):
        """PBFT Pre-prepare phase"""
        # Placeholder for PBFT pre-prepare logic
        pass

    async def _pbft_prepare(self, proposal: ConsensusProposal):
        """PBFT Prepare phase"""
        # Placeholder for PBFT prepare logic
        proposal.votes.add(self.node_id)

    async def _pbft_commit(self, proposal: ConsensusProposal):
        """PBFT Commit phase"""
        # Placeholder for PBFT commit logic
        pass

    # PAXOS Consensus Implementation

    async def _paxos_consensus_loop(self):
        """Main PAXOS consensus loop"""
        while True:
            try:
                # PAXOS-specific logic would go here
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.exception(f"PAXOS consensus loop error: {e}")
                await asyncio.sleep(1.0)

    async def _paxos_propose(self, proposal: ConsensusProposal) -> bool:
        """Propose value using PAXOS"""
        try:
            # PAXOS two-phase protocol: Prepare, Accept
            proposal.term = self.current_term

            # Phase 1: Prepare
            promises = await self._paxos_prepare(proposal)

            if len(promises) >= (len(self.nodes) // 2 + 1):
                # Phase 2: Accept
                accepts = await self._paxos_accept(proposal)

                if len(accepts) >= (len(self.nodes) // 2 + 1):
                    proposal.accepted = True
                    return True

            return False

        except Exception as e:
            logger.exception(f"PAXOS proposal failed: {e}")
            return False

    async def _paxos_prepare(self, proposal: ConsensusProposal) -> List[str]:
        """PAXOS Prepare phase"""
        # Placeholder for PAXOS prepare logic
        return [self.node_id]  # Simulate promises

    async def _paxos_accept(self, proposal: ConsensusProposal) -> List[str]:
        """PAXOS Accept phase"""
        # Placeholder for PAXOS accept logic
        return [self.node_id]  # Simulate accepts

    # Node Discovery

    async def _node_discovery_loop(self):
        """Continuous node discovery"""
        while self.discovery_enabled:
            try:
                await self._discover_nodes()
                await asyncio.sleep(self.discovery_interval)
            except Exception as e:
                logger.exception(f"Node discovery error: {e}")
                await asyncio.sleep(self.discovery_interval)

    async def _discover_nodes(self):
        """Discover new nodes in the network"""
        try:
            # Placeholder for node discovery logic
            # In a real implementation, this might use:
            # - Multicast discovery
            # - Service registry
            # - Gossip protocol
            # - DNS-SD

            # Simulate discovering a new node occasionally
            if random.random() < 0.1:  # 10% chance
                new_node_id = f"discovered_node_{random.randint(1000, 9999)}"
                if new_node_id not in self.nodes:
                    self.nodes[new_node_id] = ConsensusNode(
                        node_id=new_node_id,
                        address=f"discovered:{random.randint(8000, 9000)}",
                        last_heartbeat=time.time()
                    )
                    logger.info(f"Discovered new node: {new_node_id}")

        except Exception as e:
            logger.exception(f"Node discovery failed: {e}")

    # Cluster Management

    async def _add_cluster_node(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new node to the cluster"""
        try:
            node_id = params.get("node_id")
            address = params.get("address")

            if not node_id or not address:
                return {"status": "error", "error": "node_id and address required"}

            if node_id in self.nodes:
                return {"status": "error", "error": "Node already exists"}

            self.nodes[node_id] = ConsensusNode(
                node_id=node_id,
                address=address,
                last_heartbeat=time.time()
            )

            logger.info(f"Added node to cluster: {node_id}")
            return {"status": "success", "node_id": node_id}

        except Exception as e:
            logger.exception(f"Failed to add cluster node: {e}")
            return {"status": "error", "error": str(e)}

    async def _remove_cluster_node(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a node from the cluster"""
        try:
            node_id = params.get("node_id")

            if not node_id:
                return {"status": "error", "error": "node_id required"}

            if node_id not in self.nodes:
                return {"status": "error", "error": "Node not found"}

            del self.nodes[node_id]

            # If we removed the leader, trigger election
            if self.leader_id == node_id:
                self.leader_id = None
                if self.algorithm == ConsensusAlgorithm.RAFT:
                    await self._start_election()

            logger.info(f"Removed node from cluster: {node_id}")
            return {"status": "success", "node_id": node_id}

        except Exception as e:
            logger.exception(f"Failed to remove cluster node: {e}")
            return {"status": "error", "error": str(e)}

    async def _force_election(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Force a leader election"""
        try:
            if self.algorithm == ConsensusAlgorithm.RAFT:
                await self._start_election()
                return {"status": "success", "election_started": True}
            else:
                return {"status": "error", "error": "Election forcing not supported for this algorithm"}

        except Exception as e:
            logger.exception(f"Failed to force election: {e}")
            return {"status": "error", "error": str(e)}

    # Utility Methods

    def _generate_election_timeout(self) -> float:
        """Generate random election timeout"""
        return random.uniform(self.election_timeout_min, self.election_timeout_max)

    async def _get_consensus_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive consensus status"""
        return {
            "status": "success",
            "node_id": self.node_id,
            "algorithm": self.algorithm.value,
            "current_term": self.current_term,
            "state": self.state.value,
            "leader": self.leader_id,
            "cluster_size": len(self.nodes),
            "nodes": {nid: {"address": node.address, "state": node.state.value}
                     for nid, node in self.nodes.items()},
            "pending_proposals": len(self.pending_proposals),
            "accepted_proposals": len(self.accepted_proposals),
            "log_size": len(self.log),
            "fault_tolerance": self.fault_tolerance
        }

    async def shutdown(self) -> bool:
        """Shutdown the consensus agent"""
        try:
            logger.info("ConsensusAgent shutting down")

            # Stop consensus loops
            self.discovery_enabled = False

            # Clear state
            self.nodes.clear()
            self.pending_proposals.clear()
            self.accepted_proposals.clear()
            self.log.clear()

            logger.info("ConsensusAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during ConsensusAgent shutdown: {e}")
            return False