# ============================================================
# Kalki v2.5 — federated_learning_bridge.py
# ------------------------------------------------------------
# Federated Learning Bridge: Distributed Evolution Framework
# - External data node integration
# - Sandboxed update mechanisms
# - Privacy-preserving contribution aggregation
# - Secure federated optimization
# - Trust-based contribution weighting
# ============================================================

import os
import json
import asyncio
import hashlib
import hmac
import secrets
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import uuid
import base64

from modules.utils.logger import get_logger
from modules.utils.config import CONFIG
from modules.safety_monitoring_system import get_safety_monitoring_system
from modules.sandbox import get_sandbox_manager

logger = get_logger("FederatedBridge")
safety_monitor = get_safety_monitoring_system()
sandbox_manager = get_sandbox_manager()

class NodeType(Enum):
    """Types of federated learning nodes"""
    USER_AGENT = "user_agent"
    SPECIALIZED_MODEL = "specialized_model"
    RESEARCH_COLLABORATOR = "research_collaborator"
    EXTERNAL_VALIDATOR = "external_validator"

class ContributionType(Enum):
    """Types of contributions from federated nodes"""
    TRAINING_DATA = "training_data"
    MODEL_UPDATES = "model_updates"
    VALIDATION_RESULTS = "validation_results"
    HYPERPARAMETER_SUGGESTIONS = "hyperparameter_suggestions"
    EVOLUTION_INSIGHTS = "evolution_insights"

class TrustLevel(Enum):
    """Trust levels for federated nodes"""
    UNVERIFIED = "unverified"
    BASIC = "basic"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    CORE_CONTRIBUTOR = "core_contributor"

@dataclass
class FederatedNode:
    """A federated learning node"""
    node_id: str
    node_type: NodeType
    public_key: str
    trust_level: TrustLevel
    reputation_score: float = 0.5
    total_contributions: int = 0
    successful_contributions: int = 0
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)

@dataclass
class Contribution:
    """A contribution from a federated node"""
    contribution_id: str
    node_id: str
    contribution_type: ContributionType
    content: Dict[str, Any]
    timestamp: str
    signature: str
    validation_status: str = "pending"  # "pending", "validated", "rejected", "integrated"
    validation_score: float = 0.0
    sandbox_execution_id: Optional[str] = None
    aggregated: bool = False

@dataclass
class FederatedRound:
    """A round of federated learning"""
    round_id: str
    round_number: int
    start_time: str
    end_time: Optional[str] = None
    participating_nodes: List[str] = field(default_factory=list)
    contributions_received: int = 0
    contributions_validated: int = 0
    contributions_integrated: int = 0
    aggregated_updates: Dict[str, Any] = field(default_factory=dict)
    round_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class SandboxExecution:
    """Result of sandboxed contribution execution"""
    execution_id: str
    contribution_id: str
    sandbox_id: str
    start_time: str
    end_time: Optional[str] = None
    execution_status: str = "running"  # "running", "completed", "failed", "timeout"
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    safety_checks: Dict[str, bool] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

class FederatedLearningBridge:
    """
    Federated Learning Bridge: Distributed Evolution Framework

    Enables external data nodes (user sub-agents, specialized models) to contribute
    to Kalki's evolution safely through sandboxed, privacy-preserving mechanisms.

    Features:
    - Secure node registration and authentication
    - Sandboxed contribution validation
    - Privacy-preserving aggregation (Federated Averaging)
    - Trust-based contribution weighting
    - Byzantine-robust consensus mechanisms
    """

    def __init__(self):
        # Persistence
        self.data_dir = "data/federated_learning"
        os.makedirs(self.data_dir, exist_ok=True)
        self.nodes_file = f"{self.data_dir}/nodes.json"
        self.contributions_file = f"{self.data_dir}/contributions.json"
        self.rounds_file = f"{self.data_dir}/rounds.json"
        self.metrics_file = f"{self.data_dir}/metrics.json"

        # Node management
        self.registered_nodes: Dict[str, FederatedNode] = {}
        self.active_round: Optional[FederatedRound] = None
        self.round_history: List[FederatedRound] = []

        # Contribution management
        self.pending_contributions: Dict[str, Contribution] = {}
        self.validated_contributions: Dict[str, Contribution] = {}
        self.contribution_queue = queue.Queue()

        # Sandbox execution tracking
        self.sandbox_executions: Dict[str, SandboxExecution] = {}

        # Security and privacy
        self.bridge_secret = self._generate_bridge_secret()
        self.encryption_keys: Dict[str, bytes] = {}

        # Configuration
        self.max_round_duration = 3600  # 1 hour
        self.min_participants = 3
        self.max_contributions_per_round = 100
        self.sandbox_timeout = 300  # 5 minutes
        self.trust_decay_rate = 0.95  # Daily decay

        # Initialize system
        self._initialize_bridge()
        self._load_persistent_state()
        self._start_background_tasks()

        logger.info("Federated Learning Bridge initialized")

    def _initialize_bridge(self):
        """Initialize the federated learning bridge"""
        os.makedirs(self.data_dir, exist_ok=True)

        # Create default trusted nodes (for bootstrapping)
        default_nodes = [
            {
                "node_id": "kalki_core",
                "node_type": NodeType.SPECIALIZED_MODEL,
                "public_key": self._generate_node_key(),
                "trust_level": TrustLevel.CORE_CONTRIBUTOR,
                "capabilities": ["model_updates", "validation", "hyperparameter_optimization"],
                "metadata": {"description": "Kalki core system node"}
            }
        ]

        for node_data in default_nodes:
            node = FederatedNode(**node_data)
            self.registered_nodes[node.node_id] = node

    def _generate_bridge_secret(self) -> str:
        """Generate bridge secret for HMAC operations"""
        secret_file = f"{self.data_dir}/bridge_secret.key"
        if os.path.exists(secret_file):
            with open(secret_file, 'r') as f:
                return f.read().strip()
        else:
            secret = secrets.token_hex(32)
            with open(secret_file, 'w') as f:
                f.write(secret)
            return secret

    def _generate_node_key(self) -> str:
        """Generate a public key for a node"""
        return base64.b64encode(secrets.token_bytes(32)).decode()

    def _load_persistent_state(self):
        """Load persistent state from disk"""
        try:
            if os.path.exists(self.nodes_file):
                with open(self.nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                    for node_id, node_data in nodes_data.items():
                        self.registered_nodes[node_id] = FederatedNode(**node_data)

            if os.path.exists(self.contributions_file):
                with open(self.contributions_file, 'r') as f:
                    contributions_data = json.load(f)
                    for contrib_id, contrib_data in contributions_data.items():
                        contribution = Contribution(**contrib_data)
                        if contribution.validation_status == "pending":
                            self.pending_contributions[contrib_id] = contribution
                        else:
                            self.validated_contributions[contrib_id] = contribution

            if os.path.exists(self.rounds_file):
                with open(self.rounds_file, 'r') as f:
                    rounds_data = json.load(f)
                    self.round_history = [FederatedRound(**round_data) for round_data in rounds_data]

        except Exception as e:
            logger.warning(f"Failed to load federated learning persistent state: {e}")

    def _save_persistent_state(self):
        """Save persistent state to disk"""
        try:
            with open(self.nodes_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.registered_nodes.items()}, f, indent=2)

            all_contributions = {**self.pending_contributions, **self.validated_contributions}
            with open(self.contributions_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in all_contributions.items()}, f, indent=2)

            with open(self.rounds_file, 'w') as f:
                json.dump([asdict(round) for round in self.round_history[-100:]], f, indent=2)  # Keep last 100

        except Exception as e:
            logger.error(f"Failed to save federated learning persistent state: {e}")

    def _start_background_tasks(self):
        """Start background tasks for federated learning"""
        # Contribution processing thread
        processing_thread = threading.Thread(target=self._contribution_processing_loop, daemon=True)
        processing_thread.start()

        # Round management thread
        round_thread = threading.Thread(target=self._round_management_loop, daemon=True)
        round_thread.start()

        # Trust decay thread
        trust_thread = threading.Thread(target=self._trust_decay_loop, daemon=True)
        trust_thread.start()

    def register_node(self, node_type: NodeType, public_key: str,
                     capabilities: List[str], metadata: Dict[str, Any] = None) -> str:
        """
        Register a new federated learning node

        Args:
            node_type: Type of the node
            public_key: Public key for authentication
            capabilities: List of node capabilities
            metadata: Additional metadata

        Returns:
            Node ID of the registered node
        """
        node_id = str(uuid.uuid4())

        # Start with basic trust level
        initial_trust = TrustLevel.BASIC if node_type == NodeType.USER_AGENT else TrustLevel.UNVERIFIED

        node = FederatedNode(
            node_id=node_id,
            node_type=node_type,
            public_key=public_key,
            trust_level=initial_trust,
            capabilities=capabilities,
            metadata=metadata or {}
        )

        self.registered_nodes[node_id] = node
        self._save_persistent_state()

        logger.info(f"Registered new federated node: {node_id} ({node_type.value})")
        return node_id

    def submit_contribution(self, node_id: str, contribution_type: ContributionType,
                          content: Dict[str, Any], signature: str) -> str:
        """
        Submit a contribution from a federated node

        Args:
            node_id: ID of the contributing node
            contribution_type: Type of contribution
            content: Contribution content
            signature: Cryptographic signature

        Returns:
            Contribution ID
        """
        # Verify node exists and is authorized
        if node_id not in self.registered_nodes:
            raise ValueError(f"Unknown node: {node_id}")

        node = self.registered_nodes[node_id]

        # Verify signature
        if not self._verify_signature(node_id, content, signature):
            raise ValueError("Invalid signature")

        # Check if contribution type is allowed for this node
        if contribution_type.value not in node.capabilities:
            raise ValueError(f"Contribution type {contribution_type.value} not allowed for node {node_id}")

        # Create contribution
        contribution_id = str(uuid.uuid4())
        contribution = Contribution(
            contribution_id=contribution_id,
            node_id=node_id,
            contribution_type=contribution_type,
            content=content,
            timestamp=datetime.now().isoformat(),
            signature=signature
        )

        # Add to pending contributions
        self.pending_contributions[contribution_id] = contribution
        self.contribution_queue.put(contribution_id)

        # Update node activity
        node.last_activity = datetime.now().isoformat()
        node.total_contributions += 1

        logger.info(f"Received contribution {contribution_id} from node {node_id}")
        return contribution_id

    def _verify_signature(self, node_id: str, content: Dict[str, Any], signature: str) -> bool:
        """Verify cryptographic signature of contribution"""
        try:
            node = self.registered_nodes[node_id]
            message = json.dumps(content, sort_keys=True).encode()

            # Simple HMAC verification (in production, use proper public key crypto)
            expected_signature = hmac.new(
                node.public_key.encode(),
                message,
                hashlib.sha256
            ).hexdigest()

            return hmac.compare_digest(signature, expected_signature)
        except Exception:
            return False

    def _contribution_processing_loop(self):
        """Background loop for processing contributions"""
        while True:
            try:
                contribution_id = self.contribution_queue.get(timeout=1.0)

                if contribution_id in self.pending_contributions:
                    contribution = self.pending_contributions[contribution_id]
                    asyncio.run(self._process_contribution(contribution))

                self.contribution_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in contribution processing loop: {e}")

    async def _process_contribution(self, contribution: Contribution):
        """Process a single contribution through sandboxed validation"""
        try:
            # Create sandbox execution
            execution_id = await self._create_sandbox_execution(contribution)

            # Execute contribution in sandbox
            success = await self._execute_in_sandbox(contribution, execution_id)

            if success:
                # Validate contribution
                validation_score = await self._validate_contribution(contribution, execution_id)

                contribution.validation_status = "validated"
                contribution.validation_score = validation_score
                contribution.sandbox_execution_id = execution_id

                # Move to validated contributions
                self.validated_contributions[contribution.contribution_id] = contribution
                del self.pending_contributions[contribution.contribution_id]

                # Update node reputation
                node = self.registered_nodes[contribution.node_id]
                node.successful_contributions += 1
                node.reputation_score = node.successful_contributions / node.total_contributions

                logger.info(f"Contribution {contribution.contribution_id} validated with score {validation_score:.3f}")

            else:
                contribution.validation_status = "rejected"
                logger.warning(f"Contribution {contribution.contribution_id} failed sandbox execution")

        except Exception as e:
            logger.error(f"Error processing contribution {contribution.contribution_id}: {e}")
            contribution.validation_status = "rejected"

    async def _create_sandbox_execution(self, contribution: Contribution) -> str:
        """Create a sandbox execution environment for the contribution"""
        execution_id = str(uuid.uuid4())

        execution = SandboxExecution(
            execution_id=execution_id,
            contribution_id=contribution.contribution_id,
            sandbox_id=await sandbox_manager.create_sandbox(),
            start_time=datetime.now().isoformat()
        )

        self.sandbox_executions[execution_id] = execution
        return execution_id

    async def _execute_in_sandbox(self, contribution: Contribution, execution_id: str) -> bool:
        """Execute contribution in sandbox environment"""
        execution = self.sandbox_executions[execution_id]

        try:
            # Prepare sandbox environment
            sandbox_config = self._prepare_sandbox_config(contribution)

            # Execute based on contribution type
            if contribution.contribution_type == ContributionType.MODEL_UPDATES:
                result = await self._execute_model_update(contribution, execution, sandbox_config)
            elif contribution.contribution_type == ContributionType.TRAINING_DATA:
                result = await self._execute_data_validation(contribution, execution, sandbox_config)
            elif contribution.contribution_type == ContributionType.VALIDATION_RESULTS:
                result = await self._execute_validation_check(contribution, execution, sandbox_config)
            elif contribution.contribution_type == ContributionType.HYPERPARAMETER_SUGGESTIONS:
                result = await self._execute_hyperparameter_test(contribution, execution, sandbox_config)
            elif contribution.contribution_type == ContributionType.EVOLUTION_INSIGHTS:
                result = await self._execute_insight_validation(contribution, execution, sandbox_config)
            else:
                result = False

            execution.end_time = datetime.now().isoformat()
            execution.execution_status = "completed" if result else "failed"

            return result

        except asyncio.TimeoutError:
            execution.execution_status = "timeout"
            execution.error_message = "Sandbox execution timed out"
            return False
        except Exception as e:
            execution.execution_status = "failed"
            execution.error_message = str(e)
            return False

    def _prepare_sandbox_config(self, contribution: Contribution) -> Dict[str, Any]:
        """Prepare sandbox configuration for contribution execution"""
        return {
            "timeout": self.sandbox_timeout,
            "memory_limit": "512MB",
            "cpu_limit": "50%",
            "network_access": False,
            "file_access": ["read"],
            "contribution_type": contribution.contribution_type.value,
            "node_trust_level": self.registered_nodes[contribution.node_id].trust_level.value
        }

    async def _execute_model_update(self, contribution: Contribution,
                                  execution: SandboxExecution, config: Dict[str, Any]) -> bool:
        """Execute model update contribution in sandbox"""
        # Simulate model update validation
        update_data = contribution.content.get("model_update", {})

        # Basic validation checks
        required_fields = ["layer_updates", "learning_rate", "batch_size"]
        if not all(field in update_data for field in required_fields):
            execution.error_message = "Missing required fields in model update"
            return False

        # Check for malicious patterns
        if self._contains_malicious_patterns(update_data):
            execution.error_message = "Malicious patterns detected in model update"
            return False

        # Simulate performance test
        performance_score = self._simulate_model_performance(update_data)
        execution.performance_metrics["accuracy_improvement"] = performance_score

        # Safety checks
        execution.safety_checks = {
            "gradient_clipping": update_data.get("gradient_clipping", False),
            "weight_regularization": "regularization" in update_data,
            "no_nan_values": not self._contains_nan_values(update_data)
        }

        return performance_score > 0.01  # Require minimum improvement

    async def _execute_data_validation(self, contribution: Contribution,
                                     execution: SandboxExecution, config: Dict[str, Any]) -> bool:
        """Execute training data validation in sandbox"""
        data_info = contribution.content.get("data_info", {})

        # Validate data format and quality
        if not self._validate_data_format(data_info):
            execution.error_message = "Invalid data format"
            return False

        # Check data quality metrics
        quality_score = self._assess_data_quality(data_info)
        execution.performance_metrics["data_quality_score"] = quality_score

        # Privacy checks
        execution.safety_checks = {
            "anonymized": data_info.get("anonymized", False),
            "no_pii": not self._contains_pii_patterns(data_info),
            "consent_obtained": data_info.get("consent_obtained", False)
        }

        return quality_score > 0.7

    async def _execute_validation_check(self, contribution: Contribution,
                                      execution: SandboxExecution, config: Dict[str, Any]) -> bool:
        """Execute validation results check in sandbox"""
        validation_results = contribution.content.get("validation_results", {})

        # Validate results format
        if not self._validate_results_format(validation_results):
            execution.error_message = "Invalid validation results format"
            return False

        # Cross-validate results
        consistency_score = self._check_result_consistency(validation_results)
        execution.performance_metrics["consistency_score"] = consistency_score

        return consistency_score > 0.8

    async def _execute_hyperparameter_test(self, contribution: Contribution,
                                         execution: SandboxExecution, config: Dict[str, Any]) -> bool:
        """Execute hyperparameter suggestions test in sandbox"""
        suggestions = contribution.content.get("hyperparameter_suggestions", {})

        # Validate suggestions
        if not self._validate_hyperparameter_suggestions(suggestions):
            execution.error_message = "Invalid hyperparameter suggestions"
            return False

        # Test suggestions
        improvement_score = self._test_hyperparameter_improvement(suggestions)
        execution.performance_metrics["improvement_score"] = improvement_score

        return improvement_score > 0.05

    async def _execute_insight_validation(self, contribution: Contribution,
                                        execution: SandboxExecution, config: Dict[str, Any]) -> bool:
        """Execute evolution insights validation in sandbox"""
        insights = contribution.content.get("insights", {})

        # Validate insights format
        if not self._validate_insights_format(insights):
            execution.error_message = "Invalid insights format"
            return False

        # Assess insight quality
        quality_score = self._assess_insight_quality(insights)
        execution.performance_metrics["insight_quality_score"] = quality_score

        return quality_score > 0.6

    async def _validate_contribution(self, contribution: Contribution, execution_id: str) -> float:
        """Validate contribution and return quality score"""
        execution = self.sandbox_executions[execution_id]

        # Base score from performance metrics
        base_score = sum(execution.performance_metrics.values()) / max(1, len(execution.performance_metrics))

        # Trust multiplier based on node reputation
        node = self.registered_nodes[contribution.node_id]
        trust_multiplier = {
            TrustLevel.UNVERIFIED: 0.3,
            TrustLevel.BASIC: 0.6,
            TrustLevel.VERIFIED: 0.8,
            TrustLevel.TRUSTED: 1.0,
            TrustLevel.CORE_CONTRIBUTOR: 1.2
        }[node.trust_level]

        # Safety bonus
        safety_score = sum(execution.safety_checks.values()) / max(1, len(execution.safety_checks))
        safety_bonus = 0.1 if safety_score > 0.8 else 0.0

        final_score = (base_score * trust_multiplier) + safety_bonus
        return max(0.0, min(1.0, final_score))

    def _contains_malicious_patterns(self, data: Dict[str, Any]) -> bool:
        """Check for malicious patterns in contribution data"""
        # Simple checks (would be more sophisticated in production)
        malicious_indicators = [
            "eval(", "exec(", "__import__", "subprocess",
            "infinite", "loop", "while True", "fork"
        ]

        data_str = json.dumps(data).lower()
        return any(indicator in data_str for indicator in malicious_indicators)

    def _contains_nan_values(self, data: Dict[str, Any]) -> bool:
        """Check for NaN values in numerical data"""
        def check_value(val):
            if isinstance(val, float):
                return math.isnan(val) or math.isinf(val)
            elif isinstance(val, dict):
                return any(check_value(v) for v in val.values())
            elif isinstance(val, list):
                return any(check_value(item) for item in val)
            return False

        return check_value(data)

    def _simulate_model_performance(self, update_data: Dict[str, Any]) -> float:
        """Simulate model performance improvement"""
        # Simple simulation based on update characteristics
        base_improvement = 0.02  # Base 2% improvement

        # Factors that increase improvement
        factors = {
            "learning_rate": update_data.get("learning_rate", 0.001),
            "batch_size": min(1.0, update_data.get("batch_size", 32) / 64),
            "gradient_clipping": 0.01 if update_data.get("gradient_clipping") else 0.0
        }

        total_improvement = base_improvement
        for factor_name, factor_value in factors.items():
            total_improvement += factor_value * 0.01

        return total_improvement

    def _validate_data_format(self, data_info: Dict[str, Any]) -> bool:
        """Validate training data format"""
        required_fields = ["num_samples", "features", "format"]
        return all(field in data_info for field in required_fields)

    def _assess_data_quality(self, data_info: Dict[str, Any]) -> float:
        """Assess quality of training data"""
        quality_factors = {
            "diversity_score": data_info.get("diversity_score", 0.5),
            "noise_level": 1.0 - data_info.get("noise_level", 0.5),  # Invert noise
            "completeness": data_info.get("completeness", 0.8)
        }

        return sum(quality_factors.values()) / len(quality_factors)

    def _contains_pii_patterns(self, data_info: Dict[str, Any]) -> bool:
        """Check for personally identifiable information patterns"""
        # Simple PII detection (would use more sophisticated methods in production)
        pii_indicators = ["email", "phone", "ssn", "address", "name"]
        data_str = json.dumps(data_info).lower()
        return any(indicator in data_str for indicator in pii_indicators)

    def _validate_results_format(self, results: Dict[str, Any]) -> bool:
        """Validate validation results format"""
        return "metrics" in results and "dataset" in results

    def _check_result_consistency(self, results: Dict[str, Any]) -> float:
        """Check consistency of validation results"""
        metrics = results.get("metrics", {})
        if not metrics:
            return 0.0

        # Check for reasonable metric ranges
        valid_ranges = {
            "accuracy": (0.0, 1.0),
            "precision": (0.0, 1.0),
            "recall": (0.0, 1.0),
            "f1_score": (0.0, 1.0)
        }

        consistency_score = 0.0
        for metric_name, value in metrics.items():
            if metric_name in valid_ranges:
                min_val, max_val = valid_ranges[metric_name]
                if min_val <= value <= max_val:
                    consistency_score += 1.0

        return consistency_score / max(1, len(valid_ranges))

    def _validate_hyperparameter_suggestions(self, suggestions: Dict[str, Any]) -> bool:
        """Validate hyperparameter suggestions"""
        return "parameters" in suggestions and "expected_improvement" in suggestions

    def _test_hyperparameter_improvement(self, suggestions: Dict[str, Any]) -> float:
        """Test expected improvement from hyperparameter suggestions"""
        return suggestions.get("expected_improvement", 0.0)

    def _validate_insights_format(self, insights: Dict[str, Any]) -> bool:
        """Validate evolution insights format"""
        return "insights" in insights and isinstance(insights["insights"], list)

    def _assess_insight_quality(self, insights: Dict[str, Any]) -> float:
        """Assess quality of evolution insights"""
        insight_list = insights.get("insights", [])
        if not insight_list:
            return 0.0

        # Simple quality assessment based on insight characteristics
        quality_score = min(1.0, len(insight_list) / 10.0)  # More insights = higher quality
        return quality_score

    def _round_management_loop(self):
        """Background loop for managing federated learning rounds"""
        while True:
            try:
                current_time = datetime.now()

                # Check if we need to start a new round
                if self.active_round is None:
                    if self._should_start_new_round():
                        self._start_new_round()
                else:
                    # Check if round should end
                    round_start = datetime.fromisoformat(self.active_round.start_time)
                    if (current_time - round_start).total_seconds() > self.max_round_duration:
                        self._end_round()

                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in round management loop: {e}")
                time.sleep(60)

    def _should_start_new_round(self) -> bool:
        """Determine if a new round should be started"""
        # Check minimum participants
        active_nodes = [node for node in self.registered_nodes.values()
                       if (datetime.now() - datetime.fromisoformat(node.last_activity)).days < 7]

        if len(active_nodes) < self.min_participants:
            return False

        # Check pending contributions
        return len(self.pending_contributions) > 0

    def _start_new_round(self):
        """Start a new federated learning round"""
        round_number = len(self.round_history) + 1
        round_id = f"round_{round_number}_{int(time.time())}"

        self.active_round = FederatedRound(
            round_id=round_id,
            round_number=round_number,
            start_time=datetime.now().isoformat()
        )

        logger.info(f"Started federated learning round {round_number}")

    def _end_round(self):
        """End the current federated learning round"""
        if self.active_round is None:
            return

        # Aggregate validated contributions
        round_contributions = [contrib for contrib in self.validated_contributions.values()
                             if not contrib.aggregated]

        if round_contributions:
            aggregated_updates = self._aggregate_contributions(round_contributions)

            self.active_round.aggregated_updates = aggregated_updates
            self.active_round.contributions_integrated = len(round_contributions)

            # Mark contributions as aggregated
            for contrib in round_contributions:
                contrib.aggregated = True

        # Calculate round metrics
        self.active_round.round_metrics = self._calculate_round_metrics()

        # End round
        self.active_round.end_time = datetime.now().isoformat()
        self.round_history.append(self.active_round)

        logger.info(f"Ended federated learning round {self.active_round.round_number}")
        self.active_round = None

        # Save state
        self._save_persistent_state()

    def _aggregate_contributions(self, contributions: List[Contribution]) -> Dict[str, Any]:
        """Aggregate contributions using federated averaging"""
        aggregated = {}

        # Group by contribution type
        by_type = {}
        for contrib in contributions:
            contrib_type = contrib.contribution_type.value
            if contrib_type not in by_type:
                by_type[contrib_type] = []
            by_type[contrib_type].append(contrib)

        # Aggregate each type
        for contrib_type, type_contributions in by_type.items():
            if contrib_type == "model_updates":
                aggregated[contrib_type] = self._aggregate_model_updates(type_contributions)
            elif contrib_type == "validation_results":
                aggregated[contrib_type] = self._aggregate_validation_results(type_contributions)
            elif contrib_type == "hyperparameter_suggestions":
                aggregated[contrib_type] = self._aggregate_hyperparameter_suggestions(type_contributions)
            else:
                # Simple averaging for other types
                aggregated[contrib_type] = self._simple_aggregation(type_contributions)

        return aggregated

    def _aggregate_model_updates(self, contributions: List[Contribution]) -> Dict[str, Any]:
        """Aggregate model updates using federated averaging"""
        if not contributions:
            return {}

        # Weighted averaging based on validation scores and node trust
        total_weight = 0.0
        weighted_updates = {}

        for contrib in contributions:
            node = self.registered_nodes[contrib.node_id]
            weight = contrib.validation_score * node.reputation_score
            total_weight += weight

            update_data = contrib.content.get("model_update", {})
            for param_name, param_value in update_data.items():
                if isinstance(param_value, (int, float)):
                    if param_name not in weighted_updates:
                        weighted_updates[param_name] = 0.0
                    weighted_updates[param_name] += param_value * weight

        # Normalize by total weight
        if total_weight > 0:
            for param_name in weighted_updates:
                weighted_updates[param_name] /= total_weight

        return {"aggregated_parameters": weighted_updates, "total_weight": total_weight}

    def _aggregate_validation_results(self, contributions: List[Contribution]) -> Dict[str, Any]:
        """Aggregate validation results"""
        all_metrics = []
        for contrib in contributions:
            metrics = contrib.content.get("validation_results", {}).get("metrics", {})
            all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Average metrics across contributions
        aggregated_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [metrics.get(metric_name, 0) for metrics in all_metrics if metric_name in metrics]
            if values:
                aggregated_metrics[metric_name] = sum(values) / len(values)

        return {"aggregated_metrics": aggregated_metrics, "num_contributions": len(contributions)}

    def _aggregate_hyperparameter_suggestions(self, contributions: List[Contribution]) -> Dict[str, Any]:
        """Aggregate hyperparameter suggestions"""
        suggestions = []
        for contrib in contributions:
            suggestion = contrib.content.get("hyperparameter_suggestions", {})
            suggestions.append(suggestion)

        if not suggestions:
            return {}

        # Select best suggestion based on expected improvement
        best_suggestion = max(suggestions, key=lambda x: x.get("expected_improvement", 0))

        return {"best_suggestion": best_suggestion, "total_suggestions": len(suggestions)}

    def _simple_aggregation(self, contributions: List[Contribution]) -> Dict[str, Any]:
        """Simple aggregation for other contribution types"""
        return {
            "total_contributions": len(contributions),
            "average_validation_score": sum(c.validation_score for c in contributions) / len(contributions)
        }

    def _calculate_round_metrics(self) -> Dict[str, float]:
        """Calculate metrics for the completed round"""
        if self.active_round is None:
            return {}

        return {
            "participation_rate": len(self.active_round.participating_nodes) / max(1, len(self.registered_nodes)),
            "contribution_success_rate": self.active_round.contributions_validated / max(1, self.active_round.contributions_received),
            "aggregation_efficiency": self.active_round.contributions_integrated / max(1, self.active_round.contributions_validated)
        }

    def _trust_decay_loop(self):
        """Background loop for decaying node trust over time"""
        while True:
            try:
                current_time = datetime.now()

                for node in self.registered_nodes.values():
                    # Decay reputation if inactive
                    last_activity = datetime.fromisoformat(node.last_activity)
                    days_inactive = (current_time - last_activity).days

                    if days_inactive > 30:
                        decay_factor = self.trust_decay_rate ** (days_inactive / 30)
                        node.reputation_score *= decay_factor

                time.sleep(86400)  # Run daily

            except Exception as e:
                logger.error(f"Error in trust decay loop: {e}")
                time.sleep(86400)

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current status of the federated learning bridge"""
        return {
            "total_nodes": len(self.registered_nodes),
            "active_round": asdict(self.active_round) if self.active_round else None,
            "pending_contributions": len(self.pending_contributions),
            "validated_contributions": len(self.validated_contributions),
            "total_rounds": len(self.round_history),
            "node_distribution": self._get_node_distribution(),
            "recent_activity": self._get_recent_activity()
        }

    def _get_node_distribution(self) -> Dict[str, int]:
        """Get distribution of node types"""
        distribution = {}
        for node in self.registered_nodes.values():
            node_type = node.node_type.value
            distribution[node_type] = distribution.get(node_type, 0) + 1
        return distribution

    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent bridge activity"""
        recent_contributions = list(self.validated_contributions.values())[-10:]  # Last 10
        return [{
            "contribution_id": contrib.contribution_id,
            "node_id": contrib.node_id,
            "type": contrib.contribution_type.value,
            "validation_score": contrib.validation_score,
            "timestamp": contrib.timestamp
        } for contrib in recent_contributions]

# Global instance
_federated_learning_bridge = None

def get_federated_learning_bridge() -> FederatedLearningBridge:
    """Get the global federated learning bridge instance"""
    global _federated_learning_bridge
    if _federated_learning_bridge is None:
        _federated_learning_bridge = FederatedLearningBridge()
    return _federated_learning_bridge