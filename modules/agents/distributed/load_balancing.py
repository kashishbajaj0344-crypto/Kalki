"""
Load Balancing Agent (Phase 8)
==============================

Implements intelligent load balancing using consistent hashing,
least connections, and predictive algorithms for optimal distribution.
"""

import asyncio
import hashlib
import time
import random
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.LoadBalancing")


@dataclass
class ServerInstance:
    """Represents a server instance in the load balancing pool"""
    id: str
    address: str
    port: int
    weight: int = 1
    current_connections: int = 0
    max_connections: int = 1000
    health_score: float = 1.0  # 0.0 to 1.0
    last_health_check: float = 0
    response_time: float = 0.1


@dataclass
class LoadBalancingRequest:
    """Load balancing request with metadata"""
    client_id: str
    request_type: str
    priority: int = 1
    estimated_load: float = 1.0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class LoadBalancingAgent(BaseAgent):
    """
    Intelligent load balancing agent using multiple algorithms.
    Implements consistent hashing, least connections, and predictive balancing.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="LoadBalancingAgent",
            capabilities=[
                AgentCapability.LOAD_BALANCING,
                AgentCapability.OPTIMIZATION,
                AgentCapability.SELF_HEALING
            ],
            description="Intelligent load balancing with multiple algorithms",
            config=config or {}
        )

        # Load balancing configuration
        self.algorithm = self.config.get('algorithm', 'consistent_hashing')  # or 'least_connections', 'round_robin'
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.max_connections_per_server = self.config.get('max_connections_per_server', 1000)

        # Server pool
        self.servers: Dict[str, ServerInstance] = {}
        self.healthy_servers: Set[str] = set()

        # Consistent hashing ring
        self.hash_ring: List[Tuple[int, str]] = []
        self.virtual_nodes_per_server = 100

        # Integration hooks for cross-agent cooperation
        self.pre_balance_hooks: List[Callable] = []
        self.post_balance_hooks: List[Callable] = []

        # AI-driven balancing components
        self.node_reputation_scores = defaultdict(lambda: 1.0)
        self.task_similarity_cache = {}
        self.performance_history = deque(maxlen=1000)

        # Advanced ML-based task similarity system
        self.task_similarity_model = None
        self.task_feature_scaler = StandardScaler()
        self.task_feature_pca = PCA(n_components=10)
        self.task_performance_history = defaultdict(list)  # Track performance per task type
        self.similarity_learning_enabled = True
        self.task_embedding_history = deque(maxlen=5000)  # Store task embeddings for learning

    async def initialize(self) -> bool:
        """Initialize load balancing system"""
        try:
            logger.info("LoadBalancingAgent initializing with consistent hashing")

            # Initialize with default servers if none provided
            if not self.servers:
                await self._initialize_default_servers()

            # Build consistent hashing ring
            await self._build_hash_ring()

            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())

            logger.info(f"LoadBalancingAgent initialized with {len(self.servers)} servers")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize LoadBalancingAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute load balancing operations"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "balance_request":
            return await self._balance_request(params)
        elif action == "add_server":
            return await self._add_server(params)
        elif action == "remove_server":
            return await self._remove_server(params)
        elif action == "get_stats":
            return await self._get_load_stats(params)
        elif action == "rebalance":
            return await self._rebalance_pool(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _balance_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Balance an incoming request to the optimal server"""
        try:
            request = LoadBalancingRequest(
                client_id=params.get('client_id', f"client_{random.randint(1000, 9999)}"),
                request_type=params.get('request_type', 'http'),
                priority=params.get('priority', 1),
                estimated_load=params.get('estimated_load', 1.0)
            )

            # Select server based on algorithm
            if self.algorithm == 'consistent_hashing':
                server_id = await self._consistent_hash_balance(request)
            elif self.algorithm == 'least_connections':
                server_id = await self._least_connections_balance(request)
            elif self.algorithm == 'predictive':
                server_id = await self._predictive_balance(request)
            elif self.algorithm == 'adaptive_ai':
                server_id = await self._adaptive_ai_balance(request)
            else:
                server_id = await self._round_robin_balance(request)

            if not server_id or server_id not in self.healthy_servers:
                return {"status": "error", "error": "No healthy servers available"}

            # Update server load
            server = self.servers[server_id]
            server.current_connections += 1

            # Record request
            self.request_history.append({
                'timestamp': request.timestamp,
                'client_id': request.client_id,
                'server_id': server_id,
                'request_type': request.request_type,
                'load': request.estimated_load
            })

            self.total_requests += 1
            self.balanced_requests += 1

            return {
                "status": "success",
                "assigned_server": {
                    "id": server_id,
                    "address": server.address,
                    "port": server.port,
                    "current_connections": server.current_connections
                },
                "algorithm_used": self.algorithm,
                "request_id": f"{request.client_id}_{int(request.timestamp)}"
            }

        except Exception as e:
            logger.exception(f"Request balancing error: {e}")
            return {"status": "error", "error": str(e)}

    async def _consistent_hash_balance(self, request: LoadBalancingRequest) -> Optional[str]:
        """Balance using consistent hashing"""
        if not self.hash_ring:
            return None

        # Hash the client ID
        key_hash = int(hashlib.md5(request.client_id.encode()).hexdigest(), 16)

        # Find the server on the hash ring
        for ring_hash, server_id in self.hash_ring:
            if key_hash <= ring_hash:
                if server_id in self.healthy_servers:
                    return server_id

        # Wrap around to first server
        if self.hash_ring:
            return self.hash_ring[0][1] if self.hash_ring[0][1] in self.healthy_servers else None

        return None

    async def _least_connections_balance(self, request: LoadBalancingRequest) -> Optional[str]:
        """Balance using least connections algorithm"""
        healthy_servers = [(sid, self.servers[sid]) for sid in self.healthy_servers]

        if not healthy_servers:
            return None

        # Find server with least connections
        min_connections = min(s.current_connections for _, s in healthy_servers)
        candidates = [sid for sid, s in healthy_servers if s.current_connections == min_connections]

        # Random selection among candidates for load distribution
        return random.choice(candidates) if candidates else None

    async def _predictive_balance(self, request: LoadBalancingRequest) -> Optional[str]:
        """Balance using predictive load analysis"""
        if not self.healthy_servers:
            return None

        # Calculate predicted load for each server
        server_scores = {}
        for server_id in self.healthy_servers:
            predicted_load = await self._predict_server_load(server_id, request.estimated_load)
            server_scores[server_id] = predicted_load

        # Select server with lowest predicted load
        best_server = min(server_scores.items(), key=lambda x: x[1])
        return best_server[0]

    async def _round_robin_balance(self, request: LoadBalancingRequest) -> Optional[str]:
        """Simple round-robin balancing"""
        healthy_list = list(self.healthy_servers)
        if not healthy_list:
            return None

        # Use request count for round-robin
        index = self.total_requests % len(healthy_list)
        return healthy_list[index]

    async def _predict_server_load(self, server_id: str, additional_load: float) -> float:
        """Predict future load for a server"""
        server = self.servers[server_id]
        load_history = list(self.server_load_history[server_id])

        if len(load_history) < 5:
            # Not enough history, use current connections
            return server.current_connections + additional_load

        # Simple linear regression on recent load
        recent_loads = load_history[-10:]
        if len(recent_loads) >= 2:
            # Calculate trend
            x = np.arange(len(recent_loads))
            y = np.array(recent_loads)
            slope = np.polyfit(x, y, 1)[0]

            # Predict next load
            predicted_base = recent_loads[-1] + slope
        else:
            predicted_base = recent_loads[-1]

        return max(0, predicted_base + additional_load)

    async def _add_server(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new server to the pool"""
        try:
            server_id = params.get('server_id')
            address = params.get('address')
            port = params.get('port', 80)
            weight = params.get('weight', 1)
            max_connections = params.get('max_connections', self.max_connections_per_server)

            if not all([server_id, address]):
                return {"status": "error", "error": "server_id and address are required"}

            if server_id in self.servers:
                return {"status": "error", "error": f"Server {server_id} already exists"}

            # Create server instance
            server = ServerInstance(
                id=server_id,
                address=address,
                port=port,
                weight=weight,
                max_connections=max_connections
            )

            self.servers[server_id] = server

            # Add to healthy servers (assume healthy initially)
            self.healthy_servers.add(server_id)

            # Rebuild hash ring
            await self._build_hash_ring()

            logger.info(f"Added server {server_id} to load balancing pool")

            return {
                "status": "success",
                "server_added": server_id,
                "total_servers": len(self.servers),
                "healthy_servers": len(self.healthy_servers)
            }

        except Exception as e:
            logger.exception(f"Add server error: {e}")
            return {"status": "error", "error": str(e)}

    async def _remove_server(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a server from the pool"""
        try:
            server_id = params.get('server_id')

            if server_id not in self.servers:
                return {"status": "error", "error": f"Server {server_id} not found"}

            # Remove from pools
            self.healthy_servers.discard(server_id)
            del self.servers[server_id]

            # Rebuild hash ring
            await self._build_hash_ring()

            logger.info(f"Removed server {server_id} from load balancing pool")

            return {
                "status": "success",
                "server_removed": server_id,
                "total_servers": len(self.servers),
                "healthy_servers": len(self.healthy_servers)
            }

        except Exception as e:
            logger.exception(f"Remove server error: {e}")
            return {"status": "error", "error": str(e)}

    async def _build_hash_ring(self):
        """Build consistent hashing ring"""
        self.hash_ring = []

        for server_id, server in self.servers.items():
            if server_id in self.healthy_servers:
                # Add virtual nodes for this server
                for i in range(self.virtual_nodes_per_server * server.weight):
                    key = f"{server_id}:{i}"
                    hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
                    self.hash_ring.append((hash_value, server_id))

        # Sort the ring
        self.hash_ring.sort()

    async def _health_monitoring_loop(self):
        """Continuous health monitoring of servers"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.exception(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _perform_health_checks(self):
        """Perform health checks on all servers"""
        for server_id, server in self.servers.items():
            try:
                # Simulate health check (would be actual HTTP/TCP check)
                is_healthy = await self._check_server_health(server)

                if is_healthy and server_id not in self.healthy_servers:
                    self.healthy_servers.add(server_id)
                    logger.info(f"Server {server_id} is now healthy")
                    await self._build_hash_ring()
                elif not is_healthy and server_id in self.healthy_servers:
                    self.healthy_servers.remove(server_id)
                    logger.warning(f"Server {server_id} is unhealthy")
                    await self._build_hash_ring()

                # Update health score and response time
                server.last_health_check = time.time()
                server.health_score = 1.0 if is_healthy else 0.0

            except Exception as e:
                logger.exception(f"Health check failed for {server_id}: {e}")

    async def _check_server_health(self, server: ServerInstance) -> bool:
        """Check if a server is healthy"""
        # Simulate health check with some failure probability
        # In real implementation, this would make actual network requests
        failure_rate = 0.05  # 5% failure rate
        return random.random() > failure_rate

    async def _initialize_default_servers(self):
        """Initialize with default server pool"""
        default_servers = [
            {"id": "server_01", "address": "10.0.0.1", "port": 8080},
            {"id": "server_02", "address": "10.0.0.2", "port": 8080},
            {"id": "server_03", "address": "10.0.0.3", "port": 8080},
        ]

        for server_config in default_servers:
            server = ServerInstance(**server_config)
            self.servers[server.id] = server
            self.healthy_servers.add(server.id)

    async def _get_load_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive load balancing statistics"""
        server_stats = {}
        for server_id, server in self.servers.items():
            load_history = list(self.server_load_history[server_id])
            avg_load = np.mean(load_history) if load_history else 0

            server_stats[server_id] = {
                "address": f"{server.address}:{server.port}",
                "healthy": server_id in self.healthy_servers,
                "current_connections": server.current_connections,
                "max_connections": server.max_connections,
                "health_score": server.health_score,
                "average_load": avg_load,
                "response_time": server.response_time
            }

        return {
            "status": "success",
            "algorithm": self.algorithm,
            "total_servers": len(self.servers),
            "healthy_servers": len(self.healthy_servers),
            "total_requests": self.total_requests,
            "balanced_requests": self.balanced_requests,
            "server_stats": server_stats
        }

    async def _rebalance_pool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Rebalance the server pool"""
        try:
            # Rebuild hash ring
            await self._build_hash_ring()

            # Reset connection counts if requested
            if params.get('reset_connections', False):
                for server in self.servers.values():
                    server.current_connections = 0

            logger.info("Server pool rebalanced")

            return {
                "status": "success",
                "message": "Server pool rebalanced",
                "healthy_servers": len(self.healthy_servers),
                "total_servers": len(self.servers)
            }

        except Exception as e:
            logger.exception(f"Rebalance error: {e}")
            return {"status": "error", "error": str(e)}

    # Integration hook management

    def add_pre_balance_hook(self, hook: Callable):
        """Add a hook to run before load balancing decisions."""
        self.pre_balance_hooks.append(hook)

    def add_post_balance_hook(self, hook: Callable):
        """Add a hook to run after load balancing decisions."""
        self.post_balance_hooks.append(hook)

    # AI-driven load balancing

    async def _adaptive_ai_balance(self, request: LoadBalancingRequest) -> Optional[str]:
        """AI-driven adaptive load balancing using reputation and performance history."""
        try:
            available_servers = list(self.healthy_servers)

            if not available_servers:
                return None

            # Run pre-balance hooks for external intelligence
            hook_results = []
            for hook in self.pre_balance_hooks:
                try:
                    result = await hook({
                        "request": request,
                        "available_servers": available_servers
                    })
                    hook_results.append(result)
                except Exception as e:
                    logger.warning(f"Pre-balance hook failed: {e}")

            # Extract preferred server from hooks
            preferred_server = None
            for result in hook_results:
                if isinstance(result, dict) and "preferred_node" in result:
                    preferred_server = result["preferred_node"]
                    break

            if preferred_server and preferred_server in available_servers:
                server_id = preferred_server
            else:
                # Use adaptive task similarity for intelligent server selection
                similar_servers = self.get_similar_task_servers({
                    "request_type": request.request_type,
                    "priority": request.priority,
                    "estimated_load": request.estimated_load
                })

                if similar_servers:
                    # Boost scores for servers that have handled similar tasks
                    server_scores = {}
                    for server_id in available_servers:
                        server = self.servers[server_id]
                        base_score = 1.0 / (server.current_connections + 1)

                        # Factor in reputation
                        reputation_bonus = self.node_reputation_scores[server_id]

                        # Factor in recent performance
                        performance_factor = self._calculate_performance_factor(server_id)

                        # Factor in task similarity bonus
                        similarity_bonus = 1.0
                        for similar_server, similarity in similar_servers[:3]:  # Top 3 similar
                            if similar_server == server_id:
                                similarity_bonus = 1.0 + (similarity * 0.5)  # Up to 50% bonus
                                break

                        server_scores[server_id] = base_score * reputation_bonus * performance_factor * similarity_bonus

                    server_id = max(server_scores, key=server_scores.get)
                else:
                    # Fallback to reputation-based selection
                    server_scores = {}
                    for server_id in available_servers:
                        server = self.servers[server_id]
                        base_score = 1.0 / (server.current_connections + 1)

                        # Factor in reputation
                        reputation_bonus = self.node_reputation_scores[server_id]

                        # Factor in recent performance
                        performance_factor = self._calculate_performance_factor(server_id)

                        server_scores[server_id] = base_score * reputation_bonus * performance_factor

                    server_id = max(server_scores, key=server_scores.get)

            # Record the assignment
            self._record_assignment(server_id, request)

            # Run post-balance hooks
            assignment = {
                "server_id": server_id,
                "request": request,
                "timestamp": request.timestamp
            }

            for hook in self.post_balance_hooks:
                try:
                    await hook(assignment)
                except Exception as e:
                    logger.warning(f"Post-balance hook failed: {e}")

            return server_id

        except Exception as e:
            logger.exception(f"AI-driven balancing failed: {e}")
            # Fallback to round-robin
            return await self._round_robin_balance(request)

    def _calculate_performance_factor(self, server_id: str) -> float:
        """Calculate performance factor based on recent history."""
        try:
            history = self.server_load_history[server_id]
            if not history:
                return 1.0

            recent_loads = list(history)[-10:]  # Last 10 measurements
            if not recent_loads:
                return 1.0

            avg_load = sum(recent_loads) / len(recent_loads)
            # Lower load is better, so invert the factor
            return max(0.1, 2.0 - avg_load)  # Range: 0.1 to 2.0

        except Exception as e:
            logger.exception(f"Failed to calculate performance factor for {server_id}: {e}")
            return 1.0

    def _record_assignment(self, server_id: str, request: LoadBalancingRequest, performance_score: float = None):
        """Record load balancing assignment for learning with performance tracking."""
        try:
            task_record = {
                "server_id": server_id,
                "request_type": request.request_type,
                "priority": request.priority,
                "estimated_load": request.estimated_load,
                "timestamp": request.timestamp
            }

            self.request_history.append(task_record)

            # Update task similarity cache
            self._update_task_similarity_cache(task_record)

            # Learn from performance if provided
            if performance_score is not None:
                self._learn_from_task_performance(task_record, server_id, performance_score)

            # Update server load
            if server_id in self.servers:
                self.servers[server_id].current_connections += 1

        except Exception as e:
            logger.exception(f"Failed to record assignment: {e}")

    def update_node_reputation(self, node_id: str, success: bool):
        """Update node reputation based on task success/failure."""
        try:
            current_reputation = self.node_reputation_scores[node_id]

            if success:
                # Increase reputation (with diminishing returns)
                self.node_reputation_scores[node_id] = min(2.0, current_reputation + 0.1)
            else:
                # Decrease reputation
                self.node_reputation_scores[node_id] = max(0.1, current_reputation - 0.2)

        except Exception as e:
            logger.exception(f"Failed to update reputation for {node_id}: {e}")

    # Task similarity caching for intelligent scheduling

    def _get_task_similarity(self, task_a: dict, task_b: dict) -> float:
        """Calculate similarity between two tasks based on type, priority, and load."""
        try:
            # Type similarity (exact match = 1.0, different = 0.0)
            type_sim = 1.0 if task_a.get("request_type") == task_b.get("request_type") else 0.0

            # Priority similarity (closer priorities are more similar)
            priority_a = task_a.get("priority", 1)
            priority_b = task_b.get("priority", 1)
            priority_sim = 1.0 / (1.0 + abs(priority_a - priority_b))

            # Load similarity (closer loads are more similar)
            load_a = task_a.get("estimated_load", 1.0)
            load_b = task_b.get("estimated_load", 1.0)
            load_sim = 1.0 / (1.0 + abs(load_a - load_b))

            # Weighted combination
            return (0.5 * type_sim) + (0.3 * priority_sim) + (0.2 * load_sim)

        except Exception as e:
            logger.exception(f"Failed to calculate task similarity: {e}")
            return 0.0

    def _update_task_similarity_cache(self, new_task: dict):
        """Update the task similarity cache with a new task using advanced ML similarity."""
        try:
            cache_key = self._get_task_cache_key(new_task)
            if cache_key not in self.task_similarity_cache:
                self.task_similarity_cache[cache_key] = []

            # Update embedding history for learning
            self._update_task_embedding_history(new_task)

            # Calculate similarities with recent tasks using adaptive similarity
            recent_tasks = list(self.request_history)[-20:]  # Last 20 tasks
            similarities = []

            for recent_task in recent_tasks:
                similarity = self._get_adaptive_similarity(new_task, recent_task)
                if similarity > 0.3:  # Lower threshold for ML-based similarity
                    similarities.append({
                        "task": recent_task,
                        "similarity": similarity,
                        "server_id": recent_task.get("server_id")
                    })

            # Sort by similarity and keep top matches
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            self.task_similarity_cache[cache_key] = similarities[:10]  # Keep top 10

        except Exception as e:
            logger.exception(f"Failed to update task similarity cache: {e}")

    def _get_task_cache_key(self, task: dict) -> str:
        """Generate a cache key for a task."""
        try:
            key_parts = [
                str(task.get("request_type", "unknown")),
                str(task.get("priority", 1)),
                f"{task.get('estimated_load', 1.0):.2f}"
            ]
            return "|".join(key_parts)
        except Exception as e:
            return "unknown|1|1.00"

    def get_similar_task_servers(self, task: dict) -> list:
        """Get servers that have successfully handled similar tasks."""
        try:
            cache_key = self._get_task_cache_key(task)
            similar_tasks = self.task_similarity_cache.get(cache_key, [])

            # Group by server and calculate average similarity
            server_similarities = defaultdict(list)
            for similar_task in similar_tasks:
                server_id = similar_task["server_id"]
                server_similarities[server_id].append(similar_task["similarity"])

            # Return servers sorted by average similarity
            server_scores = {}
            for server_id, similarities in server_similarities.items():
                server_scores[server_id] = sum(similarities) / len(similarities)

            return sorted(server_scores.items(), key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.exception(f"Failed to get similar task servers: {e}")
            return []

    # Advanced ML-based task similarity system

    def _extract_task_features(self, task: dict) -> np.ndarray:
        """Extract multi-dimensional feature vector from task for ML similarity."""
        try:
            features = []

            # Categorical features (one-hot encoded)
            request_type = task.get("request_type", "unknown")
            type_features = self._encode_categorical_feature(request_type, "request_type")
            features.extend(type_features)

            # Numerical features
            priority = task.get("priority", 1)
            estimated_load = task.get("estimated_load", 1.0)

            # Normalize numerical features
            features.extend([priority / 10.0, estimated_load / 10.0])  # Scale to 0-1 range

            # Time-based features
            timestamp = task.get("timestamp", time.time())
            hour_of_day = (timestamp % 86400) / 3600  # Hour 0-24
            day_of_week = ((timestamp // 86400) + 4) % 7  # Day 0-6 (0=Monday)
            features.extend([hour_of_day / 24.0, day_of_week / 7.0])

            # Performance history features (if available)
            task_type = request_type
            if task_type in self.task_performance_history:
                perf_history = self.task_performance_history[task_type][-10:]  # Last 10 executions
                if perf_history:
                    avg_performance = np.mean(perf_history)
                    perf_std = np.std(perf_history)
                    features.extend([avg_performance, perf_std])
                else:
                    features.extend([0.5, 0.1])  # Default values
            else:
                features.extend([0.5, 0.1])  # Default values

            return np.array(features)

        except Exception as e:
            logger.exception(f"Failed to extract task features: {e}")
            return np.array([0.0] * 20)  # Return zero vector on error

    def _encode_categorical_feature(self, value: str, feature_name: str) -> List[float]:
        """Encode categorical feature using learned embeddings or simple hashing."""
        try:
            # Simple approach: use hash-based encoding for now
            # In production, this would use learned embeddings
            hash_value = int(hashlib.md5(f"{feature_name}:{value}".encode()).hexdigest(), 16)
            # Create a 5-dimensional encoding
            encoding = []
            for i in range(5):
                encoding.append((hash_value >> (i * 8)) & 0xFF)  # Extract bytes
            # Normalize to 0-1
            max_val = 255.0
            return [x / max_val for x in encoding]

        except Exception as e:
            logger.exception(f"Failed to encode categorical feature: {e}")
            return [0.0] * 5

    def _get_ml_task_similarity(self, task_a: dict, task_b: dict) -> float:
        """Calculate ML-based task similarity using feature vectors and cosine similarity."""
        try:
            if not self.similarity_learning_enabled:
                return self._get_task_similarity(task_a, task_b)  # Fallback to rule-based

            # Extract feature vectors
            features_a = self._extract_task_features(task_a)
            features_b = self._extract_task_features(task_b)

            # Scale features if we have enough data
            if len(self.task_embedding_history) > 50:
                try:
                    # Fit scaler on historical data
                    historical_features = np.array(list(self.task_embedding_history))
                    self.task_feature_scaler.fit(historical_features)

                    features_a = self.task_feature_scaler.transform(features_a.reshape(1, -1)).flatten()
                    features_b = self.task_feature_scaler.transform(features_b.reshape(1, -1)).flatten()

                    # Apply PCA for dimensionality reduction
                    features_a = self.task_feature_pca.fit_transform(features_a.reshape(1, -1)).flatten()
                    features_b = self.task_feature_pca.fit_transform(features_b.reshape(1, -1)).flatten()

                except Exception as e:
                    logger.warning(f"Feature scaling failed, using raw features: {e}")

            # Calculate cosine similarity
            similarity = cosine_similarity(
                features_a.reshape(1, -1),
                features_b.reshape(1, -1)
            )[0][0]

            # Ensure similarity is in [0, 1] range
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

            return similarity

        except Exception as e:
            logger.exception(f"ML-based similarity calculation failed: {e}")
            return self._get_task_similarity(task_a, task_b)  # Fallback

    def _update_task_embedding_history(self, task: dict):
        """Update the task embedding history for learning."""
        try:
            if self.similarity_learning_enabled:
                features = self._extract_task_features(task)
                self.task_embedding_history.append(features)

        except Exception as e:
            logger.exception(f"Failed to update task embedding history: {e}")

    def _learn_from_task_performance(self, task: dict, server_id: str, performance_score: float):
        """Learn from task execution performance to improve future similarity calculations."""
        try:
            task_type = task.get("request_type", "unknown")

            # Store performance score
            self.task_performance_history[task_type].append(performance_score)

            # Keep only recent performance data
            if len(self.task_performance_history[task_type]) > 100:
                self.task_performance_history[task_type] = self.task_performance_history[task_type][-100:]

            # Update server reputation based on performance
            if performance_score > 0.8:  # Good performance
                self.update_node_reputation(server_id, True)
            elif performance_score < 0.4:  # Poor performance
                self.update_node_reputation(server_id, False)

        except Exception as e:
            logger.exception(f"Failed to learn from task performance: {e}")

    def _get_adaptive_similarity(self, task_a: dict, task_b: dict) -> float:
        """Get adaptive similarity using both rule-based and ML approaches."""
        try:
            # Use ML similarity if available and confident
            ml_similarity = self._get_ml_task_similarity(task_a, task_b)

            # Use rule-based similarity as baseline
            rule_similarity = self._get_task_similarity(task_a, task_b)

            # Adaptive weighting based on learning confidence
            learning_confidence = min(1.0, len(self.task_embedding_history) / 1000.0)

            # Combine similarities with adaptive weighting
            combined_similarity = (learning_confidence * ml_similarity) + ((1 - learning_confidence) * rule_similarity)

            return combined_similarity

        except Exception as e:
            logger.exception(f"Adaptive similarity calculation failed: {e}")
            return self._get_task_similarity(task_a, task_b)  # Fallback

    async def shutdown(self) -> bool:
        """Shutdown the load balancing agent"""
        try:
            logger.info("LoadBalancingAgent shutting down")

            # Clear server pools
            self.servers.clear()
            self.healthy_servers.clear()
            self.hash_ring.clear()

            # Clear request history
            self.request_history.clear()
            self.server_load_history.clear()

            logger.info("LoadBalancingAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during LoadBalancingAgent shutdown: {e}")
            return False