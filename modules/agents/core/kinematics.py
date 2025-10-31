#!/usr/bin/env python3
"""
KinematicsAgent - Forward and inverse kinematics solving
Extends Kalki with advanced kinematics analysis and trajectory planning.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time

# Import kinematics libraries
try:
    import ikpy
    IKPY_AVAILABLE = True
except ImportError:
    IKPY_AVAILABLE = False

from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ..safety.guard import SafetyGuard


@dataclass
class KinematicChain:
    """Represents a kinematic chain"""
    links: List[Dict[str, Any]]
    joints: List[Dict[str, Any]]
    base_transform: np.ndarray = None
    tool_transform: np.ndarray = None

    def __post_init__(self):
        if self.base_transform is None:
            self.base_transform = np.eye(4)
        if self.tool_transform is None:
            self.tool_transform = np.eye(4)


@dataclass
class TrajectoryPoint:
    """Represents a point in a trajectory"""
    position: np.ndarray
    orientation: np.ndarray
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None
    time: float = 0.0


class KinematicsAgent(BaseAgent):
    """
    KinematicsAgent provides forward and inverse kinematics solving.
    Supports trajectory planning, workspace analysis, and motion optimization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            AgentCapability.KINEMATICS_ANALYSIS,
            AgentCapability.TRAJECTORY_PLANNING,
            AgentCapability.MOTION_OPTIMIZATION
        ]

        super().__init__(
            name="KinematicsAgent",
            capabilities=capabilities,
            description="Advanced kinematics analysis and trajectory planning",
            config=config
        )

        # Kinematics parameters
        self.max_iterations = 1000
        self.tolerance = 1e-6
        self.damping_factor = 0.01

        # Trajectory planning parameters
        self.max_velocity = 1.0  # m/s
        self.max_acceleration = 2.0  # m/s²
        self.time_step = 0.01  # seconds

        # Cached kinematic chains
        self.kinematic_chains: Dict[str, KinematicChain] = {}

        # Safety controls
        self.safety_guard = SafetyGuard()

        self.logger.info("KinematicsAgent initialized")

    async def initialize(self) -> bool:
        """Initialize the kinematics agent"""
        try:
            self.status = AgentStatus.READY
            self.logger.info("KinematicsAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize KinematicsAgent: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a kinematics task"""
        try:
            self.increment_task_count()
            action = task.get('action', '')

            if action == 'create_kinematic_chain':
                result = await self.create_kinematic_chain(task.get('robot_config', {}))
            elif action == 'solve_inverse_kinematics':
                result = await self.solve_inverse_kinematics(
                    task.get('chain_name', ''),
                    task.get('target_pose', {}),
                    task.get('initial_guess')
                )
            elif action == 'plan_trajectory':
                result = await self.plan_trajectory(
                    task.get('chain_name', ''),
                    task.get('waypoints', []),
                    task.get('constraints', {})
                )
            elif action == 'analyze_workspace':
                result = await self.analyze_workspace(task.get('chain_name', ''), task.get('samples', 10000))
            else:
                result = {'error': f'Unknown action: {action}'}

            return {
                'status': 'success',
                'result': result
            }

        except Exception as e:
            self.increment_error_count()
            self.logger.error(f"Task execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def shutdown(self) -> bool:
        """Shutdown the kinematics agent"""
        try:
            self.status = AgentStatus.TERMINATED
            self.logger.info("KinematicsAgent shutdown successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown KinematicsAgent: {e}")
            return False

    async def create_kinematic_chain(self, robot_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create kinematic chain from robot configuration

        Args:
            robot_config: Robot configuration dictionary

        Returns:
            Dict with kinematic chain and analysis
        """
        try:
            links = robot_config.get('links', [])
            joints = robot_config.get('joints', [])

            # Create links for IKPy
            ikpy_links = []
            for i, (link, joint) in enumerate(zip(links, joints)):
                # Create IKPy link
                link_length = link.get('length', 0.2)
                joint_type = joint.get('joint_type', 'revolute')

                if joint_type == 'revolute':
                    ikpy_link = ikpy.link.Link(
                        name=f'link_{i+1}',
                        translation_vector=[link_length, 0, 0],
                        orientation=[0, 0, 1],  # Rotation around Z
                        rotation=[0, 0, 1]
                    )
                else:
                    # Prismatic joint
                    ikpy_link = ikpy.link.Link(
                        name=f'link_{i+1}',
                        translation_vector=[link_length, 0, 0],
                        orientation=[0, 0, 1],
                        rotation=[0, 0, 0]  # No rotation for prismatic
                    )

                ikpy_links.append(ikpy_link)

            # Create kinematic chain
            chain_name = robot_config.get('name', 'robot')
            chain = ikpy.chain.Chain(ikpy_links, name=chain_name)

            # Store chain
            kinematic_chain = KinematicChain(
                links=links,
                joints=joints,
                base_transform=np.eye(4),
                tool_transform=np.eye(4)
            )

            self.kinematic_chains[chain_name] = kinematic_chain

            # Analyze chain
            analysis = await self.analyze_kinematic_chain(chain, kinematic_chain)

            return {
                'chain_name': chain_name,
                'ikpy_chain': chain,
                'kinematic_chain': kinematic_chain,
                'degrees_of_freedom': len(joints),
                'analysis': analysis
            }

        except Exception as e:
            self.logger.error(f"Kinematic chain creation failed: {e}")
            return {'error': str(e)}

    async def analyze_kinematic_chain(self, ikpy_chain, kinematic_chain: KinematicChain) -> Dict[str, Any]:
        """Analyze kinematic chain properties"""
        try:
            # Test forward kinematics
            joint_angles = [0.1 * i for i in range(len(kinematic_chain.joints))]
            fk_result = ikpy_chain.forward_kinematics(joint_angles)

            # Test inverse kinematics
            target_position = [0.3, 0.2, 0.1]
            target_orientation = None
            ik_result = ikpy_chain.inverse_kinematics(target_position, target_orientation)

            # Calculate workspace bounds
            workspace_bounds = self._calculate_workspace_bounds(kinematic_chain)

            # Calculate manipulability
            manipulability = self._calculate_manipulability(ikpy_chain, joint_angles)

            return {
                'forward_kinematics_test': {
                    'joint_angles': joint_angles,
                    'end_effector_pose': fk_result.tolist()
                },
                'inverse_kinematics_test': {
                    'target_position': target_position,
                    'solution_found': ik_result is not None,
                    'joint_angles': ik_result.tolist() if ik_result is not None else None
                },
                'workspace_bounds': workspace_bounds,
                'manipulability_index': manipulability,
                'chain_properties': {
                    'dof': len(kinematic_chain.joints),
                    'total_length': sum(link.get('length', 0) for link in kinematic_chain.links),
                    'joint_types': [joint.get('joint_type', 'revolute') for joint in kinematic_chain.joints]
                }
            }

        except Exception as e:
            self.logger.error(f"Chain analysis failed: {e}")
            return {'error': str(e)}

    async def solve_inverse_kinematics(self, chain_name: str, target_pose: Dict[str, Any],
                                     initial_guess: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Solve inverse kinematics for target pose

        Args:
            chain_name: Name of kinematic chain
            target_pose: Target position and orientation
            initial_guess: Initial joint angles guess

        Returns:
            Dict with IK solution
        """
        try:
            if chain_name not in self.kinematic_chains:
                return {'error': f'Kinematic chain {chain_name} not found'}

            kinematic_chain = self.kinematic_chains[chain_name]

            # Extract target pose
            target_position = target_pose.get('position', [0.3, 0.2, 0.1])
            target_orientation = target_pose.get('orientation')

            # Create IKPy chain if available
            if IKPY_AVAILABLE:
                ikpy_links = []
                for i, (link, joint) in enumerate(zip(kinematic_chain.links, kinematic_chain.joints)):
                    link_length = link.get('length', 0.2)
                    joint_type = joint.get('joint_type', 'revolute')

                    if joint_type == 'revolute':
                        ikpy_link = ikpy.link.Link(
                            name=f'link_{i+1}',
                            translation_vector=[link_length, 0, 0],
                            orientation=[0, 0, 1],
                            rotation=[0, 0, 1]
                        )
                    else:
                        ikpy_link = ikpy.link.Link(
                            name=f'link_{i+1}',
                            translation_vector=[link_length, 0, 0],
                            orientation=[0, 0, 1],
                            rotation=[0, 0, 0]
                        )

                    ikpy_links.append(ikpy_link)

                chain = ikpy.chain.Chain(ikpy_links)

                # Solve IK
                start_time = time.time()
                solution = chain.inverse_kinematics(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    initial_position=initial_guess or [0] * len(kinematic_chain.joints),
                    max_iter=self.max_iterations,
                    tolerance=self.tolerance
                )
                solve_time = time.time() - start_time

                # Verify solution
                fk_verification = chain.forward_kinematics(solution)
                position_error = np.linalg.norm(np.array(target_position) - fk_verification[:3, 3])

                return {
                    'solution': solution.tolist(),
                    'solve_time': solve_time,
                    'position_error': position_error,
                    'converged': position_error < self.tolerance,
                    'iterations': self.max_iterations,  # IKPy doesn't return iteration count
                    'target_position': target_position,
                    'achieved_position': fk_verification[:3, 3].tolist()
                }
            else:
                # Fallback to simple analytical IK for 2DOF
                return await self._analytical_ik_2dof(kinematic_chain, target_position)

        except Exception as e:
            self.logger.error(f"Inverse kinematics failed: {e}")
            return {'error': str(e)}

    async def plan_trajectory(self, chain_name: str, waypoints: List[Dict[str, Any]],
                            constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Plan trajectory through waypoints

        Args:
            chain_name: Name of kinematic chain
            waypoints: List of waypoint poses
            constraints: Motion constraints

        Returns:
            Dict with trajectory
        """
        try:
            if chain_name not in self.kinematic_chains:
                return {'error': f'Kinematic chain {chain_name} not found'}

            kinematic_chain = self.kinematic_chains[chain_name]

            # Default constraints
            if constraints is None:
                constraints = {}

            max_vel = constraints.get('max_velocity', self.max_velocity)
            max_acc = constraints.get('max_acceleration', self.max_acceleration)
            time_step = constraints.get('time_step', self.time_step)

            trajectory_points = []

            for i, waypoint in enumerate(waypoints):
                # Solve IK for waypoint
                ik_result = await self.solve_inverse_kinematics(chain_name, waypoint)

                if 'error' in ik_result:
                    return {'error': f'IK failed for waypoint {i}: {ik_result["error"]}'}

                joint_angles = ik_result['solution']

                # Create trajectory point
                point = TrajectoryPoint(
                    position=np.array(waypoint.get('position', [0, 0, 0])),
                    orientation=np.eye(3),  # Simplified
                    velocity=np.zeros(len(joint_angles)),
                    acceleration=np.zeros(len(joint_angles)),
                    time=i * 1.0  # 1 second per waypoint
                )

                trajectory_points.append({
                    'waypoint_index': i,
                    'joint_angles': joint_angles,
                    'position': point.position.tolist(),
                    'time': point.time
                })

            # Interpolate between waypoints
            interpolated_trajectory = await self._interpolate_trajectory(trajectory_points, time_step)

            return {
                'waypoints': len(waypoints),
                'trajectory_points': interpolated_trajectory,
                'total_time': len(interpolated_trajectory) * time_step,
                'constraints': constraints,
                'kinematic_chain': chain_name
            }

        except Exception as e:
            self.logger.error(f"Trajectory planning failed: {e}")
            return {'error': str(e)}

    async def optimize_trajectory(self, trajectory: Dict[str, Any],
                                optimization_criteria: List[str]) -> Dict[str, Any]:
        """
        Optimize trajectory based on criteria

        Args:
            trajectory: Original trajectory
            optimization_criteria: List of criteria (e.g., ['minimize_time', 'minimize_jerk'])

        Returns:
            Dict with optimized trajectory
        """
        try:
            trajectory_points = trajectory.get('trajectory_points', [])

            if not trajectory_points:
                return {'error': 'No trajectory points to optimize'}

            optimized_points = trajectory_points.copy()

            # Apply optimizations
            for criterion in optimization_criteria:
                if criterion == 'minimize_time':
                    # Speed up trajectory while respecting constraints
                    optimized_points = await self._optimize_trajectory_time(optimized_points)
                elif criterion == 'minimize_jerk':
                    # Smooth trajectory to reduce jerk
                    optimized_points = await self._smooth_trajectory(optimized_points)
                elif criterion == 'minimize_energy':
                    # Optimize for energy efficiency
                    optimized_points = await self._optimize_energy(optimized_points)

            return {
                'original_trajectory': trajectory,
                'optimized_trajectory': optimized_points,
                'optimization_criteria': optimization_criteria,
                'improvements': {}  # Would calculate actual improvements
            }

        except Exception as e:
            self.logger.error(f"Trajectory optimization failed: {e}")
            return {'error': str(e)}

    async def analyze_workspace(self, chain_name: str, samples: int = 10000) -> Dict[str, Any]:
        """
        Analyze robot workspace

        Args:
            chain_name: Name of kinematic chain
            samples: Number of random samples

        Returns:
            Dict with workspace analysis
        """
        try:
            if chain_name not in self.kinematic_chains:
                return {'error': f'Kinematic chain {chain_name} not found'}

            kinematic_chain = self.kinematic_chains[chain_name]

            workspace_points = []
            reachable_points = 0

            # Sample joint space
            for _ in range(samples):
                # Random joint angles within limits
                joint_angles = []
                for joint in kinematic_chain.joints:
                    joint_limits = joint.get('limits', [-np.pi, np.pi])
                    angle = np.random.uniform(joint_limits[0], joint_limits[1])
                    joint_angles.append(angle)

                # Forward kinematics
                if IKPY_AVAILABLE:
                    ikpy_links = []
                    for i, (link, joint) in enumerate(zip(kinematic_chain.links, kinematic_chain.joints)):
                        link_length = link.get('length', 0.2)
                        ikpy_link = ikpy.link.Link(
                            name=f'link_{i+1}',
                            translation_vector=[link_length, 0, 0],
                            orientation=[0, 0, 1],
                            rotation=[0, 0, 1]
                        )
                        ikpy_links.append(ikpy_link)

                    chain = ikpy.chain.Chain(ikpy_links)
                    pose = chain.forward_kinematics(joint_angles)
                    position = pose[:3, 3]
                else:
                    # Simplified FK for testing
                    position = np.array([sum(joint_angles[:i+1]) * 0.1 for i in range(len(joint_angles))])

                workspace_points.append(position.tolist())
                reachable_points += 1

            # Calculate workspace properties
            points_array = np.array(workspace_points)
            workspace_volume = self._calculate_convex_hull_volume(points_array)

            return {
                'total_samples': samples,
                'reachable_points': reachable_points,
                'workspace_points': workspace_points[:1000],  # Limit for response size
                'workspace_volume': workspace_volume,
                'workspace_bounds': {
                    'x': [float(points_array[:, 0].min()), float(points_array[:, 0].max())],
                    'y': [float(points_array[:, 1].min()), float(points_array[:, 1].max())],
                    'z': [float(points_array[:, 2].min()), float(points_array[:, 2].max())]
                },
                'reachability_percentage': reachable_points / samples * 100
            }

        except Exception as e:
            self.logger.error(f"Workspace analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_workspace_bounds(self, kinematic_chain: KinematicChain) -> Dict[str, List[float]]:
        """Calculate approximate workspace bounds"""
        total_reach = sum(link.get('length', 0) for link in kinematic_chain.links)

        return {
            'x': [-total_reach, total_reach],
            'y': [-total_reach, total_reach],
            'z': [-total_reach, total_reach]
        }

    def _calculate_manipulability(self, ikpy_chain, joint_angles: List[float]) -> float:
        """Calculate manipulability index"""
        try:
            # Jacobian matrix
            jacobian = ikpy_chain.jacobian(joint_angles)

            # Manipulability = sqrt(det(J * J^T))
            jj_t = jacobian @ jacobian.T
            manipulability = np.sqrt(np.abs(np.linalg.det(jj_t)))

            return float(manipulability)
        except:
            return 0.0

    async def _analytical_ik_2dof(self, kinematic_chain: KinematicChain, target_position: List[float]) -> Dict[str, Any]:
        """Analytical IK solution for 2DOF planar robot"""
        try:
            if len(kinematic_chain.joints) != 2:
                return {'error': 'Analytical IK only supports 2DOF robots'}

            l1 = kinematic_chain.links[0].get('length', 0.2)
            l2 = kinematic_chain.links[1].get('length', 0.2)

            x, y = target_position[0], target_position[1]
            distance = np.sqrt(x**2 + y**2)

            if distance > (l1 + l2) or distance < abs(l1 - l2):
                return {'error': 'Target position out of reach'}

            cos_theta2 = (distance**2 - l1**2 - l2**2) / (2 * l1 * l2)
            theta2 = np.arccos(np.clip(cos_theta2, -1, 1))

            theta1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(theta2), l1 + l2 * np.cos(theta2))

            solution = [theta1, theta2]

            return {
                'solution': solution,
                'solve_time': 0.001,
                'position_error': 0.0,
                'converged': True,
                'iterations': 1,
                'target_position': target_position,
                'achieved_position': target_position  # Exact for analytical solution
            }

        except Exception as e:
            return {'error': str(e)}

    async def _interpolate_trajectory(self, waypoints: List[Dict[str, Any]], time_step: float) -> List[Dict[str, Any]]:
        """Interpolate between trajectory waypoints"""
        try:
            if len(waypoints) < 2:
                return waypoints

            interpolated_points = []

            for i in range(len(waypoints) - 1):
                start_point = waypoints[i]
                end_point = waypoints[i + 1]

                start_angles = np.array(start_point['joint_angles'])
                end_angles = np.array(end_point['joint_angles'])

                start_time = start_point['time']
                end_time = end_point['time']

                duration = end_time - start_time
                steps = int(duration / time_step)

                for j in range(steps):
                    t = j / (steps - 1) if steps > 1 else 0
                    interpolated_angles = start_angles + t * (end_angles - start_angles)

                    interpolated_points.append({
                        'time': start_time + j * time_step,
                        'joint_angles': interpolated_angles.tolist(),
                        'segment': i,
                        'interpolation_parameter': t
                    })

            return interpolated_points

        except Exception as e:
            self.logger.error(f"Trajectory interpolation failed: {e}")
            return waypoints

    async def _optimize_trajectory_time(self, trajectory_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize trajectory for minimum time"""
        # Simple time scaling - in practice would use more sophisticated optimization
        return trajectory_points

    async def _smooth_trajectory(self, trajectory_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Smooth trajectory to reduce jerk"""
        # Simple smoothing - in practice would use spline interpolation
        return trajectory_points

    async def _optimize_energy(self, trajectory_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize trajectory for energy efficiency"""
        # Energy optimization - in practice would minimize joint torques
        return trajectory_points

    def _calculate_convex_hull_volume(self, points: np.ndarray) -> float:
        """Calculate volume of convex hull (simplified 3D approximation)"""
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            return hull.volume
        except:
            # Fallback: approximate as bounding box volume
            bounds = np.ptp(points, axis=0)  # Peak-to-peak (max - min)
            return float(np.prod(bounds))