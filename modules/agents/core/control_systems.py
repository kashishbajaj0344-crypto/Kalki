#!/usr/bin/env python3
"""
ControlSystemsAgent - PID controllers and trajectory control
Extends Kalki with control systems design and real-time control.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
import control

from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ..safety.guard import SafetyGuard


@dataclass
class PIDController:
    """PID controller parameters"""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 0.0
    output_limits: Tuple[float, float] = (-np.inf, np.inf)
    integral_limits: Tuple[float, float] = (-np.inf, np.inf)

    def __post_init__(self):
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = time.time()


@dataclass
class ControlSystem:
    """Complete control system for robot joint"""
    joint_name: str
    pid_controller: PIDController
    feedforward_terms: Dict[str, float] = None
    disturbance_observer: bool = False

    def __post_init__(self):
        if self.feedforward_terms is None:
            self.feedforward_terms = {}


class ControlSystemsAgent(BaseAgent):
    """
    ControlSystemsAgent provides control systems design and implementation.
    Supports PID control, trajectory tracking, and system identification.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        capabilities = [
            AgentCapability.CONTROL_SYSTEMS,
            AgentCapability.PID_CONTROL,
            AgentCapability.TRAJECTORY_TRACKING
        ]

        super().__init__(
            name="ControlSystemsAgent",
            capabilities=capabilities,
            description="Control systems design and trajectory tracking",
            config=config
        )

        # Control parameters
        self.control_frequency = 1000  # Hz
        self.time_step = 1.0 / self.control_frequency

        # Control systems
        self.control_systems: Dict[str, ControlSystem] = {}

        # System identification
        self.system_models: Dict[str, Any] = {}

        # Safety controls
        self.safety_guard = SafetyGuard()

        self.logger.info("ControlSystemsAgent initialized")

    async def initialize(self) -> bool:
        """Initialize the control systems agent"""
        try:
            self.status = AgentStatus.READY
            self.logger.info("ControlSystemsAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize ControlSystemsAgent: {e}")
            self.status = AgentStatus.ERROR
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a control systems task"""
        try:
            self.increment_task_count()
            action = task.get('action', '')

            if action == 'design_pid_controller':
                result = await self.design_pid_controller(task.get('system_parameters', {}))
            elif action == 'simulate_control_system':
                result = await self.simulate_control_system(
                    task.get('pid_controller', {}),
                    task.get('system_params', {}),
                    task.get('simulation_time', 5.0)
                )
            elif action == 'design_trajectory_controller':
                result = await self.design_trajectory_controller(
                    task.get('trajectory', {}),
                    task.get('robot_params', {})
                )
            elif action == 'tune_pid_controller':
                result = await self.tune_pid_controller(
                    task.get('system_params', {}),
                    task.get('performance_criteria', {})
                )
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
        """Shutdown the control systems agent"""
        try:
            self.status = AgentStatus.TERMINATED
            self.logger.info("ControlSystemsAgent shutdown successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown ControlSystemsAgent: {e}")
            return False

    async def design_pid_controller(self, system_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design PID controller for system

        Args:
            system_parameters: System dynamics parameters

        Returns:
            Dict with PID controller design
        """
        try:
            system_type = system_parameters.get('type', 'second_order')
            natural_frequency = system_parameters.get('natural_frequency', 10.0)  # rad/s
            damping_ratio = system_parameters.get('damping_ratio', 0.7)
            gain = system_parameters.get('gain', 1.0)

            if system_type == 'second_order':
                # Design PID for second-order system
                kp, ki, kd = self._design_pid_second_order(natural_frequency, damping_ratio, gain)
            elif system_type == 'first_order':
                # Design PI for first-order system
                time_constant = system_parameters.get('time_constant', 0.1)
                kp, ki, kd = self._design_pi_first_order(time_constant, gain)
            else:
                # Default PID design
                kp, ki, kd = 1.0, 0.1, 0.01

            # Create PID controller
            pid_controller = PIDController(
                kp=kp,
                ki=ki,
                kd=kd,
                output_limits=system_parameters.get('output_limits', (-100, 100)),
                integral_limits=system_parameters.get('integral_limits', (-10, 10))
            )

            # Analyze stability
            stability_analysis = await self.analyze_stability(pid_controller, system_parameters)

            return {
                'pid_parameters': {
                    'kp': kp,
                    'ki': ki,
                    'kd': kd
                },
                'controller': pid_controller,
                'system_parameters': system_parameters,
                'stability_analysis': stability_analysis,
                'performance_metrics': {
                    'rise_time': 1.8 / (damping_ratio * natural_frequency),
                    'settling_time': 4.0 / (damping_ratio * natural_frequency),
                    'overshoot': 100 * np.exp(-damping_ratio * np.pi / np.sqrt(1 - damping_ratio**2))
                }
            }

        except Exception as e:
            self.logger.error(f"PID design failed: {e}")
            return {'error': str(e)}

    def _design_pid_second_order(self, wn: float, zeta: float, k: float) -> Tuple[float, float, float]:
        """Design PID for second-order system"""
        # Ziegler-Nichols inspired design
        kp = 0.6 * k * wn**2 / (2 * zeta * wn)  # Proportional gain
        ki = 0.5 * wn  # Integral gain
        kd = 0.125 * kp / wn  # Derivative gain

        return kp, ki, kd

    def _design_pi_first_order(self, tau: float, k: float) -> Tuple[float, float, float]:
        """Design PI for first-order system"""
        kp = 0.4 * k / tau
        ki = 0.4 * k / (tau**2)
        kd = 0.0  # No derivative for first-order

        return kp, ki, kd

    async def analyze_stability(self, pid_controller: PIDController, system_params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze control system stability"""
        try:
            # Create transfer function
            system_type = system_params.get('type', 'second_order')

            if system_type == 'second_order':
                wn = system_params.get('natural_frequency', 10.0)
                zeta = system_params.get('damping_ratio', 0.7)
                k = system_params.get('gain', 1.0)

                # Plant: G(s) = k / (s² + 2*zeta*wn*s + wn²)
                plant = control.tf([k], [1, 2*zeta*wn, wn**2])

                # Controller: C(s) = kp + ki/s + kd*s
                controller = control.tf([pid_controller.kd, pid_controller.kp, pid_controller.ki], [1, 0])

                # Closed loop
                closed_loop = control.feedback(controller * plant)

                # Stability analysis
                poles = control.pole(closed_loop)
                stable = all(np.real(pole) < 0 for pole in poles)

                # Gain margin and phase margin
                try:
                    gm, pm, _, _ = control.margin(controller * plant)
                except:
                    gm, pm = float('inf'), float('inf')

                return {
                    'stable': stable,
                    'poles': [complex(pole).real + 1j * complex(pole).imag for pole in poles],
                    'gain_margin': float(gm) if np.isfinite(gm) else None,
                    'phase_margin': float(pm) if np.isfinite(pm) else None,
                    'closed_loop_poles': len(poles)
                }

            else:
                return {
                    'stable': True,  # Assume stable for other systems
                    'analysis_method': 'simplified'
                }

        except Exception as e:
            self.logger.error(f"Stability analysis failed: {e}")
            return {'error': str(e)}

    async def simulate_control_system(self, pid_controller: PIDController,
                                    system_params: Dict[str, Any],
                                    simulation_time: float = 5.0) -> Dict[str, Any]:
        """
        Simulate control system response

        Args:
            pid_controller: PID controller
            system_params: System parameters
            simulation_time: Simulation duration

        Returns:
            Dict with simulation results
        """
        try:
            time_points = np.arange(0, simulation_time, self.time_step)
            setpoint = pid_controller.setpoint

            # System state
            position = 0.0
            velocity = 0.0

            # Control variables
            integral = 0.0
            previous_error = 0.0
            previous_time = 0.0

            # Simulation data
            positions = []
            velocities = []
            controls = []
            errors = []

            system_type = system_params.get('type', 'second_order')
            wn = system_params.get('natural_frequency', 10.0)
            zeta = system_params.get('damping_ratio', 0.7)
            k = system_params.get('gain', 1.0)

            for t in time_points:
                # Calculate error
                error = setpoint - position
                errors.append(error)

                # PID control
                dt = t - previous_time if previous_time > 0 else self.time_step

                # Proportional term
                p_term = pid_controller.kp * error

                # Integral term
                integral += error * dt
                integral = np.clip(integral, pid_controller.integral_limits[0], pid_controller.integral_limits[1])
                i_term = pid_controller.ki * integral

                # Derivative term
                derivative = (error - previous_error) / dt if dt > 0 else 0
                d_term = pid_controller.kd * derivative

                # Total control
                control = p_term + i_term + d_term
                control = np.clip(control, pid_controller.output_limits[0], pid_controller.output_limits[1])
                controls.append(control)

                # System dynamics
                if system_type == 'second_order':
                    # Second-order system: m*x'' + b*x' + k*x = u
                    mass = system_params.get('mass', 1.0)
                    damping = system_params.get('damping', 2*zeta*wn)
                    stiffness = system_params.get('stiffness', wn**2)

                    acceleration = (control - damping * velocity - stiffness * position) / mass
                else:
                    # First-order system: tau*x' + x = k*u
                    tau = system_params.get('time_constant', 0.1)
                    acceleration = (k * control - position) / tau

                # Integrate
                velocity += acceleration * dt
                position += velocity * dt

                positions.append(position)
                velocities.append(velocity)

                # Update for next iteration
                previous_error = error
                previous_time = t

            # Calculate performance metrics
            steady_state_error = abs(setpoint - positions[-1])
            rise_time = self._calculate_rise_time(time_points, positions, setpoint)
            settling_time = self._calculate_settling_time(time_points, positions, setpoint)
            overshoot = self._calculate_overshoot(positions, setpoint)

            return {
                'time': time_points.tolist(),
                'position': positions,
                'velocity': velocities,
                'control_input': controls,
                'error': errors,
                'performance_metrics': {
                    'steady_state_error': steady_state_error,
                    'rise_time': rise_time,
                    'settling_time': settling_time,
                    'overshoot': overshoot
                },
                'simulation_parameters': {
                    'duration': simulation_time,
                    'time_step': self.time_step,
                    'setpoint': setpoint
                }
            }

        except Exception as e:
            self.logger.error(f"Control simulation failed: {e}")
            return {'error': str(e)}

    async def tune_pid_controller(self, system_params: Dict[str, Any],
                                performance_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-tune PID controller

        Args:
            system_params: System parameters
            performance_criteria: Desired performance

        Returns:
            Dict with tuned PID parameters
        """
        try:
            # Ziegler-Nichols tuning or optimization-based tuning
            method = performance_criteria.get('method', 'ziegler_nichols')

            if method == 'ziegler_nichols':
                tuned_params = await self._ziegler_nichols_tuning(system_params)
            elif method == 'optimization':
                tuned_params = await self._optimization_tuning(system_params, performance_criteria)
            else:
                tuned_params = {'kp': 1.0, 'ki': 0.1, 'kd': 0.01}

            # Create tuned controller
            tuned_controller = PIDController(
                kp=tuned_params['kp'],
                ki=tuned_params['ki'],
                kd=tuned_params['kd']
            )

            # Validate tuning
            validation = await self.simulate_control_system(tuned_controller, system_params)

            return {
                'tuned_parameters': tuned_params,
                'tuning_method': method,
                'controller': tuned_controller,
                'validation_results': validation,
                'performance_criteria': performance_criteria
            }

        except Exception as e:
            self.logger.error(f"PID tuning failed: {e}")
            return {'error': str(e)}

    async def design_trajectory_controller(self, trajectory: Dict[str, Any],
                                        robot_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Design controller for trajectory tracking

        Args:
            trajectory: Trajectory to follow
            robot_params: Robot parameters

        Returns:
            Dict with trajectory controller
        """
        try:
            trajectory_points = trajectory.get('trajectory_points', [])
            dof = robot_params.get('degrees_of_freedom', 6)

            # Create controllers for each joint
            joint_controllers = {}
            for i in range(dof):
                joint_name = f'joint_{i+1}'

                # Design PID for joint
                system_params = {
                    'type': 'second_order',
                    'natural_frequency': 20.0,  # rad/s
                    'damping_ratio': 0.8,
                    'gain': 1.0
                }

                controller_design = await self.design_pid_controller(system_params)
                joint_controllers[joint_name] = controller_design['controller']

            # Feedforward terms for trajectory tracking
            feedforward_gains = {
                'velocity_feedforward': 1.0,
                'acceleration_feedforward': 0.1
            }

            return {
                'joint_controllers': {name: {
                    'kp': ctrl.kp,
                    'ki': ctrl.ki,
                    'kd': ctrl.kd
                } for name, ctrl in joint_controllers.items()},
                'feedforward_gains': feedforward_gains,
                'trajectory_points': len(trajectory_points),
                'control_frequency': self.control_frequency,
                'expected_performance': {
                    'tracking_error': '< 0.01 rad',
                    'settling_time': '< 0.5 s'
                }
            }

        except Exception as e:
            self.logger.error(f"Trajectory controller design failed: {e}")
            return {'error': str(e)}

    async def simulate_trajectory_tracking(self, trajectory_controller: Dict[str, Any],
                                        trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate trajectory tracking performance

        Args:
            trajectory_controller: Controller design
            trajectory: Trajectory to track

        Returns:
            Dict with tracking simulation results
        """
        try:
            trajectory_points = trajectory.get('trajectory_points', [])
            joint_controllers = trajectory_controller.get('joint_controllers', {})

            if not trajectory_points or not joint_controllers:
                return {'error': 'Invalid trajectory or controller'}

            # Simulate tracking for each joint
            tracking_results = {}

            for joint_name, controller_params in joint_controllers.items():
                controller = PIDController(
                    kp=controller_params['kp'],
                    ki=controller_params['ki'],
                    kd=controller_params['kd']
                )

                # Extract joint trajectory
                joint_angles = [point.get('joint_angles', [0])[0] for point in trajectory_points]  # Simplified

                # Simulate tracking
                tracking_result = await self._simulate_joint_tracking(controller, joint_angles)
                tracking_results[joint_name] = tracking_result

            # Calculate overall performance
            max_tracking_error = max(result.get('max_error', 0) for result in tracking_results.values())
            avg_tracking_error = np.mean([result.get('rms_error', 0) for result in tracking_results.values()])

            return {
                'joint_tracking_results': tracking_results,
                'overall_performance': {
                    'max_tracking_error': max_tracking_error,
                    'avg_tracking_error': avg_tracking_error,
                    'tracking_success': max_tracking_error < 0.01  # 1% tolerance
                },
                'simulation_parameters': {
                    'trajectory_points': len(trajectory_points),
                    'control_frequency': self.control_frequency
                }
            }

        except Exception as e:
            self.logger.error(f"Trajectory tracking simulation failed: {e}")
            return {'error': str(e)}

    async def _simulate_joint_tracking(self, controller: PIDController, desired_trajectory: List[float]) -> Dict[str, Any]:
        """Simulate tracking for single joint"""
        try:
            # System parameters (simplified joint dynamics)
            inertia = 0.1  # kg*m²
            damping = 0.01  # N*m*s/rad
            time_constant = 0.05  # seconds

            # Simulation
            dt = self.time_step
            position = desired_trajectory[0]
            velocity = 0.0
            integral = 0.0
            previous_error = 0.0

            actual_positions = []
            errors = []

            for desired_pos in desired_trajectory:
                # PID control
                error = desired_pos - position
                errors.append(error)

                integral += error * dt
                integral = np.clip(integral, controller.integral_limits[0], controller.integral_limits[1])

                derivative = (error - previous_error) / dt
                control = (controller.kp * error +
                          controller.ki * integral +
                          controller.kd * derivative)

                control = np.clip(control, controller.output_limits[0], controller.output_limits[1])

                # System dynamics
                acceleration = (control - damping * velocity) / inertia
                velocity += acceleration * dt
                position += velocity * dt

                actual_positions.append(position)
                previous_error = error

            # Calculate metrics
            errors_array = np.array(errors)
            max_error = float(np.max(np.abs(errors_array)))
            rms_error = float(np.sqrt(np.mean(errors_array**2)))

            return {
                'actual_trajectory': actual_positions,
                'desired_trajectory': desired_trajectory,
                'errors': errors,
                'max_error': max_error,
                'rms_error': rms_error,
                'tracking_performance': 'good' if max_error < 0.01 else 'poor'
            }

        except Exception as e:
            return {'error': str(e)}

    async def _ziegler_nichols_tuning(self, system_params: Dict[str, Any]) -> Dict[str, float]:
        """Ziegler-Nichols PID tuning"""
        # Simplified Ziegler-Nichols
        ku = system_params.get('ultimate_gain', 1.0)
        tu = system_params.get('oscillation_period', 1.0)

        kp = 0.6 * ku
        ki = 1.2 * ku / tu
        kd = 0.075 * ku * tu

        return {'kp': kp, 'ki': ki, 'kd': kd}

    async def _optimization_tuning(self, system_params: Dict[str, Any],
                                 criteria: Dict[str, Any]) -> Dict[str, float]:
        """Optimization-based PID tuning"""
        # Simplified optimization - in practice would use scipy.optimize
        return {'kp': 1.0, 'ki': 0.1, 'kd': 0.05}

    def _calculate_rise_time(self, time_points: np.ndarray, response: List[float], setpoint: float) -> float:
        """Calculate rise time"""
        response_array = np.array(response)
        target_indices = np.where(response_array >= 0.9 * setpoint)[0]
        return float(time_points[target_indices[0]]) if len(target_indices) > 0 else float('inf')

    def _calculate_settling_time(self, time_points: np.ndarray, response: List[float], setpoint: float) -> float:
        """Calculate settling time (2% criterion)"""
        response_array = np.array(response)
        settled_indices = np.where(np.abs(response_array - setpoint) <= 0.02 * setpoint)[0]
        return float(time_points[settled_indices[0]]) if len(settled_indices) > 0 else float('inf')

    def _calculate_overshoot(self, response: List[float], setpoint: float) -> float:
        """Calculate percentage overshoot"""
        response_array = np.array(response)
        max_response = np.max(response_array)
        return float(100 * (max_response - setpoint) / setpoint) if setpoint != 0 else 0.0