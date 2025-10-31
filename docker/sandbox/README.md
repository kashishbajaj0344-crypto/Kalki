# Kalki Simulation Sandbox

This directory contains the Docker-based sandbox environment for secure simulation execution in the Kalki framework.

## Overview

The simulation sandbox provides isolated, secure execution environments for running simulations with resource limits and security controls. It supports multiple isolation levels:

- **Container**: Full Docker container isolation (recommended)
- **Process**: Process-level isolation with resource limits
- **None**: Direct execution (not recommended for production)

## Files

- `Dockerfile`: Container definition for simulation sandbox
- `requirements.txt`: Python dependencies for simulations
- `docker-compose.yml`: Docker Compose configuration
- `build.sh`: Build script for the sandbox image

## Building the Sandbox

```bash
./build.sh
```

This will build the `kalki/simulation-sandbox:latest` Docker image.

## Security Features

- **Non-root user**: Simulations run as unprivileged user
- **Resource limits**: CPU, memory, and disk restrictions
- **Network isolation**: No network access by default
- **Filesystem restrictions**: Read-only root filesystem
- **Capability dropping**: No elevated privileges
- **Temporary filesystem**: Isolated /tmp with size limits

## Usage

The sandbox is automatically managed by the `SimulationAgent` when `enable_sandbox=True` in the configuration. Simulations are executed in isolated containers with automatic cleanup.

## Configuration

Sandbox behavior can be configured through the `SandboxConfig` class:

```python
config = SandboxConfig(
    isolation_level="container",
    resource_limits=ResourceLimits(cpu_percent=80.0, memory_mb=1024),
    network_access=False,
    security_profile="standard"
)
```

## Monitoring

Sandbox execution is monitored through the metrics system, tracking:

- Resource utilization
- Security events
- Execution time
- Success/failure rates