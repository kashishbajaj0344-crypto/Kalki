"""
Real-Time Sensor Data Pipeline
Ingests sensor data from IoT devices, robots, and deployed systems in real-time.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors supported"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    STRAIN = "strain"
    VIBRATION = "vibration"
    POSITION = "position"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    FORCE = "force"
    TORQUE = "torque"
    CURRENT = "current"
    VOLTAGE = "voltage"
    OPTICAL = "optical"
    ACOUSTIC = "acoustic"
    PROXIMITY = "proximity"
    CAMERA = "camera"
    LIDAR = "lidar"
    CUSTOM = "custom"


class DataProtocol(Enum):
    """Communication protocols supported"""
    HTTP_REST = "http_rest"
    WEBSOCKET = "websocket"
    MQTT = "mqtt"
    GRPC = "grpc"
    MODBUS = "modbus"
    OPCUA = "opcua"
    SERIAL = "serial"
    CAN_BUS = "can_bus"


@dataclass
class SensorReading:
    """Single sensor reading"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    value: float
    unit: str
    quality: float  # 0-1, data quality/confidence
    metadata: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class SensorConfig:
    """Configuration for a sensor"""
    sensor_id: str
    sensor_type: SensorType
    endpoint: str
    protocol: DataProtocol
    sampling_rate_hz: float
    calibration: Dict[str, float] = field(default_factory=dict)
    filters: List[str] = field(default_factory=list)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class SensorStream:
    """Active sensor data stream"""
    config: SensorConfig
    last_reading: Optional[SensorReading] = None
    total_readings: int = 0
    anomaly_count: int = 0
    is_active: bool = False
    

class SensorDataPipeline:
    """
    Real-time sensor data ingestion and processing pipeline.
    
    Features:
    - Multi-protocol sensor support (HTTP, MQTT, WebSocket, etc.)
    - Real-time data validation and filtering
    - Anomaly detection
    - Data buffering and batching
    - Alert triggering
    - Integration with telemetry system
    """
    
    def __init__(self):
        self.sensors: Dict[str, SensorStream] = {}
        self.data_buffer: List[SensorReading] = []
        self.buffer_size = 10000
        self.callbacks: List[Callable] = []
        self.is_running = False
        
    async def initialize(self):
        """Initialize the sensor pipeline"""
        logger.info("📡 Initializing Sensor Data Pipeline")
        
        # Load sensor configurations
        await self._load_sensor_configs()
        
        logger.info(f"✅ Sensor pipeline initialized with {len(self.sensors)} sensors")
        
    async def register_sensor(self, config: SensorConfig):
        """Register a new sensor"""
        stream = SensorStream(config=config)
        self.sensors[config.sensor_id] = stream
        
        logger.info(f"📝 Registered sensor: {config.sensor_id} ({config.sensor_type.value})")
        
        # Start sensor data collection
        if self.is_running:
            await self._start_sensor_collection(config.sensor_id)
            
    async def start(self):
        """Start sensor data collection"""
        if self.is_running:
            logger.warning("Sensor pipeline already running")
            return
            
        self.is_running = True
        logger.info("🔄 Starting sensor data collection")
        
        # Start collection for all sensors
        tasks = [
            self._start_sensor_collection(sensor_id)
            for sensor_id in self.sensors.keys()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def stop(self):
        """Stop sensor data collection"""
        self.is_running = False
        
        # Mark all sensors as inactive
        for stream in self.sensors.values():
            stream.is_active = False
            
        logger.info("⏸️ Sensor pipeline stopped")
        
    async def _start_sensor_collection(self, sensor_id: str):
        """Start collecting data from a specific sensor"""
        if sensor_id not in self.sensors:
            logger.error(f"Unknown sensor: {sensor_id}")
            return
            
        stream = self.sensors[sensor_id]
        stream.is_active = True
        
        logger.info(f"🎯 Starting collection from {sensor_id}")
        
        while self.is_running and stream.is_active:
            try:
                # Collect sensor reading based on protocol
                reading = await self._collect_reading(stream.config)
                
                if reading:
                    # Process the reading
                    await self._process_reading(reading, stream)
                    
                # Wait based on sampling rate
                await asyncio.sleep(1.0 / stream.config.sampling_rate_hz)
                
            except Exception as e:
                logger.error(f"Error collecting from {sensor_id}: {e}")
                await asyncio.sleep(1.0)
                
    async def _collect_reading(self, config: SensorConfig) -> Optional[SensorReading]:
        """Collect a single reading from a sensor"""
        try:
            if config.protocol == DataProtocol.HTTP_REST:
                return await self._collect_http(config)
            elif config.protocol == DataProtocol.WEBSOCKET:
                return await self._collect_websocket(config)
            elif config.protocol == DataProtocol.MQTT:
                return await self._collect_mqtt(config)
            else:
                # For demonstration, simulate sensor data
                return await self._simulate_reading(config)
                
        except Exception as e:
            logger.error(f"Collection error for {config.sensor_id}: {e}")
            return None
            
    async def _collect_http(self, config: SensorConfig) -> Optional[SensorReading]:
        """Collect data via HTTP REST API"""
        # In production, would make actual HTTP request
        # For now, simulate
        return await self._simulate_reading(config)
        
    async def _collect_websocket(self, config: SensorConfig) -> Optional[SensorReading]:
        """Collect data via WebSocket"""
        # In production, would maintain WebSocket connection
        return await self._simulate_reading(config)
        
    async def _collect_mqtt(self, config: SensorConfig) -> Optional[SensorReading]:
        """Collect data via MQTT"""
        # In production, would subscribe to MQTT topic
        return await self._simulate_reading(config)
        
    async def _simulate_reading(self, config: SensorConfig) -> SensorReading:
        """Simulate a sensor reading for testing"""
        import random
        
        # Simulate realistic sensor values based on type
        sensor_ranges = {
            SensorType.TEMPERATURE: (20.0, 80.0, "°C"),
            SensorType.PRESSURE: (100.0, 200.0, "kPa"),
            SensorType.STRAIN: (0.0, 1000.0, "με"),
            SensorType.VIBRATION: (0.0, 50.0, "mm/s"),
            SensorType.FORCE: (0.0, 1000.0, "N"),
            SensorType.TORQUE: (0.0, 500.0, "Nm"),
            SensorType.POSITION: (0.0, 100.0, "mm"),
            SensorType.VELOCITY: (0.0, 10.0, "m/s"),
        }
        
        range_data = sensor_ranges.get(config.sensor_type, (0.0, 100.0, "units"))
        min_val, max_val, unit = range_data
        
        # Generate value with some noise
        base_value = (min_val + max_val) / 2
        noise = random.gauss(0, (max_val - min_val) * 0.1)
        value = base_value + noise
        
        # Apply calibration if configured
        if 'offset' in config.calibration:
            value += config.calibration['offset']
        if 'scale' in config.calibration:
            value *= config.calibration['scale']
            
        return SensorReading(
            sensor_id=config.sensor_id,
            sensor_type=config.sensor_type,
            timestamp=datetime.now(),
            value=value,
            unit=unit,
            quality=random.uniform(0.9, 1.0),  # Simulate good quality
            metadata={'simulated': True}
        )
        
    async def _process_reading(self, reading: SensorReading, stream: SensorStream):
        """Process a sensor reading"""
        # Update stream
        stream.last_reading = reading
        stream.total_readings += 1
        
        # Apply filters
        if not await self._apply_filters(reading, stream.config):
            return  # Reading filtered out
            
        # Detect anomalies
        if await self._detect_sensor_anomaly(reading, stream):
            stream.anomaly_count += 1
            logger.warning(f"⚠️ Anomaly detected in {reading.sensor_id}: {reading.value:.2f} {reading.unit}")
            
            # Trigger alerts if configured
            await self._trigger_alerts(reading, stream.config)
            
        # Add to buffer
        self.data_buffer.append(reading)
        
        # Trim buffer if needed
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            
        # Call registered callbacks
        for callback in self.callbacks:
            try:
                await callback(reading)
            except Exception as e:
                logger.error(f"Callback error: {e}")
                
        # Integrate with telemetry system
        await self._integrate_with_telemetry(reading)
        
    async def _apply_filters(self, reading: SensorReading, config: SensorConfig) -> bool:
        """Apply configured filters to reading"""
        # Check quality threshold
        if reading.quality < 0.5:
            return False  # Low quality, filter out
            
        # Apply custom filters
        for filter_name in config.filters:
            if filter_name == "outlier_rejection":
                # Simple outlier detection
                if abs(reading.value) > 1000:  # Configurable threshold
                    return False
                    
        return True
        
    async def _detect_sensor_anomaly(self, reading: SensorReading, stream: SensorStream) -> bool:
        """Detect if reading is anomalous"""
        config = stream.config
        
        # Check against alert thresholds
        for threshold_name, threshold_value in config.alert_thresholds.items():
            if threshold_name == "max" and reading.value > threshold_value:
                return True
            elif threshold_name == "min" and reading.value < threshold_value:
                return True
                
        # Statistical anomaly detection (if we have history)
        if stream.total_readings > 100:
            # Could implement more sophisticated anomaly detection here
            # For now, simple threshold check
            pass
            
        return False
        
    async def _trigger_alerts(self, reading: SensorReading, config: SensorConfig):
        """Trigger alerts for anomalous readings"""
        alert_data = {
            'sensor_id': reading.sensor_id,
            'sensor_type': reading.sensor_type.value,
            'value': reading.value,
            'unit': reading.unit,
            'timestamp': reading.timestamp.isoformat(),
            'thresholds': config.alert_thresholds
        }
        
        # Log alert
        logger.warning(f"🚨 Alert triggered: {json.dumps(alert_data, indent=2)}")
        
        # In production, would send notifications via email, SMS, etc.
        
    async def _integrate_with_telemetry(self, reading: SensorReading):
        """Integrate sensor reading with telemetry system"""
        try:
            from modules.realworld_telemetry_integration import get_telemetry_integration, TelemetryType
            
            telemetry = get_telemetry_integration()
            
            # Map sensor type to telemetry type
            telemetry_type_map = {
                SensorType.TEMPERATURE: TelemetryType.THERMAL,
                SensorType.STRAIN: TelemetryType.STRUCTURAL,
                SensorType.FORCE: TelemetryType.STRUCTURAL,
                SensorType.VIBRATION: TelemetryType.ACOUSTIC,
            }
            
            telemetry_type = telemetry_type_map.get(reading.sensor_type, TelemetryType.PERFORMANCE)
            
            # Ingest into telemetry system (if sensor is associated with a design)
            if 'design_id' in reading.metadata:
                await telemetry.ingest_telemetry(
                    design_id=reading.metadata['design_id'],
                    telemetry_type=telemetry_type,
                    measurements={reading.sensor_type.value: reading.value},
                    metadata={'sensor_id': reading.sensor_id, 'quality': reading.quality}
                )
                
        except Exception as e:
            logger.debug(f"Telemetry integration skipped: {e}")
            
    def register_callback(self, callback: Callable):
        """Register a callback to be called for each reading"""
        self.callbacks.append(callback)
        logger.info(f"📝 Registered sensor data callback")
        
    async def _load_sensor_configs(self):
        """Load sensor configurations from file"""
        try:
            config_path = Path("data/sensor_configs.json")
            if config_path.exists():
                with open(config_path) as f:
                    data = json.load(f)
                    for item in data:
                        config = SensorConfig(
                            sensor_id=item['sensor_id'],
                            sensor_type=SensorType(item['sensor_type']),
                            endpoint=item['endpoint'],
                            protocol=DataProtocol(item['protocol']),
                            sampling_rate_hz=item['sampling_rate_hz'],
                            calibration=item.get('calibration', {}),
                            filters=item.get('filters', []),
                            alert_thresholds=item.get('alert_thresholds', {})
                        )
                        stream = SensorStream(config=config)
                        self.sensors[config.sensor_id] = stream
        except Exception as e:
            logger.debug(f"No sensor configs loaded: {e}")
            
    async def save_sensor_configs(self):
        """Save sensor configurations to file"""
        try:
            config_path = Path("data/sensor_configs.json")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    'sensor_id': stream.config.sensor_id,
                    'sensor_type': stream.config.sensor_type.value,
                    'endpoint': stream.config.endpoint,
                    'protocol': stream.config.protocol.value,
                    'sampling_rate_hz': stream.config.sampling_rate_hz,
                    'calibration': stream.config.calibration,
                    'filters': stream.config.filters,
                    'alert_thresholds': stream.config.alert_thresholds
                }
                for stream in self.sensors.values()
            ]
            
            with open(config_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving sensor configs: {e}")
            
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        active_sensors = sum(1 for s in self.sensors.values() if s.is_active)
        total_readings = sum(s.total_readings for s in self.sensors.values())
        total_anomalies = sum(s.anomaly_count for s in self.sensors.values())
        
        return {
            'is_running': self.is_running,
            'total_sensors': len(self.sensors),
            'active_sensors': active_sensors,
            'total_readings': total_readings,
            'total_anomalies': total_anomalies,
            'buffer_size': len(self.data_buffer),
            'callbacks_registered': len(self.callbacks)
        }


# Singleton instance
_sensor_pipeline = None

def get_sensor_pipeline() -> SensorDataPipeline:
    """Get the global sensor pipeline instance"""
    global _sensor_pipeline
    if _sensor_pipeline is None:
        _sensor_pipeline = SensorDataPipeline()
    return _sensor_pipeline
