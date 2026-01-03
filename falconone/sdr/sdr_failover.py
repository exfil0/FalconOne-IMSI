"""
FalconOne SDR Enhanced Failover Manager
v1.9.4: Robust device failover with probing, health metrics, and automatic recovery

Features:
- uhd_usrp_probe integration for hardware validation
- Multi-device health scoring with weighted metrics
- Automatic failover with <10s switchover target
- Prometheus metrics for monitoring
- Device pool management for parallel operations
"""

import subprocess
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import statistics

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..utils.logger import ModuleLogger
from ..core.circuit_breaker import CircuitBreaker, RetryConfig


class SDRDeviceStatus(Enum):
    """SDR device health states"""
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    PROBING = "probing"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class SDRHealthMetrics:
    """Health metrics for an SDR device"""
    device_id: str
    device_type: str
    status: SDRDeviceStatus = SDRDeviceStatus.OFFLINE
    
    # Health scores (0-100)
    overall_health_score: float = 100.0
    signal_quality_score: float = 100.0
    timing_accuracy_score: float = 100.0
    throughput_score: float = 100.0
    
    # Operational metrics
    samples_received: int = 0
    samples_dropped: int = 0
    drop_rate: float = 0.0
    
    # Temperature and power (if available)
    temperature_c: Optional[float] = None
    power_consumption_w: Optional[float] = None
    
    # Timing
    last_probe_time: float = 0.0
    probe_latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    
    # Failover history
    failover_count: int = 0
    last_failover_time: float = 0.0
    consecutive_failures: int = 0
    
    # Additional sensor data
    frequency_offset_hz: float = 0.0
    sample_rate_actual: float = 0.0
    gain_db: float = 0.0


@dataclass
class SDRFailoverEvent:
    """Record of a failover event"""
    timestamp: float
    from_device: str
    to_device: str
    reason: str
    duration_ms: float
    success: bool
    error_message: Optional[str] = None


class SDRDeviceProbe:
    """Hardware-level device probing using vendor tools"""
    
    PROBE_COMMANDS = {
        'USRP': ['uhd_usrp_probe', '--args', 'addr={ip}'],
        'USRP-N310': ['uhd_usrp_probe', '--args', 'type=n3xx'],
        'USRP-B210': ['uhd_usrp_probe', '--args', 'type=b200'],
        'USRP-X410': ['uhd_usrp_probe', '--args', 'type=x4xx'],
        'HackRF': ['hackrf_info'],
        'BladeRF': ['bladeRF-cli', '-e', 'info'],
        'RTL-SDR': ['rtl_test', '-t'],
        'LimeSDR': ['LimeUtil', '--find'],
        'ADRV9009': ['iio_info', '-u', 'ip:{ip}'],
    }
    
    PROBE_TIMEOUT_S = 10
    
    def __init__(self, logger: logging.Logger):
        self.logger = ModuleLogger('SDR-Probe', logger)
        self._probe_cache: Dict[str, Tuple[float, bool, Dict]] = {}
        self._cache_ttl = 30.0  # Cache probe results for 30 seconds
    
    def probe_device(self, device_type: str, device_args: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Probe a specific SDR device for health and capabilities.
        
        Args:
            device_type: Type of SDR device
            device_args: Device-specific arguments (IP address, serial, etc.)
            
        Returns:
            Tuple of (is_healthy, probe_data)
        """
        cache_key = f"{device_type}:{str(device_args)}"
        
        # Check cache
        if cache_key in self._probe_cache:
            cached_time, cached_healthy, cached_data = self._probe_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_healthy, cached_data
        
        probe_start = time.time()
        
        try:
            cmd_template = self.PROBE_COMMANDS.get(device_type)
            if not cmd_template:
                self.logger.warning(f"No probe command for device type: {device_type}")
                return True, {'status': 'no_probe_available'}
            
            # Build command with arguments
            cmd = []
            for part in cmd_template:
                if '{ip}' in part and device_args:
                    part = part.replace('{ip}', device_args.get('addr', '192.168.10.2'))
                cmd.append(part)
            
            self.logger.debug(f"Probing {device_type}", command=' '.join(cmd))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.PROBE_TIMEOUT_S
            )
            
            probe_duration = (time.time() - probe_start) * 1000
            
            probe_data = {
                'probe_time_ms': probe_duration,
                'return_code': result.returncode,
                'stdout': result.stdout[:2000] if result.stdout else '',
                'stderr': result.stderr[:500] if result.stderr else '',
            }
            
            # Parse device-specific data
            is_healthy = result.returncode == 0
            
            if device_type.startswith('USRP'):
                probe_data.update(self._parse_usrp_probe(result.stdout))
            elif device_type == 'HackRF':
                probe_data.update(self._parse_hackrf_info(result.stdout))
            elif device_type == 'BladeRF':
                probe_data.update(self._parse_bladerf_info(result.stdout))
            elif device_type == 'LimeSDR':
                probe_data.update(self._parse_lime_info(result.stdout))
            
            # Cache result
            self._probe_cache[cache_key] = (time.time(), is_healthy, probe_data)
            
            self.logger.info(f"Probe complete: {device_type}", 
                           healthy=is_healthy, 
                           duration_ms=probe_duration)
            
            return is_healthy, probe_data
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Probe timeout for {device_type}")
            return False, {'error': 'probe_timeout'}
        except FileNotFoundError as e:
            self.logger.warning(f"Probe tool not found for {device_type}: {e}")
            return True, {'status': 'probe_tool_missing'}
        except Exception as e:
            self.logger.error(f"Probe error for {device_type}: {e}")
            return False, {'error': str(e)}
    
    def _parse_usrp_probe(self, output: str) -> Dict[str, Any]:
        """Parse uhd_usrp_probe output"""
        data = {}
        
        if not output:
            return data
        
        # Extract device type
        if 'Device:' in output:
            for line in output.split('\n'):
                if 'mboard' in line.lower():
                    data['mboard'] = line.strip()
                if 'Serial' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        data['serial'] = parts[1].strip()
                if 'subdev' in line.lower():
                    data['subdevice'] = line.strip()
        
        return data
    
    def _parse_hackrf_info(self, output: str) -> Dict[str, Any]:
        """Parse hackrf_info output"""
        data = {}
        
        if not output:
            return data
        
        for line in output.split('\n'):
            if 'Serial number' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    data['serial'] = parts[1].strip()
            if 'Firmware Version' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    data['firmware'] = parts[1].strip()
        
        return data
    
    def _parse_bladerf_info(self, output: str) -> Dict[str, Any]:
        """Parse bladeRF-cli info output"""
        data = {}
        
        if not output:
            return data
        
        for line in output.split('\n'):
            if 'Serial #' in line or 'Serial:' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    data['serial'] = parts[1].strip()
            if 'FPGA' in line:
                data['fpga_info'] = line.strip()
        
        return data
    
    def _parse_lime_info(self, output: str) -> Dict[str, Any]:
        """Parse LimeUtil output"""
        data = {}
        
        if not output:
            return data
        
        for line in output.split('\n'):
            if 'LimeSDR' in line:
                data['device_info'] = line.strip()
            if 'Serial' in line:
                parts = line.split(':')
                if len(parts) >= 2:
                    data['serial'] = parts[1].strip()
        
        return data


class SDRFailoverManager:
    """
    Advanced SDR failover management with health monitoring and automatic recovery.
    
    Features:
    - Multi-device pool management
    - Health scoring with weighted metrics
    - Automatic failover on device failure
    - Device probing for hardware validation
    - Prometheus metrics integration
    """
    
    # Device priority scores (higher = preferred)
    DEVICE_PRIORITY = {
        'USRP-X410': 100,
        'USRP-N310': 95,
        'USRP-B210': 85,
        'ADRV9009': 90,
        'BladeRF-x40': 75,
        'LimeSDR': 70,
        'HackRF': 60,
        'RTL-SDR': 50,
    }
    
    # Health check weights
    HEALTH_WEIGHTS = {
        'signal_quality': 0.30,
        'timing_accuracy': 0.25,
        'throughput': 0.25,
        'temperature': 0.10,
        'drop_rate': 0.10,
    }
    
    def __init__(self, config, logger: logging.Logger, sdr_manager=None):
        """
        Initialize failover manager.
        
        Args:
            config: Configuration object
            logger: Logger instance
            sdr_manager: Reference to SDRManager for device control
        """
        self.config = config
        self.logger = ModuleLogger('SDR-Failover', logger)
        self.sdr_manager = sdr_manager
        
        # Device tracking
        self._devices: Dict[str, SDRHealthMetrics] = {}
        self._active_device: Optional[str] = None
        self._device_lock = threading.RLock()
        
        # Failover configuration
        self.failover_threshold_ms = config.get('sdr.failover_threshold_ms', 10000)
        self.health_check_interval_s = config.get('sdr.health_check_interval_s', 5)
        self.min_health_score = config.get('sdr.min_health_score', 30.0)
        self.auto_failover_enabled = config.get('sdr.auto_failover', True)
        self.max_consecutive_failures = config.get('sdr.max_consecutive_failures', 3)
        
        # Device probing
        self.probe = SDRDeviceProbe(logger)
        
        # Failover history
        self._failover_history: deque = deque(maxlen=100)
        
        # Health monitoring thread
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Circuit breaker for failover operations
        self._failover_circuit = CircuitBreaker(
            name="sdr_failover",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=60.0,
            retry_config=RetryConfig(max_retries=2, base_delay_ms=500),
            logger=logger
        )
        
        # Callbacks
        self._on_failover: Optional[Callable[[SDRFailoverEvent], None]] = None
        self._on_device_status_change: Optional[Callable[[str, SDRDeviceStatus], None]] = None
        
        # Initialize Prometheus metrics if available
        self._init_prometheus_metrics()
        
        self.logger.info("SDR Failover Manager initialized",
                        threshold_ms=self.failover_threshold_ms,
                        auto_failover=self.auto_failover_enabled)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for monitoring"""
        if not PROMETHEUS_AVAILABLE:
            self.logger.debug("Prometheus client not available, metrics disabled")
            return
        
        try:
            self.metrics = {
                'failover_total': Counter(
                    'sdr_failover_total',
                    'Total number of SDR failovers',
                    ['from_device', 'to_device', 'reason']
                ),
                'failover_duration': Histogram(
                    'sdr_failover_duration_ms',
                    'SDR failover duration in milliseconds',
                    buckets=[100, 500, 1000, 2000, 5000, 10000, 30000]
                ),
                'device_health_score': Gauge(
                    'sdr_device_health_score',
                    'SDR device health score (0-100)',
                    ['device_id', 'device_type']
                ),
                'device_status': Gauge(
                    'sdr_device_status',
                    'SDR device status (1=online, 0.5=degraded, 0=offline)',
                    ['device_id', 'device_type']
                ),
                'samples_dropped_total': Counter(
                    'sdr_samples_dropped_total',
                    'Total dropped samples per device',
                    ['device_id']
                ),
            }
            self.logger.debug("Prometheus metrics initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Prometheus metrics: {e}")
            self.metrics = {}
    
    def register_device(self, device_id: str, device_type: str, device_args: Dict = None):
        """
        Register an SDR device for failover management.
        
        Args:
            device_id: Unique identifier for the device
            device_type: Type of device (USRP, HackRF, etc.)
            device_args: Device connection arguments
        """
        with self._device_lock:
            metrics = SDRHealthMetrics(
                device_id=device_id,
                device_type=device_type,
                status=SDRDeviceStatus.PROBING
            )
            self._devices[device_id] = metrics
            
            self.logger.info(f"Registered device: {device_id} ({device_type})")
            
            # Initial probe
            threading.Thread(
                target=self._probe_device_async,
                args=(device_id, device_type, device_args),
                daemon=True
            ).start()
    
    def _probe_device_async(self, device_id: str, device_type: str, device_args: Dict = None):
        """Probe device in background thread"""
        try:
            is_healthy, probe_data = self.probe.probe_device(device_type, device_args)
            
            with self._device_lock:
                if device_id in self._devices:
                    metrics = self._devices[device_id]
                    metrics.status = SDRDeviceStatus.ONLINE if is_healthy else SDRDeviceStatus.OFFLINE
                    metrics.last_probe_time = time.time()
                    metrics.probe_latency_ms = probe_data.get('probe_time_ms', 0)
                    
                    if is_healthy:
                        metrics.consecutive_failures = 0
                    else:
                        metrics.consecutive_failures += 1
                    
                    self._update_device_metrics(device_id, metrics)
                    
        except Exception as e:
            self.logger.error(f"Async probe failed for {device_id}: {e}")
    
    def _update_device_metrics(self, device_id: str, metrics: SDRHealthMetrics):
        """Update Prometheus metrics for a device"""
        if not self.metrics:
            return
        
        try:
            self.metrics['device_health_score'].labels(
                device_id=device_id,
                device_type=metrics.device_type
            ).set(metrics.overall_health_score)
            
            status_value = {
                SDRDeviceStatus.ONLINE: 1.0,
                SDRDeviceStatus.DEGRADED: 0.5,
                SDRDeviceStatus.RECOVERING: 0.5,
                SDRDeviceStatus.PROBING: 0.5,
                SDRDeviceStatus.OFFLINE: 0.0,
                SDRDeviceStatus.FAILED: 0.0,
            }.get(metrics.status, 0.0)
            
            self.metrics['device_status'].labels(
                device_id=device_id,
                device_type=metrics.device_type
            ).set(status_value)
        except Exception as e:
            self.logger.debug(f"Metrics update error: {e}")
    
    def set_active_device(self, device_id: str) -> bool:
        """
        Set the active device for operations.
        
        Args:
            device_id: Device to make active
            
        Returns:
            True if successful
        """
        with self._device_lock:
            if device_id not in self._devices:
                self.logger.error(f"Device not registered: {device_id}")
                return False
            
            metrics = self._devices[device_id]
            if metrics.status == SDRDeviceStatus.OFFLINE:
                self.logger.warning(f"Attempting to activate offline device: {device_id}")
            
            old_active = self._active_device
            self._active_device = device_id
            
            self.logger.info(f"Active device changed: {old_active} → {device_id}")
            return True
    
    def get_active_device(self) -> Optional[str]:
        """Get the currently active device ID"""
        return self._active_device
    
    def update_device_health(
        self,
        device_id: str,
        samples_received: int = 0,
        samples_dropped: int = 0,
        signal_quality: float = None,
        timing_accuracy: float = None,
        temperature_c: float = None
    ):
        """
        Update health metrics for a device.
        
        Args:
            device_id: Device to update
            samples_received: Number of samples received since last update
            samples_dropped: Number of samples dropped since last update
            signal_quality: Signal quality score (0-100)
            timing_accuracy: Timing accuracy score (0-100)
            temperature_c: Device temperature in Celsius
        """
        with self._device_lock:
            if device_id not in self._devices:
                return
            
            metrics = self._devices[device_id]
            metrics.samples_received += samples_received
            metrics.samples_dropped += samples_dropped
            
            # Calculate drop rate
            if metrics.samples_received > 0:
                metrics.drop_rate = metrics.samples_dropped / (metrics.samples_received + metrics.samples_dropped)
            
            # Update individual scores
            if signal_quality is not None:
                metrics.signal_quality_score = signal_quality
            if timing_accuracy is not None:
                metrics.timing_accuracy_score = timing_accuracy
            if temperature_c is not None:
                metrics.temperature_c = temperature_c
            
            # Calculate throughput score (based on drop rate)
            metrics.throughput_score = max(0, 100 - (metrics.drop_rate * 100))
            
            # Calculate overall health score
            metrics.overall_health_score = self._calculate_health_score(metrics)
            
            # Update device status based on health
            if metrics.overall_health_score >= 80:
                metrics.status = SDRDeviceStatus.ONLINE
            elif metrics.overall_health_score >= 50:
                metrics.status = SDRDeviceStatus.DEGRADED
            else:
                metrics.status = SDRDeviceStatus.OFFLINE
                metrics.consecutive_failures += 1
            
            # Update Prometheus metrics
            self._update_device_metrics(device_id, metrics)
            
            # Check for failover conditions
            if self.auto_failover_enabled and device_id == self._active_device:
                if metrics.overall_health_score < self.min_health_score:
                    self.logger.warning(f"Active device health below threshold: {metrics.overall_health_score:.1f}")
                    self._trigger_failover(device_id, "health_score_low")
    
    def _calculate_health_score(self, metrics: SDRHealthMetrics) -> float:
        """Calculate weighted health score"""
        score = 0.0
        
        score += self.HEALTH_WEIGHTS['signal_quality'] * metrics.signal_quality_score
        score += self.HEALTH_WEIGHTS['timing_accuracy'] * metrics.timing_accuracy_score
        score += self.HEALTH_WEIGHTS['throughput'] * metrics.throughput_score
        
        # Temperature penalty (if available and too high)
        if metrics.temperature_c is not None:
            temp_score = 100.0
            if metrics.temperature_c > 80:
                temp_score = max(0, 100 - ((metrics.temperature_c - 80) * 5))
            score += self.HEALTH_WEIGHTS['temperature'] * temp_score
        else:
            score += self.HEALTH_WEIGHTS['temperature'] * 100
        
        # Drop rate penalty
        drop_score = max(0, 100 - (metrics.drop_rate * 200))
        score += self.HEALTH_WEIGHTS['drop_rate'] * drop_score
        
        return min(100, max(0, score))
    
    def _trigger_failover(self, failed_device_id: str, reason: str):
        """Trigger failover from failed device to best available"""
        failover_start = time.time()
        
        self.logger.warning(f"🔄 INITIATING FAILOVER from {failed_device_id}: {reason}")
        
        try:
            # Find best available device
            target_device = self._select_best_device(exclude=[failed_device_id])
            
            if not target_device:
                self.logger.error("❌ FAILOVER FAILED: No healthy devices available")
                event = SDRFailoverEvent(
                    timestamp=failover_start,
                    from_device=failed_device_id,
                    to_device="NONE",
                    reason=reason,
                    duration_ms=0,
                    success=False,
                    error_message="No healthy devices available"
                )
                self._record_failover(event)
                return
            
            # Execute failover
            success = self._execute_failover(failed_device_id, target_device)
            
            duration_ms = (time.time() - failover_start) * 1000
            
            event = SDRFailoverEvent(
                timestamp=failover_start,
                from_device=failed_device_id,
                to_device=target_device,
                reason=reason,
                duration_ms=duration_ms,
                success=success
            )
            
            self._record_failover(event)
            
            if success:
                target_met = duration_ms < self.failover_threshold_ms
                self.logger.info(
                    f"✅ FAILOVER COMPLETE: {failed_device_id} → {target_device} "
                    f"in {duration_ms:.0f}ms "
                    f"({'✓ PASS' if target_met else '✗ FAIL'} <{self.failover_threshold_ms}ms target)"
                )
            else:
                self.logger.error(f"❌ FAILOVER FAILED after {duration_ms:.0f}ms")
                
        except Exception as e:
            self.logger.error(f"Failover error: {e}")
            event = SDRFailoverEvent(
                timestamp=failover_start,
                from_device=failed_device_id,
                to_device="UNKNOWN",
                reason=reason,
                duration_ms=(time.time() - failover_start) * 1000,
                success=False,
                error_message=str(e)
            )
            self._record_failover(event)
    
    def _select_best_device(self, exclude: List[str] = None) -> Optional[str]:
        """Select best available device based on health and priority"""
        exclude = exclude or []
        
        candidates = []
        
        with self._device_lock:
            for device_id, metrics in self._devices.items():
                if device_id in exclude:
                    continue
                
                if metrics.status in (SDRDeviceStatus.ONLINE, SDRDeviceStatus.DEGRADED):
                    # Calculate combined score (health + priority)
                    priority = self.DEVICE_PRIORITY.get(metrics.device_type, 50)
                    combined_score = (metrics.overall_health_score * 0.6) + (priority * 0.4)
                    candidates.append((device_id, combined_score))
        
        if not candidates:
            return None
        
        # Sort by combined score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _execute_failover(self, from_device: str, to_device: str) -> bool:
        """Execute the actual failover operation"""
        try:
            with self._device_lock:
                # Mark old device as offline
                if from_device in self._devices:
                    old_metrics = self._devices[from_device]
                    old_metrics.status = SDRDeviceStatus.OFFLINE
                    old_metrics.failover_count += 1
                    old_metrics.last_failover_time = time.time()
                
                # Activate new device
                if to_device in self._devices:
                    new_metrics = self._devices[to_device]
                    new_metrics.status = SDRDeviceStatus.ONLINE
                
                self._active_device = to_device
            
            # If we have an SDR manager reference, perform actual device switch
            if self.sdr_manager and hasattr(self.sdr_manager, 'activate_device'):
                try:
                    self.sdr_manager.activate_device(to_device)
                except Exception as e:
                    self.logger.error(f"SDR manager activation failed: {e}")
            
            # Notify callback
            if self._on_failover:
                try:
                    event = SDRFailoverEvent(
                        timestamp=time.time(),
                        from_device=from_device,
                        to_device=to_device,
                        reason="health_check",
                        duration_ms=0,
                        success=True
                    )
                    self._on_failover(event)
                except Exception as e:
                    self.logger.debug(f"Failover callback error: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failover execution error: {e}")
            return False
    
    def _record_failover(self, event: SDRFailoverEvent):
        """Record failover event for history and metrics"""
        self._failover_history.append(event)
        
        # Update Prometheus metrics
        if self.metrics and event.success:
            try:
                self.metrics['failover_total'].labels(
                    from_device=event.from_device,
                    to_device=event.to_device,
                    reason=event.reason
                ).inc()
                
                self.metrics['failover_duration'].observe(event.duration_ms)
            except Exception as e:
                self.logger.debug(f"Metrics recording error: {e}")
    
    def start_monitoring(self):
        """Start health monitoring thread"""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="SDR-FailoverMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("Failover monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring thread"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Failover monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                # Check all devices
                with self._device_lock:
                    for device_id, metrics in list(self._devices.items()):
                        # Re-probe devices that haven't been checked recently
                        if time.time() - metrics.last_probe_time > 60:
                            threading.Thread(
                                target=self._probe_device_async,
                                args=(device_id, metrics.device_type, None),
                                daemon=True
                            ).start()
                
                time.sleep(self.health_check_interval_s)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.health_check_interval_s)
    
    def get_status(self) -> Dict[str, Any]:
        """Get failover manager status"""
        with self._device_lock:
            devices_status = {
                device_id: {
                    'type': metrics.device_type,
                    'status': metrics.status.value,
                    'health_score': metrics.overall_health_score,
                    'failover_count': metrics.failover_count,
                    'consecutive_failures': metrics.consecutive_failures,
                }
                for device_id, metrics in self._devices.items()
            }
        
        return {
            'active_device': self._active_device,
            'devices': devices_status,
            'monitoring_active': self._monitoring_active,
            'auto_failover_enabled': self.auto_failover_enabled,
            'failover_threshold_ms': self.failover_threshold_ms,
            'total_failovers': len(self._failover_history),
            'recent_failovers': [
                {
                    'timestamp': e.timestamp,
                    'from': e.from_device,
                    'to': e.to_device,
                    'duration_ms': e.duration_ms,
                    'success': e.success
                }
                for e in list(self._failover_history)[-5:]
            ]
        }
    
    def on_failover(self, callback: Callable[[SDRFailoverEvent], None]):
        """Register failover event callback"""
        self._on_failover = callback
    
    def on_device_status_change(self, callback: Callable[[str, SDRDeviceStatus], None]):
        """Register device status change callback"""
        self._on_device_status_change = callback


class MultiSDRPool:
    """
    Pool manager for parallel SDR operations across multiple devices.
    
    Enables concurrent capture from multiple SDR devices for increased throughput.
    """
    
    def __init__(self, config, logger: logging.Logger, failover_manager: SDRFailoverManager):
        self.config = config
        self.logger = ModuleLogger('SDR-Pool', logger)
        self.failover_manager = failover_manager
        
        # Pool configuration
        self.max_concurrent = config.get('sdr.max_concurrent_devices', 4)
        self._active_devices: List[str] = []
        self._device_assignments: Dict[str, List[int]] = {}  # device_id -> [ARFCNs]
        self._lock = threading.Lock()
        
        self.logger.info(f"SDR Pool initialized (max concurrent: {self.max_concurrent})")
    
    def allocate_arfcns(self, arfcns: List[int]) -> Dict[str, List[int]]:
        """
        Allocate ARFCNs across available SDR devices.
        
        Args:
            arfcns: List of ARFCNs to allocate
            
        Returns:
            Dict mapping device_id to assigned ARFCNs
        """
        with self._lock:
            # Get available devices
            available = self._get_available_devices()
            
            if not available:
                self.logger.warning("No devices available for ARFCN allocation")
                return {}
            
            # Distribute ARFCNs evenly
            num_devices = min(len(available), self.max_concurrent, len(arfcns))
            assignments = {dev: [] for dev in available[:num_devices]}
            
            for i, arfcn in enumerate(arfcns):
                device = available[i % num_devices]
                assignments[device].append(arfcn)
            
            self._device_assignments = assignments
            self._active_devices = list(assignments.keys())
            
            self.logger.info(f"Allocated {len(arfcns)} ARFCNs across {num_devices} devices")
            return assignments
    
    def _get_available_devices(self) -> List[str]:
        """Get list of available devices sorted by priority"""
        status = self.failover_manager.get_status()
        
        available = []
        for device_id, info in status.get('devices', {}).items():
            if info['status'] in ('online', 'degraded'):
                available.append((device_id, info.get('health_score', 0)))
        
        # Sort by health score
        available.sort(key=lambda x: x[1], reverse=True)
        return [dev[0] for dev in available]
    
    def get_device_for_arfcn(self, arfcn: int) -> Optional[str]:
        """Get the device assigned to a specific ARFCN"""
        with self._lock:
            for device_id, arfcns in self._device_assignments.items():
                if arfcn in arfcns:
                    return device_id
        return None
    
    def release_device(self, device_id: str):
        """Release a device back to the pool"""
        with self._lock:
            if device_id in self._active_devices:
                self._active_devices.remove(device_id)
            if device_id in self._device_assignments:
                del self._device_assignments[device_id]
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get pool status"""
        with self._lock:
            return {
                'max_concurrent': self.max_concurrent,
                'active_devices': self._active_devices.copy(),
                'assignments': {k: v.copy() for k, v in self._device_assignments.items()},
                'total_arfcns_assigned': sum(len(v) for v in self._device_assignments.values())
            }
