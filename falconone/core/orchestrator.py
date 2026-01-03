"""
FalconOne Core Orchestrator
Main coordination layer for all operations with safety interlocks
Version 1.4.1: Complete integration with all modules + Signal Bus
Version 1.9.2: Added HealthMonitor for periodic component health checks and automatic restarts
"""

import logging
import signal
import sys
import os
import time
import threading
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from ..utils.config import Config
from ..utils.logger import setup_logger, ModuleLogger, AuditLogger
from ..utils.exceptions import SafetyViolation, ConfigurationError, IntegrationError
from .signal_bus import SignalBus


class ComponentStatus(Enum):
    """Health status of a component"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    STOPPED = "stopped"


@dataclass
class ComponentHealth:
    """Health information for a single component"""
    name: str
    status: ComponentStatus = ComponentStatus.STOPPED
    last_check: Optional[datetime] = None
    last_healthy: Optional[datetime] = None
    consecutive_failures: int = 0
    restart_count: int = 0
    error_message: Optional[str] = None
    response_time_ms: float = 0.0


class HealthMonitor:
    """
    Monitors component health and performs automatic restarts on failure.
    
    Features:
    - Periodic health checks for all registered components
    - Automatic restart with exponential backoff
    - Circuit breaker pattern to prevent restart storms
    - Health metrics for monitoring/alerting integration
    
    Version: 1.9.2
    """
    
    def __init__(
        self,
        check_interval: float = 30.0,
        max_restart_attempts: int = 3,
        restart_backoff_base: float = 5.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HealthMonitor.
        
        Args:
            check_interval: Seconds between health checks (default: 30)
            max_restart_attempts: Maximum restart attempts before giving up (default: 3)
            restart_backoff_base: Base seconds for exponential backoff (default: 5)
            logger: Logger instance for health monitor events
        """
        self.check_interval = check_interval
        self.max_restart_attempts = max_restart_attempts
        self.restart_backoff_base = restart_backoff_base
        self.logger = logger or logging.getLogger(__name__)
        
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Component registry: name -> (component, health_check_fn, restart_fn)
        self._components: Dict[str, tuple] = {}
        self._health_status: Dict[str, ComponentHealth] = {}
        
        # Callbacks for external monitoring integration
        self._on_health_change: Optional[Callable[[str, ComponentHealth], None]] = None
        self._on_restart: Optional[Callable[[str, bool], None]] = None
    
    def register_component(
        self,
        name: str,
        component: Any,
        health_check_fn: Optional[Callable[[], bool]] = None,
        restart_fn: Optional[Callable[[], Any]] = None
    ):
        """
        Register a component for health monitoring.
        
        Args:
            name: Component identifier
            component: The component instance
            health_check_fn: Custom health check function (returns True if healthy)
            restart_fn: Function to restart the component
        """
        with self._lock:
            # Default health check: verify component has expected attributes
            if health_check_fn is None:
                health_check_fn = lambda c=component: self._default_health_check(c)
            
            self._components[name] = (component, health_check_fn, restart_fn)
            self._health_status[name] = ComponentHealth(
                name=name,
                status=ComponentStatus.HEALTHY,
                last_check=datetime.now(),
                last_healthy=datetime.now()
            )
            self.logger.debug(f"Registered component for health monitoring: {name}")
    
    def unregister_component(self, name: str):
        """Remove a component from health monitoring."""
        with self._lock:
            self._components.pop(name, None)
            self._health_status.pop(name, None)
            self.logger.debug(f"Unregistered component: {name}")
    
    def _default_health_check(self, component: Any) -> bool:
        """
        Default health check implementation.
        
        Checks:
        1. Component is not None
        2. If component has 'running' attribute, it should be True
        3. If component has 'is_healthy()' method, call it
        4. If component has 'get_status()' method, verify no errors
        """
        try:
            if component is None:
                return False
            
            # Check 'running' attribute
            if hasattr(component, 'running'):
                if not getattr(component, 'running', True):
                    return False
            
            # Check 'is_healthy()' method
            if hasattr(component, 'is_healthy') and callable(getattr(component, 'is_healthy')):
                return component.is_healthy()
            
            # Check 'get_status()' method
            if hasattr(component, 'get_status') and callable(getattr(component, 'get_status')):
                status = component.get_status()
                if isinstance(status, dict):
                    return not status.get('error', False)
            
            # Component exists and has no obvious issues
            return True
            
        except Exception as e:
            self.logger.debug(f"Health check exception: {e}")
            return False
    
    def check_component_health(self, name: str) -> ComponentHealth:
        """
        Perform immediate health check on a specific component.
        
        Args:
            name: Component name to check
            
        Returns:
            ComponentHealth with current status
        """
        with self._lock:
            if name not in self._components:
                return ComponentHealth(name=name, status=ComponentStatus.STOPPED)
            
            component, health_check_fn, _ = self._components[name]
            health = self._health_status[name]
            
            start_time = time.time()
            try:
                is_healthy = health_check_fn()
                response_time = (time.time() - start_time) * 1000
                
                health.last_check = datetime.now()
                health.response_time_ms = response_time
                
                if is_healthy:
                    health.status = ComponentStatus.HEALTHY
                    health.last_healthy = datetime.now()
                    health.consecutive_failures = 0
                    health.error_message = None
                else:
                    health.consecutive_failures += 1
                    health.status = ComponentStatus.DEGRADED if health.consecutive_failures < 2 else ComponentStatus.UNHEALTHY
                    health.error_message = "Health check returned False"
                    
            except Exception as e:
                health.consecutive_failures += 1
                health.status = ComponentStatus.UNHEALTHY
                health.error_message = str(e)
                health.response_time_ms = (time.time() - start_time) * 1000
                health.last_check = datetime.now()
                self.logger.warning(f"Health check failed for {name}: {e}")
            
            # Notify listeners
            if self._on_health_change:
                try:
                    self._on_health_change(name, health)
                except Exception as e:
                    self.logger.error(f"Health change callback error: {e}")
            
            return health
    
    def restart_component(self, name: str, force: bool = False) -> bool:
        """
        Attempt to restart a component.
        
        Args:
            name: Component name to restart
            force: Force restart even if within backoff period
            
        Returns:
            True if restart was successful
        """
        with self._lock:
            if name not in self._components:
                self.logger.warning(f"Cannot restart unknown component: {name}")
                return False
            
            component, health_check_fn, restart_fn = self._components[name]
            health = self._health_status[name]
            
            # Check if restart function is available
            if restart_fn is None:
                self.logger.warning(f"No restart function registered for {name}")
                return False
            
            # Check restart limits
            if not force and health.restart_count >= self.max_restart_attempts:
                self.logger.error(f"Component {name} exceeded max restart attempts ({self.max_restart_attempts})")
                return False
            
            # Apply backoff if not forced
            if not force and health.restart_count > 0:
                backoff = self.restart_backoff_base * (2 ** (health.restart_count - 1))
                self.logger.info(f"Waiting {backoff:.1f}s before restarting {name}...")
                time.sleep(backoff)
            
            health.status = ComponentStatus.RESTARTING
            self.logger.info(f"Restarting component: {name} (attempt {health.restart_count + 1})")
            
            try:
                # Stop component if it has a stop method
                if hasattr(component, 'stop') and callable(getattr(component, 'stop')):
                    try:
                        component.stop()
                    except Exception as e:
                        self.logger.debug(f"Error stopping {name}: {e}")
                
                # Execute restart function
                new_component = restart_fn()
                
                # Update component reference if restart returned a new instance
                if new_component is not None:
                    self._components[name] = (new_component, health_check_fn, restart_fn)
                
                health.restart_count += 1
                
                # Verify health after restart
                time.sleep(1)  # Brief pause for component startup
                post_health = self.check_component_health(name)
                
                success = post_health.status == ComponentStatus.HEALTHY
                
                if success:
                    self.logger.info(f"✓ Successfully restarted {name}")
                    health.consecutive_failures = 0
                else:
                    self.logger.warning(f"Component {name} still unhealthy after restart")
                
                # Notify listeners
                if self._on_restart:
                    try:
                        self._on_restart(name, success)
                    except Exception as e:
                        self.logger.error(f"Restart callback error: {e}")
                
                return success
                
            except Exception as e:
                health.status = ComponentStatus.UNHEALTHY
                health.error_message = f"Restart failed: {e}"
                health.restart_count += 1
                self.logger.error(f"Failed to restart {name}: {e}")
                return False
    
    def _monitor_loop(self):
        """Background thread that periodically checks all component health."""
        self.logger.info(f"Health monitor started (interval: {self.check_interval}s)")
        
        while self._running:
            try:
                # Check all registered components
                with self._lock:
                    component_names = list(self._components.keys())
                
                for name in component_names:
                    if not self._running:
                        break
                    
                    health = self.check_component_health(name)
                    
                    # Auto-restart unhealthy components
                    if health.status == ComponentStatus.UNHEALTHY:
                        if health.restart_count < self.max_restart_attempts:
                            self.logger.warning(f"Component {name} unhealthy, attempting restart...")
                            self.restart_component(name)
                        else:
                            self.logger.error(f"Component {name} remains unhealthy (max restarts exceeded)")
                
                # Wait for next check interval
                for _ in range(int(self.check_interval)):
                    if not self._running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(5)
        
        self.logger.info("Health monitor stopped")
    
    def start(self):
        """Start the health monitoring background thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="HealthMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.info("✓ Health monitor started")
    
    def stop(self):
        """Stop the health monitoring thread."""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        self.logger.info("Health monitor stopped")
    
    def get_all_health(self) -> Dict[str, ComponentHealth]:
        """Get health status for all registered components."""
        with self._lock:
            return dict(self._health_status)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of system health for dashboard/API."""
        with self._lock:
            total = len(self._health_status)
            healthy = sum(1 for h in self._health_status.values() if h.status == ComponentStatus.HEALTHY)
            degraded = sum(1 for h in self._health_status.values() if h.status == ComponentStatus.DEGRADED)
            unhealthy = sum(1 for h in self._health_status.values() if h.status == ComponentStatus.UNHEALTHY)
            
            return {
                'total_components': total,
                'healthy': healthy,
                'degraded': degraded,
                'unhealthy': unhealthy,
                'health_percentage': (healthy / total * 100) if total > 0 else 100,
                'components': {
                    name: {
                        'status': h.status.value,
                        'last_check': h.last_check.isoformat() if h.last_check else None,
                        'consecutive_failures': h.consecutive_failures,
                        'restart_count': h.restart_count,
                        'response_time_ms': h.response_time_ms
                    }
                    for name, h in self._health_status.items()
                }
            }
    
    def set_callbacks(
        self,
        on_health_change: Optional[Callable[[str, ComponentHealth], None]] = None,
        on_restart: Optional[Callable[[str, bool], None]] = None
    ):
        """Set callback functions for health events."""
        self._on_health_change = on_health_change
        self._on_restart = on_restart


class FalconOneOrchestrator:
    """Main orchestrator for FalconOne operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)
        
        # Setup logging
        log_level = self.config.get('logging.level', 'INFO')
        log_file = self.config.get('logging.file', 'logs/falconone.log')
        self.root_logger = setup_logger(log_level, log_file)
        
        self.logger = ModuleLogger('Orchestrator', self.root_logger)
        
        # Setup audit logging
        audit_dir = self.config.get('logging.audit_dir', 'logs/audit')
        self.audit_logger = AuditLogger(audit_dir)
        
        # Initialize state
        self.running = False
        self.components_initialized = False
        self.active_monitors = {}
        self.components = {}  # Track all initialized components
        
        # Initialize Signal Bus (v1.4.1 optimization)
        self.signal_bus = SignalBus(
            buffer_size=self.config.get('signal_bus.buffer_size', 10000),
            enable_encryption=self.config.get('signal_bus.enable_encryption', False)
        )
        self.logger.info("Signal Bus initialized")
        
        # Dynamic scaling support (v1.8.0)
        self.resource_monitor = None
        self.scaling_thresholds = {
            'cpu_high': self.config.get('orchestrator.scaling_thresholds.cpu_high', 0.85),
            'memory_high': self.config.get('orchestrator.scaling_thresholds.memory_high', 0.80),
            'anomaly_rate_high': self.config.get('orchestrator.scaling_thresholds.anomaly_rate_high', 0.20)
        }
        self.scaling_enabled = self.config.get('orchestrator.dynamic_scaling', True)
        
        # Component references (lazy initialization)
        self.sdr_manager = None
        self.gsm_monitor = None
        self.cdma_monitor = None
        self.umts_monitor = None
        self.lte_monitor = None
        self.fiveg_monitor = None
        self.sixg_monitor = None
        self.exploit_engine = None
        self.signal_classifier = None
        self.ric_optimizer = None
        self.crypto_analyzer = None
        self.detector_scanner = None
        self.sustainability_monitor = None
        self.federated_coordinator = None
        self.kpi_monitor = None
        
        # LE Mode components (v1.8.1)
        self.evidence_chain = None
        self.intercept_enhancer = None
        self._initialize_le_mode()
        
        # Health Monitor (v1.9.2) - Periodic component health checks and auto-restart
        health_check_interval = self.config.get('orchestrator.health_check_interval', 30.0)
        max_restart_attempts = self.config.get('orchestrator.max_restart_attempts', 3)
        self.health_monitor = HealthMonitor(
            check_interval=health_check_interval,
            max_restart_attempts=max_restart_attempts,
            restart_backoff_base=self.config.get('orchestrator.restart_backoff_base', 5.0),
            logger=self.root_logger
        )
        
        # Set health monitor callbacks for audit logging
        self.health_monitor.set_callbacks(
            on_health_change=self._on_component_health_change,
            on_restart=self._on_component_restart
        )
        self.logger.info("Health Monitor initialized")
        
        # Safety checks
        self._perform_safety_checks()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("FalconOne Orchestrator initialized successfully")
        self.audit_logger.log_event('SYSTEM_INIT', 'Orchestrator initialized v1.4')
    
    def _perform_safety_checks(self):
        """Perform safety and compliance checks"""
        self.logger.info("Performing safety checks...")
        
        # Check for Faraday cage (if required)
        if self.config.get('safety.require_faraday_cage', True):
            if not self._detect_faraday_cage():
                error_msg = "⚠️  SAFETY VIOLATION: Faraday cage not detected!"
                self.logger.critical(error_msg)
                self.logger.critical("For ethical compliance, set FALCONONE_FARADAY_CAGE=true or disable in config")
                self.audit_logger.log_event('SAFETY_VIOLATION', 'Faraday cage required but not detected')
                
                raise SafetyViolation("Faraday cage required but not detected. Set FALCONONE_FARADAY_CAGE=true")
        
        # Verify audit logging
        if self.config.get('safety.audit_logging', True):
            try:
                self.audit_logger.log_event('SAFETY_CHECK', 'Audit logging functional')
            except Exception as e:
                self.logger.critical(f"Audit logging failure: {e}")
                raise SafetyViolation(f"Audit logging not functional: {e}")
        
        self.logger.info("✓ Safety checks passed")
    
    def _on_component_health_change(self, name: str, health: ComponentHealth):
        """
        Callback when component health status changes.
        Logs health changes to audit trail for compliance.
        
        Args:
            name: Component name
            health: Updated health status
        """
        if health.status in (ComponentStatus.DEGRADED, ComponentStatus.UNHEALTHY):
            self.logger.warning(f"Component health degraded: {name} → {health.status.value}")
            self.audit_logger.log_event(
                'COMPONENT_HEALTH_DEGRADED',
                f'{name} is {health.status.value}',
                consecutive_failures=health.consecutive_failures,
                error=health.error_message
            )
        elif health.status == ComponentStatus.HEALTHY and health.consecutive_failures > 0:
            self.logger.info(f"Component recovered: {name} → healthy")
            self.audit_logger.log_event('COMPONENT_RECOVERED', f'{name} is now healthy')
    
    def _on_component_restart(self, name: str, success: bool):
        """
        Callback when a component restart is attempted.
        
        Args:
            name: Component name
            success: Whether restart was successful
        """
        if success:
            self.logger.info(f"✓ Component restart successful: {name}")
            self.audit_logger.log_event('COMPONENT_RESTART_SUCCESS', name)
        else:
            self.logger.error(f"✗ Component restart failed: {name}")
            self.audit_logger.log_event('COMPONENT_RESTART_FAILED', name)
    
    def _create_component_restart_fn(self, component_name: str) -> Callable:
        """
        Create a restart function for a specific component.
        
        Args:
            component_name: Name of the component to create restart function for
            
        Returns:
            Callable that restarts the component
        """
        def restart_fn():
            self.logger.info(f"Executing restart for {component_name}...")
            try:
                # Attempt to reinitialize the component
                return self._reinitialize_component(component_name)
            except Exception as e:
                self.logger.error(f"Restart function failed for {component_name}: {e}")
                return None
        return restart_fn
    
    def _reinitialize_component(self, component_name: str) -> Any:
        """
        Reinitialize a specific component.
        
        Args:
            component_name: Name of the component to reinitialize
            
        Returns:
            New component instance or None if failed
        """
        # Component initialization mapping
        init_map = {
            'gsm_monitor': self._init_gsm_monitor,
            'lte_monitor': self._init_lte_monitor,
            'fiveg_monitor': self._init_fiveg_monitor,
            'sixg_monitor': self._init_sixg_monitor,
            'signal_classifier': self._init_signal_classifier,
            'exploit_engine': self._init_exploit_engine,
            'ric_optimizer': self._init_ric_optimizer,
            'detector_scanner': self._init_detector_scanner,
            'sustainability_monitor': self._init_sustainability_monitor,
        }
        
        if component_name in init_map:
            try:
                new_component = init_map[component_name]()
                if new_component:
                    setattr(self, component_name, new_component)
                    self.components[component_name] = new_component
                    return new_component
            except Exception as e:
                self.logger.error(f"Failed to reinitialize {component_name}: {e}")
        
        return None
    
    # Component initialization helpers for restart functionality
    def _init_gsm_monitor(self):
        """Initialize GSM monitor component."""
        from ..monitoring.gsm_monitor import GSMMonitor
        if self.config.get('monitoring.gsm.enabled', True) and self.sdr_manager:
            return GSMMonitor(self.sdr_manager, self.config, self.root_logger)
        return None
    
    def _init_lte_monitor(self):
        """Initialize LTE monitor component."""
        from ..monitoring.lte_monitor import LTEMonitor
        if self.config.get('monitoring.lte.enabled', True) and self.sdr_manager:
            return LTEMonitor(self.sdr_manager, self.config, self.root_logger)
        return None
    
    def _init_fiveg_monitor(self):
        """Initialize 5G monitor component."""
        from ..monitoring.fiveg_monitor import FiveGMonitor
        if self.config.get('monitoring.5g.enabled', True) and self.sdr_manager:
            return FiveGMonitor(self.sdr_manager, self.config, self.root_logger)
        return None
    
    def _init_sixg_monitor(self):
        """Initialize 6G monitor component."""
        from ..monitoring.sixg_monitor import SixGMonitor
        if self.config.get('monitoring.6g.enabled', False) and self.sdr_manager:
            return SixGMonitor(self.sdr_manager, self.config, self.root_logger)
        return None
    
    def _init_signal_classifier(self):
        """Initialize signal classifier component."""
        from ..ai.signal_classifier import SignalClassifier
        if self.config.get('ai.enabled', True):
            return SignalClassifier(self.config, self.root_logger)
        return None
    
    def _init_exploit_engine(self):
        """Initialize exploit engine component."""
        from ..exploit.exploit_engine import ExploitationEngine
        if self.config.get('exploitation.enabled', True):
            return ExploitationEngine(self.config, self.root_logger)
        return None
    
    def _init_ric_optimizer(self):
        """Initialize RIC optimizer component."""
        from ..ai.ric_optimizer import RICOptimizer
        if self.config.get('ai.ric_enabled', True):
            return RICOptimizer(self.config, self.root_logger)
        return None
    
    def _init_detector_scanner(self):
        """Initialize detector scanner component."""
        from ..core.detector_scanner import DetectorScanner
        if self.config.get('detection.enabled', True):
            return DetectorScanner(self.config, self.root_logger)
        return None
    
    def _init_sustainability_monitor(self):
        """Initialize sustainability monitor component."""
        from ..utils.sustainability import SustainabilityMonitor
        if self.config.get('sustainability.enabled', False):
            return SustainabilityMonitor(self.config, self.root_logger)
        return None

    def _detect_faraday_cage(self) -> bool:
        """
        Detect Faraday cage presence
        
        Returns:
            True if detected, False otherwise
        """
        # Check environment variable
        env_cage = os.getenv('FALCONONE_FARADAY_CAGE', 'false').lower()
        if env_cage in ('true', '1', 'yes'):
            return True
        
        # Basic RF isolation detection
        try:
            # Check for extremely low ambient RF power (placeholder for actual SDR check)
            # In a real implementation, this would scan for cellular signals
            rf_power = self._measure_ambient_rf_power()
            if rf_power < -80:  # Very low signal strength
                self.logger.info("Low RF power detected, possible Faraday cage")
                return True
            
            # Check GPS signal loss (if GPS module available)
            if hasattr(self, 'geolocation_adapter') and self.geolocation_adapter:
                gps_lock = self.geolocation_adapter.check_gps_lock()
                if not gps_lock:
                    self.logger.info("GPS signal loss detected, possible Faraday cage")
                    return True
                    
        except Exception as e:
            self.logger.warning(f"RF detection failed: {e}")
        
        return False
    
    def _measure_ambient_rf_power(self) -> float:
        """
        Measure ambient RF power for Faraday cage detection.
        
        Attempts to use SDR hardware if available, otherwise falls back
        to configuration or environment-based detection.
        
        Returns:
            RF power in dBm (lower values indicate better shielding)
        """
        # Check for SDR manager availability
        if hasattr(self, 'sdr_manager') and self.sdr_manager:
            try:
                # Attempt real SDR measurement
                active_device = getattr(self.sdr_manager, 'active_device', None)
                if active_device and hasattr(active_device, 'read_samples'):
                    # Configure for wideband power measurement
                    active_device.configure(
                        sample_rate=2e6,  # 2 MHz
                        center_freq=900e6,  # GSM band
                        bandwidth=1e6,
                        gain=40,
                        channel=0
                    )
                    
                    if active_device.start_stream():
                        samples = active_device.read_samples(10000)
                        if samples is not None and len(samples) > 0:
                            # Calculate power from IQ samples
                            import numpy as np
                            power_linear = np.mean(np.abs(samples) ** 2)
                            power_dbm = 10 * np.log10(power_linear + 1e-12) + 30  # Convert to dBm
                            active_device.stop_stream()
                            self.logger.debug(f"Measured ambient RF power: {power_dbm:.1f} dBm")
                            return power_dbm
                        active_device.stop_stream()
            except Exception as e:
                self.logger.debug(f"SDR RF measurement failed: {e}")
        
        # Fallback: Check environment/config for simulated mode
        simulated_power = self.config.get('safety.simulated_rf_power', None)
        if simulated_power is not None:
            self.logger.debug(f"Using configured RF power: {simulated_power} dBm")
            return float(simulated_power)
        
        # Environment variable override
        env_power = os.getenv('FALCONONE_SIMULATED_RF_POWER')
        if env_power:
            try:
                self.logger.debug(f"Using env RF power: {env_power} dBm")
                return float(env_power)
            except ValueError:
                pass
        
        # Default: Assume normal ambient level (no Faraday cage)
        self.logger.debug("No SDR available, using default RF power: -60.0 dBm")
        return -60.0
    
    def _initialize_le_mode(self):
        """Initialize Law Enforcement Mode components (v1.8.1)"""
        # Check if LE mode is enabled in config
        if not self.config.get('law_enforcement.enabled', False):
            self.logger.info("LE Mode disabled in configuration")
            return
        
        try:
            from ..utils.evidence_chain import EvidenceChain
            from ..le.intercept_enhancer import InterceptEnhancer
            
            # Initialize evidence chain
            self.logger.info("Initializing Evidence Chain (LE Mode)...")
            self.evidence_chain = EvidenceChain(self.config.data, self.root_logger)
            
            # Initialize intercept enhancer (orchestrator linkage deferred until initialize_components)
            self.logger.info("Initializing Intercept Enhancer (LE Mode)...")
            self.intercept_enhancer = InterceptEnhancer(
                self.config.data, 
                self.root_logger,
                orchestrator=None  # Will link after exploit engine initialized
            )
            
            self.logger.info("✓ LE Mode components initialized (exploit linkage pending)")
            self.audit_logger.log_event('LE_MODE_INIT', 'Law Enforcement Mode components initialized')
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LE Mode: {e}")
            self.logger.warning("Continuing without LE Mode capabilities")

    def _initialize_component_with_retry(self, component_name: str, initializer: callable, 
                                         max_retries: int = 3, base_delay: float = 1.0) -> any:
        """
        Initialize a component with automatic retry on failure (v1.9.1)
        
        Args:
            component_name: Name of the component for logging
            initializer: Callable that creates the component
            max_retries: Maximum retry attempts
            base_delay: Base delay between retries (exponential backoff)
        
        Returns:
            Initialized component or None if all retries failed
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                component = initializer()
                if attempt > 0:
                    self.logger.info(f"✓ {component_name} initialized after {attempt + 1} attempts")
                return component
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Failed to initialize {component_name} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to initialize {component_name} after {max_retries + 1} attempts: {e}")
        
        return None
    
    def initialize_components(self):
        """Initialize all system components (lazy initialization with complete integration)"""
        if self.components_initialized:
            return
        
        self.logger.info("Initializing components...")
        
        # v1.9.1: Get retry configuration
        max_retries = self.config.get('orchestrator.init_retry.max_retries', 3)
        base_delay = self.config.get('orchestrator.init_retry.base_delay_sec', 1.0)
        
        try:
            # Import components (lazy to avoid circular imports)
            from ..sdr.sdr_layer import SDRManager
            from ..exploit.exploit_engine import ExploitationEngine
            from ..ai.signal_classifier import SignalClassifier
            
            # 1. SDR Manager (Foundation) - with auto-retry
            self.logger.info("Initializing SDR Manager...")
            self.sdr_manager = self._initialize_component_with_retry(
                "SDR Manager",
                lambda: SDRManager(self.config, self.root_logger),
                max_retries=max_retries,
                base_delay=base_delay
            )
            self.components['sdr_manager'] = self.sdr_manager
            
            # 2. Signal Classifier (AI Foundation)
            self.logger.info("Initializing Signal Classifier...")
            self.signal_classifier = SignalClassifier(self.config, self.root_logger)
            self.components['signal_classifier'] = self.signal_classifier
            
            # 3. Exploitation engine
            self.logger.info("Initializing Exploitation Engine...")
            self.exploit_engine = ExploitationEngine(self.config, self.root_logger)
            self.components['exploit_engine'] = self.exploit_engine
            
            # 3b. Link LE Mode intercept enhancer to orchestrator (v1.8.1)
            if self.intercept_enhancer is not None:
                self.logger.info("Linking Intercept Enhancer to Orchestrator...")
                self.intercept_enhancer.orchestrator = self
                self.components['intercept_enhancer'] = self.intercept_enhancer
                self.components['evidence_chain'] = self.evidence_chain
                self.logger.info("✓ LE Mode fully integrated with orchestrator")
            
            # 4. Try to initialize RIC Optimizer
            try:
                from ..ai.ric_optimizer import RICOptimizer
                if self.config.get('ai.enable_marl', True):
                    self.logger.info("Initializing RIC Optimizer...")
                    self.ric_optimizer = RICOptimizer(self.config, self.root_logger)
                    self.components['ric_optimizer'] = self.ric_optimizer
            except ImportError:
                self.logger.warning("RIC Optimizer not available")
            
            # 5. Try to initialize monitoring components
            try:
                from ..monitoring.gsm_monitor import GSMMonitor
                if self.config.get('monitoring.gsm.enabled', True):
                    self.logger.info("Initializing GSM Monitor...")
                    self.gsm_monitor = GSMMonitor(self.config, self.root_logger, self.sdr_manager)
                    self.components['gsm_monitor'] = self.gsm_monitor
            except ImportError as e:
                self.logger.warning(f"GSM monitor not available: {e}")
            
            try:
                from ..monitoring.cdma_monitor import CDMAMonitor
                if self.config.get('monitoring.cdma.enabled', False):
                    self.logger.info("Initializing CDMA Monitor...")
                    self.cdma_monitor = CDMAMonitor(self.config, self.root_logger, self.sdr_manager)
                    self.components['cdma_monitor'] = self.cdma_monitor
            except ImportError as e:
                self.logger.warning(f"CDMA monitor not available: {e}")
            
            try:
                from ..monitoring.umts_monitor import UMTSMonitor
                if self.config.get('monitoring.umts.enabled', True):
                    self.logger.info("Initializing UMTS Monitor...")
                    self.umts_monitor = UMTSMonitor(self.config, self.root_logger, self.sdr_manager)
                    self.components['umts_monitor'] = self.umts_monitor
            except ImportError as e:
                self.logger.warning(f"UMTS monitor not available: {e}")
            
            try:
                from ..monitoring.lte_monitor import LTEMonitor
                if self.config.get('monitoring.lte.enabled', True):
                    self.logger.info("Initializing LTE Monitor...")
                    self.lte_monitor = LTEMonitor(self.config, self.root_logger, self.sdr_manager)
                    self.components['lte_monitor'] = self.lte_monitor
            except ImportError as e:
                self.logger.warning(f"LTE monitor not available: {e}")
            
            try:
                from ..monitoring.fiveg_monitor import FiveGMonitor
                if self.config.get('monitoring.5g.enabled', True):
                    self.logger.info("Initializing 5G Monitor...")
                    self.fiveg_monitor = FiveGMonitor(self.config, self.root_logger, self.sdr_manager)
                    self.components['fiveg_monitor'] = self.fiveg_monitor
            except ImportError as e:
                self.logger.warning(f"5G monitor not available: {e}")
            
            try:
                from ..monitoring.sixg_monitor import SixGMonitor
                if self.config.get('monitoring.6g.enabled', False):
                    self.logger.info("Initializing 6G Monitor...")
                    self.sixg_monitor = SixGMonitor(self.config, self.root_logger, self.sdr_manager)
                    self.components['sixg_monitor'] = self.sixg_monitor
            except ImportError as e:
                self.logger.warning(f"6G monitor not available: {e}")
            
            # 6. Try to initialize Crypto Analyzer
            try:
                from ..crypto.analyzer import CryptoAnalyzer
                if self.config.get('crypto.enabled', True):
                    self.logger.info("Initializing Crypto Analyzer...")
                    self.crypto_analyzer = CryptoAnalyzer(self.config, self.root_logger)
                    self.components['crypto_analyzer'] = self.crypto_analyzer
            except ImportError as e:
                self.logger.warning(f"Crypto Analyzer not available: {e}")
            
            # 7. Try to initialize Detector Scanner (Phase 5)
            try:
                from ..core.detector_scanner import DetectorScanner
                if self.config.get('detection.enabled', True):
                    self.logger.info("Initializing Detector Scanner...")
                    self.detector_scanner = DetectorScanner(self.config, self.root_logger)
                    self.components['detector_scanner'] = self.detector_scanner
            except ImportError as e:
                self.logger.warning(f"Detector Scanner not available: {e}")
            
            # 8. Try to initialize Sustainability Monitor
            try:
                from ..utils.sustainability import SustainabilityMonitor
                if self.config.get('sustainability.enabled', False):
                    self.logger.info("Initializing Sustainability Monitor...")
                    self.sustainability_monitor = SustainabilityMonitor(self.config, self.root_logger)
                    self.components['sustainability_monitor'] = self.sustainability_monitor
            except ImportError as e:
                self.logger.warning(f"Sustainability Monitor not available: {e}")
            
            # 8b. Try to initialize KPI Monitor
            try:
                from ..ai.kpi_monitor import KPIMonitor
                if self.config.get('ai.kpi_monitoring', True):
                    self.logger.info("Initializing KPI Monitor...")
                    self.kpi_monitor = KPIMonitor(self.config, self.root_logger)
                    self.components['kpi_monitor'] = self.kpi_monitor
            except ImportError as e:
                self.logger.warning(f"KPI Monitor not available: {e}")
            
            # 9. Try to initialize Federated Coordinator (Cloud)
            try:
                from ..ai.federated_coordinator import FederatedCoordinator
                if self.config.get('api.coordinator_enabled', False):
                    self.logger.info("Initializing Federated Coordinator...")
                    self.federated_coordinator = FederatedCoordinator(self.config, self.root_logger)
                    self.components['federated_coordinator'] = self.federated_coordinator
            except ImportError as e:
                self.logger.warning(f"Federated Coordinator not available: {e}")
            
            # 10. Try to initialize Environmental Adapter (v1.7.0)
            try:
                from ..geolocation.environmental_adapter import EnvironmentalAdapter
                if self.config.get('geolocation.environmental_adaptation', True):
                    self.logger.info("Initializing Environmental Adapter...")
                    self.environmental_adapter = EnvironmentalAdapter(self.root_logger)
                    self.components['environmental_adapter'] = self.environmental_adapter
            except ImportError as e:
                self.logger.warning(f"Environmental Adapter not available: {e}")
            
            # 11. Try to initialize Profiler (v1.7.0)
            try:
                from ..monitoring.profiler import Profiler
                if self.config.get('monitoring.profiling_enabled', True):
                    self.logger.info("Initializing Profiler...")
                    self.profiler = Profiler(self.root_logger)
                    self.components['profiler'] = self.profiler
            except ImportError as e:
                self.logger.warning(f"Profiler not available: {e}")
            
            # 12. Try to initialize E2E Validator (v1.7.0)
            try:
                from ..tests.e2e_validation import E2EValidator
                if self.config.get('testing.e2e_validation', False):
                    self.logger.info("Initializing E2E Validator...")
                    self.e2e_validator = E2EValidator(self.root_logger)
                    self.components['e2e_validator'] = self.e2e_validator
            except ImportError as e:
                self.logger.warning(f"E2E Validator not available: {e}")
            
            # 13. Try to initialize Model Zoo (v1.7.0)
            try:
                from ..ai.model_zoo import ModelZoo
                if self.config.get('ai.model_zoo_enabled', True):
                    self.logger.info("Initializing Model Zoo...")
                    model_cache_dir = self.config.get('ai.model_cache_dir', '/var/cache/falconone/models')
                    self.model_zoo = ModelZoo(model_cache_dir, self.root_logger)
                    self.components['model_zoo'] = self.model_zoo
            except ImportError as e:
                self.logger.warning(f"Model Zoo not available: {e}")
            
            # 14. Setup cross-module integrations
            self._setup_integrations()
            
            # 15. Register components with Health Monitor (v1.9.2)
            self._register_components_for_health_monitoring()
            
            self.components_initialized = True
            self.logger.info(f"✓ Initialized {len(self.components)} components successfully")
            self.audit_logger.log_event('COMPONENTS_INIT', f'{len(self.components)} components loaded')
            
        except Exception as e:
            self.logger.critical(f"Component initialization failed: {e}")
            import traceback
            traceback.print_exc()
            raise IntegrationError(f"Failed to initialize components: {e}")
    
    def _setup_integrations(self):
        """
        Setup cross-module data flow and integrations
        CRITICAL: Enables Monitor → Classifier → RIC → Exploit pipeline
        """
        self.logger.info("Setting up cross-module integrations...")
        
        # Integration 1: Connect monitors to Signal Bus
        for monitor_name in ['gsm_monitor', 'lte_monitor', 'fiveg_monitor', 'sixg_monitor']:
            if hasattr(self, monitor_name) and getattr(self, monitor_name):
                monitor = getattr(self, monitor_name)
                # Subscribe signal bus to monitor outputs
                generation = monitor_name.replace('_monitor', '').upper()
                if hasattr(monitor, 'set_signal_bus'):
                    monitor.set_signal_bus(self.signal_bus)
                    self.logger.info(f"✓ {monitor_name} → Signal Bus")
                elif hasattr(monitor, 'signal_callback'):
                    # Legacy callback support
                    monitor.signal_callback = lambda sig, gen=generation: self.signal_bus.publish(gen, sig)
                    self.logger.info(f"✓ {monitor_name} → Signal Bus (callback)")
        
        # Integration 2: Classifier subscribes to Signal Bus
        if self.signal_classifier and self.signal_bus:
            # Classifier processes all signals
            def classifier_callback(signal_data):
                try:
                    result = self.signal_classifier.classify([signal_data])
                    if result:
                        signal_data['classification'] = result[0]
                        # Check for anomalies
                        anomaly = self.signal_classifier.detect_anomaly(
                            signal_data.get('iq_data', np.zeros((1024, 2))),
                            signal_data.get('kpis', {})
                        )
                        if anomaly.get('anomaly_detected'):
                            self.logger.warning(f"Anomaly detected: {anomaly}")
                            signal_data['anomaly'] = anomaly
                except Exception as e:
                    self.logger.error(f"Classifier callback error: {e}")
            
            self.signal_bus.subscribe('SignalClassifier', '*', classifier_callback)
            self.logger.info("✓ Signal Bus → Classifier integration")
        
        # Integration 3: RIC Optimizer subscribes to anomalies
        if self.ric_optimizer and self.signal_classifier:
            # Link anomaly detection to RIC adaptation
            def ric_anomaly_handler(signal_data):
                if 'anomaly' in signal_data and signal_data['anomaly'].get('anomaly_detected'):
                    try:
                        adaptation = self.signal_classifier.adapt_to_anomaly(signal_data['anomaly'])
                        if hasattr(self.ric_optimizer, 'apply_adaptation'):
                            self.ric_optimizer.apply_adaptation(adaptation)
                    except Exception as e:
                        self.logger.error(f"RIC anomaly handler error: {e}")
            
            self.signal_bus.subscribe('RICOptimizer', '*', ric_anomaly_handler)
            self.logger.info("✓ Classifier → RIC integration")
        
        # Integration 4: Exploit Engine pre-check with Detector Scanner
        if self.exploit_engine and self.detector_scanner:
            self.exploit_engine.detector_scanner = self.detector_scanner
            self.logger.info("✓ Exploit → Detector Scanner integration")
        
        # Integration 5: Sustainability tracking for all components
        if self.sustainability_monitor:
            for component_name, component in self.components.items():
                if hasattr(component, 'set_sustainability_tracker'):
                    try:
                        component.set_sustainability_tracker(self.sustainability_monitor)
                        self.logger.info(f"✓ {component_name} → Sustainability tracking")
                    except Exception as e:
                        self.logger.debug(f"Could not set sustainability for {component_name}: {e}")
        
        # Integration 6: Federated learning coordinator
        if self.federated_coordinator and self.signal_classifier:
            if hasattr(self.federated_coordinator, 'register_model'):
                self.federated_coordinator.register_model('signal_classifier', self.signal_classifier)
                self.logger.info("✓ Classifier → Federated Coordinator")
        
        self.logger.info(f"✓ Cross-module integrations complete ({len(self.components)} components)")
        self.audit_logger.log_event('INTEGRATIONS_SETUP', 'Cross-module data flow established')
    
    def _register_components_for_health_monitoring(self):
        """
        Register all initialized components with the Health Monitor (v1.9.2).
        
        Creates custom health check and restart functions for each component
        based on their specific interfaces.
        """
        self.logger.info("Registering components for health monitoring...")
        
        registered_count = 0
        
        for component_name, component in self.components.items():
            try:
                # Create custom health check based on component type
                health_check_fn = self._create_health_check_fn(component_name, component)
                
                # Create restart function
                restart_fn = self._create_component_restart_fn(component_name)
                
                # Register with health monitor
                self.health_monitor.register_component(
                    name=component_name,
                    component=component,
                    health_check_fn=health_check_fn,
                    restart_fn=restart_fn
                )
                registered_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to register {component_name} for health monitoring: {e}")
        
        self.logger.info(f"✓ Registered {registered_count}/{len(self.components)} components for health monitoring")
    
    def _create_health_check_fn(self, component_name: str, component: Any) -> Callable[[], bool]:
        """
        Create a customized health check function for a specific component.
        
        Args:
            component_name: Name of the component
            component: The component instance
            
        Returns:
            A callable that returns True if the component is healthy
        """
        def health_check() -> bool:
            try:
                # Check if component has a dedicated is_healthy method
                if hasattr(component, 'is_healthy') and callable(getattr(component, 'is_healthy')):
                    return component.is_healthy()
                
                # Check running state for monitors
                if 'monitor' in component_name:
                    if hasattr(component, 'running'):
                        return getattr(component, 'running', True)
                
                # Check get_status for components with status reporting
                if hasattr(component, 'get_status') and callable(getattr(component, 'get_status')):
                    status = component.get_status()
                    if isinstance(status, dict):
                        # Check for explicit error flags
                        if status.get('error') or status.get('failed'):
                            return False
                        # Check for running state
                        if 'running' in status:
                            return status['running']
                        return True
                
                # For AI components, verify model availability
                if component_name in ('signal_classifier', 'ric_optimizer'):
                    if hasattr(component, 'models_loaded'):
                        return getattr(component, 'models_loaded', False)
                    if hasattr(component, 'model') and component.model is not None:
                        return True
                
                # Default: component exists and is not None
                return component is not None
                
            except Exception as e:
                self.logger.debug(f"Health check exception for {component_name}: {e}")
                return False
        
        return health_check

    def start(self):
        """Start orchestrator and all enabled services"""
        self.logger.info("🚀 Starting FalconOne v1.4.1...")
        
        # Initialize components if not already done
        if not self.components_initialized:
            self.initialize_components()
        
        self.running = True
        
        # Start monitoring
        self._start_monitoring()
        
        # Start AI optimization
        if self.ric_optimizer:
            try:
                if hasattr(self.ric_optimizer, 'start_autonomous_operations'):
                    self.ric_optimizer.start_autonomous_operations()
                    self.logger.info("✓ RIC Optimizer started")
            except Exception as e:
                self.logger.error(f"RIC Optimizer failed to start: {e}")
        
        # Start federated coordinator (if cloud mode)
        if self.federated_coordinator:
            try:
                if hasattr(self.federated_coordinator, 'start'):
                    self.federated_coordinator.start()
                    self.logger.info("✓ Federated Coordinator started")
            except Exception as e:
                self.logger.error(f"Federated Coordinator failed to start: {e}")
        
        # Start sustainability monitoring
        if self.sustainability_monitor:
            try:
                if hasattr(self.sustainability_monitor, 'start'):
                    self.sustainability_monitor.start()
                    self.logger.info("✓ Sustainability Monitor started")
            except Exception as e:
                self.logger.error(f"Sustainability Monitor failed to start: {e}")
        
        # Start Health Monitor (v1.9.2) - Periodic component health checks
        if self.health_monitor:
            self.health_monitor.start()
            health_interval = self.config.get('orchestrator.health_check_interval', 30.0)
            self.logger.info(f"✓ Health Monitor started (interval: {health_interval}s)")
        
        self.logger.info("✓ FalconOne v1.9.2 operational")
        self.audit_logger.log_event('SYSTEM_START', 'All services operational', version='1.9.2')
        
        print("\n✅ FalconOne v1.9.2 is now running")
        print(f"📊 Active components: {len(self.components)}")
        print(f"📡 Active monitors: {len(self.active_monitors)}")
        print(f"🔄 Signal Bus: {self.signal_bus.get_stats()['subscribers']} subscribers")
        print(f"💓 Health Monitor: Active")
        print("🔒 Audit logging: Enabled")
        print("\nPress Ctrl+C to stop\n")
    
    def stop(self):
        """Stop all operations gracefully"""
        self.logger.info("⏹ Stopping FalconOne...")
        self.running = False
        
        # Stop Health Monitor first (v1.9.2)
        if self.health_monitor:
            self.health_monitor.stop()
            self.logger.info("Health Monitor stopped")
        
        # Stop all components
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'stop'):
                    component.stop()
                    self.logger.info(f"Stopped {component_name}")
            except Exception as e:
                self.logger.error(f"Error stopping {component_name}: {e}")
        
        # Clear signal bus
        if self.signal_bus:
            self.signal_bus.clear_buffer()
        
        self.logger.info("✓ FalconOne stopped")
        self.audit_logger.log_event('SYSTEM_STOP', 'Graceful shutdown complete')
        
        print("\n⏹  FalconOne stopped successfully")
        
        # Cleanup SDR
        if self.sdr_manager and hasattr(self.sdr_manager, 'cleanup'):
            try:
                self.sdr_manager.cleanup()
            except Exception as e:
                self.logger.warning(f"SDR cleanup error: {e}")
    
    def _start_monitoring(self):
        """Start all enabled monitors"""
        monitors_started = 0
        
        for monitor_name in ['gsm_monitor', 'lte_monitor', 'fiveg_monitor', 'sixg_monitor']:
            if hasattr(self, monitor_name) and getattr(self, monitor_name):
                monitor = getattr(self, monitor_name)
                generation = monitor_name.replace('_monitor', '').upper()
                self.active_monitors[generation] = monitor
                
                # Start monitoring if method exists
                if hasattr(monitor, 'start_monitoring'):
                    try:
                        monitor.start_monitoring()
                        monitors_started += 1
                        self.logger.info(f"✓ Started {generation} monitor")
                    except Exception as e:
                        self.logger.error(f"Failed to start {generation} monitor: {e}")
                else:
                    monitors_started += 1
        
        if monitors_started > 0:
            self.logger.info(f"✓ Started {monitors_started}/{len(self.active_monitors)} monitor(s)")
        else:
            self.logger.warning("No monitors available")
    
    def execute_exploit(self, exploit_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute exploitation operation with pre-checks and adaptation
        
        Args:
            exploit_type: Type of exploit
            params: Parameters
            
        Returns:
            Results dictionary
        """
        self.logger.warning(f"⚠️  Executing exploit: {exploit_type}")
        self.audit_logger.log_event('EXPLOIT_EXECUTE', exploit_type, params=params)
        
        if not self.exploit_engine:
            raise RuntimeError("Exploitation engine not initialized")
        
        # Pre-execution check: Detect if we're being monitored
        if self.detector_scanner:
            try:
                detection_check = self.detector_scanner.scan_for_detectors()
                if detection_check.get('detectors_found', 0) > 0:
                    self.logger.warning(f"⚠️  Detectors found: {detection_check['detectors_found']}")
                    self.logger.warning("Enabling evasion mode...")
                    params['evasion_mode'] = True
                    self.audit_logger.log_event('DETECTOR_FOUND', 'Evasion enabled', 
                                              detectors=detection_check['detectors_found'])
            except Exception as e:
                self.logger.error(f"Detector scan failed: {e}")
        
        # Execute exploit
        result = self.exploit_engine.execute(exploit_type, params)
        
        # Post-execution: Feed result to AI for learning
        if self.ric_optimizer and hasattr(self.ric_optimizer, 'learn_from_exploit'):
            try:
                self.ric_optimizer.learn_from_exploit(exploit_type, result)
            except Exception as e:
                self.logger.error(f"RIC learning failed: {e}")
        
        return result
    
    def _setup_dynamic_scaling(self):
        """Setup dynamic resource allocation based on KPI feedback (v1.8.0)"""
        if not self.scaling_enabled or not self.kpi_monitor:
            return
        
        self.logger.info("Setting up dynamic scaling...")
        
        # Subscribe to KPI alerts for resource allocation
        def scaling_callback(kpi_data):
            try:
                cpu_usage = kpi_data.get('cpu_usage', 0)
                memory_usage = kpi_data.get('memory_usage', 0)
                anomaly_rate = kpi_data.get('anomaly_rate', 0)
                
                # CPU scaling
                if cpu_usage > self.scaling_thresholds['cpu_high']:
                    self._scale_processing_resources('up', 'cpu')
                elif cpu_usage < 0.3:  # Scale down if underutilized
                    self._scale_processing_resources('down', 'cpu')
                
                # ML model scaling
                if anomaly_rate > self.scaling_thresholds['anomaly_rate_high']:
                    self._scale_ml_resources('up')
                
                # Memory management
                if memory_usage > self.scaling_thresholds['memory_high']:
                    self._trigger_memory_optimization()
                    
            except Exception as e:
                self.logger.error(f"Scaling callback error: {e}")
        
        self.signal_bus.subscribe('ResourceScaler', 'KPI', scaling_callback)
        self.logger.info("✓ Dynamic scaling enabled")
    
    def _scale_processing_resources(self, direction: str, resource_type: str):
        """Scale processing resources dynamically"""
        try:
            if resource_type == 'cpu':
                if direction == 'up':
                    # Log recommendation for configuration adjustment
                    current_workers = self.config.get('performance.thread_pool_workers', 4)
                    new_workers = min(current_workers + 2, os.cpu_count())
                    self.logger.info(f"Recommending CPU scaling: {current_workers} -> {new_workers} workers")
                    self.config.set('performance.thread_pool_workers', new_workers)
                    
        except Exception as e:
            self.logger.error(f"Resource scaling error: {e}")
    
    def _scale_ml_resources(self, direction: str):
        """Scale ML inference resources"""
        if not self.signal_classifier:
            return
        
        try:
            if direction == 'up':
                # Switch to lighter quantized model if available
                if hasattr(self.signal_classifier, 'use_quantized_model'):
                    self.signal_classifier.use_quantized_model(True)
                    self.logger.info("Switched to quantized model for performance")
                    
        except Exception as e:
            self.logger.error(f"ML scaling error: {e}")
    
    def _trigger_memory_optimization(self):
        """Trigger memory cleanup when usage is high"""
        import gc
        
        try:
            # Clear signal processing cache if available
            from ..utils.performance import get_cache
            cache = get_cache()
            cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            self.logger.warning("Triggered memory optimization - cache cleared")
        except ImportError:
            # Just do GC if performance utils not available
            gc.collect()
            self.logger.warning("Triggered garbage collection")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with all component details"""
        status = {
            'running': self.running,
            'version': '1.9.2',
            'components_initialized': self.components_initialized,
            'active_monitors': list(self.active_monitors.keys()),
            'total_components': len(self.components),
            'components_loaded': list(self.components.keys()),
            'config_loaded': True,
            'safety': {
                'faraday_cage_required': self.config.get('safety.require_faraday_cage', True),
                'audit_logging': self.config.get('safety.audit_logging', True),
                'faraday_cage_detected': self._detect_faraday_cage()
            }
        }
        
        # Health Monitor status (v1.9.2)
        if self.health_monitor:
            try:
                status['health'] = self.health_monitor.get_health_summary()
            except Exception as e:
                status['health'] = {'error': str(e)}
        
        # SDR status
        if self.sdr_manager:
            try:
                status['sdr'] = {
                    'available_devices': self.sdr_manager.get_available_devices(),
                    'active_device': self.sdr_manager.active_device.device_type if hasattr(self.sdr_manager, 'active_device') and self.sdr_manager.active_device else None
                }
            except Exception as e:
                status['sdr'] = {'error': str(e)}
        
        # Signal Bus stats
        if self.signal_bus:
            status['signal_bus'] = self.signal_bus.get_stats()
        
        # AI status
        if self.ric_optimizer:
            try:
                status['ai'] = {
                    'ric_enabled': True,
                    'autonomous': hasattr(self.ric_optimizer, 'autonomous_mode') and self.ric_optimizer.autonomous_mode
                }
            except Exception:
                pass
        
        # Sustainability metrics
        if self.sustainability_monitor:
            try:
                status['sustainability'] = self.sustainability_monitor.get_metrics()
            except Exception:
                pass
        
        return status
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)


# Alias for backward compatibility
FalconOne = FalconOneOrchestrator
