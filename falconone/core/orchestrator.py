"""
FalconOne Core Orchestrator
Main coordination layer for all operations with safety interlocks
Version 1.4.1: Complete integration with all modules + Signal Bus
"""

import logging
import signal
import sys
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

from ..utils.config import Config
from ..utils.logger import setup_logger, ModuleLogger, AuditLogger
from ..utils.exceptions import SafetyViolation, ConfigurationError, IntegrationError
from .signal_bus import SignalBus


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
        
        # TODO: Implement actual RF isolation detection
        # Could check for:
        # - Extremely low ambient RF power
        # - No detectable cellular signals
        # - GPS signal loss
        
        return False
    
    def initialize_components(self):
        """Initialize all system components (lazy initialization with complete integration)"""
        if self.components_initialized:
            return
        
        self.logger.info("Initializing components...")
        
        try:
            # Import components (lazy to avoid circular imports)
            from ..sdr.sdr_layer import SDRManager
            from ..exploit.exploit_engine import ExploitationEngine
            from ..ai.signal_classifier import SignalClassifier
            
            # 1. SDR Manager (Foundation)
            self.logger.info("Initializing SDR Manager...")
            self.sdr_manager = SDRManager(self.config, self.root_logger)
            self.components['sdr_manager'] = self.sdr_manager
            
            # 2. Signal Classifier (AI Foundation)
            self.logger.info("Initializing Signal Classifier...")
            self.signal_classifier = SignalClassifier(self.config, self.root_logger)
            self.components['signal_classifier'] = self.signal_classifier
            
            # 3. Exploitation engine
            self.logger.info("Initializing Exploitation Engine...")
            self.exploit_engine = ExploitationEngine(self.config, self.root_logger)
            self.components['exploit_engine'] = self.exploit_engine
            
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
        
        self.logger.info("✓ FalconOne v1.4.1 operational")
        self.audit_logger.log_event('SYSTEM_START', 'All services operational', version='1.4.1')
        
        print("\n✅ FalconOne v1.4.1 is now running")
        print(f"📊 Active components: {len(self.components)}")
        print(f"📡 Active monitors: {len(self.active_monitors)}")
        print(f"🔄 Signal Bus: {self.signal_bus.get_stats()['subscribers']} subscribers")
        print("🔒 Audit logging: Enabled")
        print("\nPress Ctrl+C to stop\n")
    
    def stop(self):
        """Stop all operations gracefully"""
        self.logger.info("⏹ Stopping FalconOne...")
        self.running = False
        
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
            'version': '1.4.1',
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
