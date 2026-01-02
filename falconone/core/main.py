"""
FalconOne Main Orchestration Module
Coordinates all subsystems and manages the overall workflow
"""

import sys
import time
import signal
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..core.config import Config
from ..utils.logger import setup_logger, ModuleLogger, AuditLogger
from ..sdr.sdr_layer import SDRManager
from ..monitoring.gsm_monitor import GSMMonitor
from ..monitoring.umts_monitor import UMTSMonitor
from ..monitoring.cdma_monitor import CDMAMonitor
from ..monitoring.lte_monitor import LTEMonitor
from ..monitoring.fiveg_monitor import FiveGMonitor
from ..monitoring.sixg_monitor import SixGMonitor
from ..ai.signal_classifier import SignalClassifier
from ..ai.suci_deconcealment import SUCIDeconcealmentEngine
from ..ai.kpi_monitor import KPIMonitor
from ..geolocation.locator import GeolocatorEngine
from ..voice.interceptor import VoiceInterceptor
from ..crypto.analyzer import CryptoAnalyzer
from ..exploit.exploit_engine import ExploitationEngine


class FalconOne:
    """
    Main FalconOne System Orchestrator
    
    Coordinates multi-generation cellular monitoring, AI/ML processing,
    exploitation, and advanced analytics for research purposes.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize FalconOne system
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)
        self.config.validate()
        
        # Set up logging
        self.logger = setup_logger(
            name='falconone',
            log_dir=self.config.get('system.log_dir', '/var/log/falconone'),
            log_level=self.config.get('system.log_level', 'INFO')
        )
        self.module_logger = ModuleLogger('CORE', self.logger)
        self.audit_logger = AuditLogger()
        
        # System state
        self.running = False
        self.modules: Dict[str, Any] = {}
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.module_logger.info("FalconOne system initializing...", version=self.config.get('system.version'))
        self.audit_logger.log_event('SYSTEM_INIT', 'FalconOne system starting', status='in_progress')
        
        # Initialize subsystems
        self._initialize_subsystems()
    
    def _initialize_subsystems(self):
        """Initialize all subsystems based on configuration"""
        try:
            # 1. Initialize SDR Manager (Hardware Layer)
            self.module_logger.info("Initializing SDR hardware layer...")
            self.modules['sdr'] = SDRManager(self.config, self.logger)
            
            # 2. Initialize Generation-Specific Monitors
            monitors = {}
            
            if self.config.get('monitoring.gsm.enabled'):
                self.module_logger.info("Initializing GSM monitoring...")
                monitors['gsm'] = GSMMonitor(self.config, self.logger, self.modules['sdr'])
            
            if self.config.get('monitoring.umts.enabled'):
                self.module_logger.info("Initializing UMTS monitoring...")
                monitors['umts'] = UMTSMonitor(self.config, self.logger, self.modules['sdr'])
            
            if self.config.get('monitoring.cdma2000.enabled'):
                self.module_logger.info("Initializing CDMA2000 monitoring...")
                monitors['cdma'] = CDMAMonitor(self.config, self.logger, self.modules['sdr'])
            
            if self.config.get('monitoring.lte.enabled'):
                self.module_logger.info("Initializing LTE monitoring...")
                monitors['lte'] = LTEMonitor(self.config, self.logger, self.modules['sdr'])
            
            if self.config.get('monitoring.5g.enabled'):
                self.module_logger.info("Initializing 5G monitoring...")
                monitors['5g'] = FiveGMonitor(self.config, self.logger, self.modules['sdr'])
            
            if self.config.get('monitoring.6g.enabled'):
                self.module_logger.info("Initializing 6G prototyping...")
                monitors['6g'] = SixGMonitor(self.config, self.logger, self.modules['sdr'])
            
            self.modules['monitors'] = monitors
            
            # 3. Initialize AI/ML Components
            if self.config.get('ai_ml.signal_classification.enabled'):
                self.module_logger.info("Initializing AI signal classifier...")
                self.modules['signal_classifier'] = SignalClassifier(self.config, self.logger)
            
            if self.config.get('ai_ml.suci_deconcealment.enabled'):
                self.module_logger.info("Initializing SUCI de-concealment engine...")
                self.modules['suci_engine'] = SUCIDeconcealmentEngine(self.config, self.logger)
            
            if self.config.get('ai_ml.kpi_monitoring.enabled'):
                self.module_logger.info("Initializing KPI monitoring...")
                self.modules['kpi_monitor'] = KPIMonitor(self.config, self.logger)
            
            # 4. Initialize Advanced Modules
            if self.config.get('geolocation.enabled'):
                self.module_logger.info("Initializing geolocation engine...")
                self.modules['geolocator'] = GeolocatorEngine(self.config, self.logger, self.modules['sdr'])
            
            if self.config.get('voice_interception.enabled'):
                self.module_logger.info("Initializing voice interceptor...")
                self.modules['voice_interceptor'] = VoiceInterceptor(self.config, self.logger)
            
            if self.config.get('cryptanalysis.enabled'):
                self.module_logger.info("Initializing cryptanalysis module...")
                self.modules['crypto_analyzer'] = CryptoAnalyzer(self.config, self.logger)
            
            if self.config.get('exploitation.enabled'):
                self.module_logger.info("Initializing exploitation engine...")
                self.modules['exploit_engine'] = ExploitationEngine(self.config, self.logger)
            
            self.module_logger.info("All subsystems initialized successfully", 
                                   module_count=len(self.modules))
            self.audit_logger.log_event('SYSTEM_INIT', 'All subsystems initialized', status='success')
            
        except Exception as e:
            self.module_logger.critical(f"Failed to initialize subsystems: {e}")
            self.audit_logger.log_event('SYSTEM_INIT', f'Initialization failed: {e}', status='failure')
            raise
    
    def start(self):
        """Start the FalconOne system"""
        self.module_logger.info("Starting FalconOne operational mode...")
        self.running = True
        self.audit_logger.log_event('SYSTEM_START', 'FalconOne system started')
        
        try:
            # Start monitoring threads for enabled generations
            for gen_name, monitor in self.modules.get('monitors', {}).items():
                self.module_logger.info(f"Starting {gen_name.upper()} monitoring...")
                monitor.start()
            
            # Main monitoring loop
            self._main_loop()
            
        except Exception as e:
            self.module_logger.error(f"Error in main loop: {e}")
            self.audit_logger.log_event('SYSTEM_ERROR', f'Main loop error: {e}', status='failure')
            raise
        finally:
            self.stop()
    
    def _main_loop(self):
        """Main operational loop"""
        self.module_logger.info("Entering main monitoring loop...")
        
        while self.running:
            try:
                # Process captured data from all monitors
                self._process_captured_data()
                
                # Run AI/ML analysis if enabled
                self._run_ai_analysis()
                
                # Update KPI metrics
                if 'kpi_monitor' in self.modules:
                    self.modules['kpi_monitor'].update_metrics()
                
                # Sleep briefly to avoid busy-waiting
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.module_logger.info("Keyboard interrupt received, shutting down...")
                break
            except Exception as e:
                self.module_logger.error(f"Error in main loop iteration: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _process_captured_data(self):
        """Process data captured by all monitoring modules"""
        for gen_name, monitor in self.modules.get('monitors', {}).items():
            try:
                # Get captured data
                data = monitor.get_captured_data()
                
                if data:
                    self.module_logger.debug(f"Processing {len(data)} items from {gen_name.upper()}")
                    
                    # Log captured identities
                    for item in data:
                        if 'imsi' in item:
                            self.audit_logger.log_event(
                                'IMSI_CAPTURE',
                                f"Captured IMSI from {gen_name.upper()}",
                                target=item['imsi']
                            )
                        
                        if 'tmsi' in item:
                            self.audit_logger.log_event(
                                'TMSI_CAPTURE',
                                f"Captured TMSI from {gen_name.upper()}",
                                target=item['tmsi']
                            )
                    
                    # Pass to AI for classification if enabled
                    if 'signal_classifier' in self.modules:
                        self.modules['signal_classifier'].classify(data)
                    
                    # Attempt geolocation if enabled
                    if 'geolocator' in self.modules:
                        self.modules['geolocator'].locate(data)
            
            except Exception as e:
                self.module_logger.error(f"Error processing {gen_name} data: {e}")
    
    def _run_ai_analysis(self):
        """Run AI/ML analysis on collected data"""
        try:
            # SUCI de-concealment for 5G data
            if 'suci_engine' in self.modules and '5g' in self.modules.get('monitors', {}):
                suci_data = self.modules['monitors']['5g'].get_suci_data()
                if suci_data:
                    results = self.modules['suci_engine'].deonceal(suci_data)
                    self.module_logger.info(f"SUCI de-concealment completed: {len(results)} results")
        
        except Exception as e:
            self.module_logger.error(f"Error in AI analysis: {e}")
    
    def stop(self):
        """Stop the FalconOne system gracefully"""
        self.module_logger.info("Stopping FalconOne system...")
        self.running = False
        
        # Stop all monitors
        for gen_name, monitor in self.modules.get('monitors', {}).items():
            try:
                self.module_logger.info(f"Stopping {gen_name.upper()} monitor...")
                monitor.stop()
            except Exception as e:
                self.module_logger.error(f"Error stopping {gen_name} monitor: {e}")
        
        # Clean up SDR resources
        if 'sdr' in self.modules:
            try:
                self.modules['sdr'].cleanup()
            except Exception as e:
                self.module_logger.error(f"Error cleaning up SDR: {e}")
        
        self.audit_logger.log_event('SYSTEM_STOP', 'FalconOne system stopped')
        self.module_logger.info("FalconOne system stopped successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        self.module_logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status
        
        Returns:
            Dictionary containing system status information
        """
        status = {
            'running': self.running,
            'uptime': None,
            'modules': {},
            'performance': {}
        }
        
        # Get status from each module
        for module_name, module in self.modules.items():
            if hasattr(module, 'get_status'):
                status['modules'][module_name] = module.get_status()
        
        return status
    
    def execute_exploit(self, exploit_type: str, params: Dict[str, Any]):
        """
        Execute a specific exploitation technique
        
        Args:
            exploit_type: Type of exploit (e.g., 'dos', 'downgrade', 'mitm')
            params: Exploit parameters
        """
        if not self.config.get('exploitation.enabled'):
            raise RuntimeError("Exploitation module is not enabled")
        
        if 'exploit_engine' not in self.modules:
            raise RuntimeError("Exploitation engine not initialized")
        
        self.audit_logger.log_event(
            'EXPLOIT_EXECUTION',
            f"Executing {exploit_type} exploit",
            target=params.get('target', 'unknown')
        )
        
        return self.modules['exploit_engine'].execute(exploit_type, params)


def main():
    """Main entry point for FalconOne system"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         FalconOne IMSI/TMSI and SMS Catcher v1.1         ║
    ║                 Multi-Generation SIGINT Platform          ║
    ║                    TOP CONFIDENTIAL                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Initialize and start FalconOne
        falcon = FalconOne()
        falcon.start()
        
    except KeyboardInterrupt:
        print("\n[!] Keyboard interrupt received")
    except Exception as e:
        print(f"\n[!] Critical error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
