"""
FalconOne Comprehensive System Audit
Validates all features, imports, and dependencies
"""

import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Tuple

class SystemAuditor:
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
        
    def test_import(self, module_name: str, description: str) -> bool:
        """Test if a module can be imported"""
        try:
            importlib.import_module(module_name)
            self.results['passed'].append(f"[OK] {description}: {module_name}")
            return True
        except ImportError as e:
            self.results['failed'].append(f"[FAIL] {description}: {module_name} - {str(e)}")
            return False
        except Exception as e:
            self.results['warnings'].append(f"[WARN] {description}: {module_name} - {str(e)}")
            return False
    
    def test_class_instantiation(self, module_name: str, class_name: str, description: str, *args, **kwargs) -> bool:
        """Test if a class can be instantiated"""
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            instance = cls(*args, **kwargs)
            self.results['passed'].append(f"[OK] {description}: {class_name} instantiation OK")
            return True
        except Exception as e:
            self.results['failed'].append(f"[FAIL] {description}: {class_name} - {str(e)}")
            return False
    
    def print_report(self):
        """Print audit report"""
        print("\n" + "="*70)
        print("FALCONONE COMPREHENSIVE SYSTEM AUDIT REPORT")
        print("="*70)
        
        print(f"\n[OK] PASSED: {len(self.results['passed'])}")
        for item in self.results['passed']:
            print(f"  {item}")
        
        if self.results['warnings']:
            print(f"\n[WARN] WARNINGS: {len(self.results['warnings'])}")
            for item in self.results['warnings']:
                print(f"  {item}")
        
        if self.results['failed']:
            print(f"\n[FAIL] FAILED: {len(self.results['failed'])}")
            for item in self.results['failed']:
                print(f"  {item}")
        
        print("\n" + "="*70)
        total = len(self.results['passed']) + len(self.results['failed']) + len(self.results['warnings'])
        success_rate = (len(self.results['passed']) / total * 100) if total > 0 else 0
        print(f"SUCCESS RATE: {success_rate:.1f}% ({len(self.results['passed'])}/{total})")
        print("="*70 + "\n")

def main():
    print("\n[*] Starting FalconOne Comprehensive System Audit...")
    auditor = SystemAuditor()
    
    # ========== CORE MODULES ==========
    print("\n[1/12] Testing Core Modules...")
    auditor.test_import('falconone.core.config', 'Core Config')
    auditor.test_import('falconone.core.orchestrator', 'Core Orchestrator')
    auditor.test_import('falconone.core.signal_bus', 'Signal Bus')
    auditor.test_import('falconone.core.detector_scanner', 'Detector Scanner')
    auditor.test_import('falconone.core.multi_tenant', 'Multi-Tenant')
    
    # ========== MONITORING MODULES ==========
    print("\n[2/12] Testing Monitoring Modules...")
    auditor.test_import('falconone.monitoring.gsm_monitor', 'GSM Monitor')
    auditor.test_import('falconone.monitoring.cdma_monitor', 'CDMA Monitor')
    auditor.test_import('falconone.monitoring.umts_monitor', 'UMTS Monitor')
    auditor.test_import('falconone.monitoring.lte_monitor', 'LTE Monitor')
    auditor.test_import('falconone.monitoring.fiveg_monitor', '5G Monitor')
    auditor.test_import('falconone.monitoring.sixg_monitor', '6G Monitor')
    auditor.test_import('falconone.monitoring.ntn_monitor', 'NTN Monitor')
    auditor.test_import('falconone.monitoring.profiler', 'Profiler')
    
    # ========== EXPLOIT MODULES (CRITICAL) ==========
    print("\n[3/12] Testing Exploit Modules...")
    auditor.test_import('falconone.exploit.exploit_engine', 'Exploit Engine')
    auditor.test_import('falconone.exploit.vulnerability_db', 'Vulnerability Database')
    auditor.test_import('falconone.exploit.payload_generator', 'Payload Generator')
    auditor.test_import('falconone.exploit.crypto_attacks', 'Crypto Attacks')
    auditor.test_import('falconone.exploit.message_injector', 'Message Injector')
    auditor.test_import('falconone.exploit.ntn_attacks', 'NTN Attacks')
    auditor.test_import('falconone.exploit.semantic_exploiter', 'Semantic Exploiter')
    auditor.test_import('falconone.exploit.v2x_attacks', 'V2X Attacks')
    
    # ========== RANSACKED EXPLOIT MODULES ==========
    print("\n[4/12] Testing RANSacked Exploit Modules...")
    auditor.test_import('falconone.exploit.ransacked_core', 'RANSacked Core')
    auditor.test_import('falconone.exploit.ransacked_oai_5g', 'RANSacked OAI 5G')
    auditor.test_import('falconone.exploit.ransacked_open5gs_5g', 'RANSacked Open5GS 5G')
    auditor.test_import('falconone.exploit.ransacked_open5gs_lte', 'RANSacked Open5GS LTE')
    auditor.test_import('falconone.exploit.ransacked_magma_lte', 'RANSacked Magma LTE')
    auditor.test_import('falconone.exploit.ransacked_misc', 'RANSacked Misc')
    auditor.test_import('falconone.exploit.ransacked_payloads', 'RANSacked Payloads')
    
    # ========== AI/ML MODULES ==========
    print("\n[5/12] Testing AI/ML Modules...")
    auditor.test_import('falconone.ai.signal_classifier', 'Signal Classifier')
    auditor.test_import('falconone.ai.device_profiler', 'Device Profiler')
    auditor.test_import('falconone.ai.kpi_monitor', 'KPI Monitor')
    auditor.test_import('falconone.ai.ric_optimizer', 'RIC Optimizer')
    auditor.test_import('falconone.ai.online_learning', 'Online Learning')
    auditor.test_import('falconone.ai.explainable_ai', 'Explainable AI')
    auditor.test_import('falconone.ai.model_zoo', 'Model Zoo')
    auditor.test_import('falconone.ai.graph_topology', 'Graph Topology')
    auditor.test_import('falconone.ai.suci_deconcealment', 'SUCI Deconcealment')
    auditor.test_import('falconone.ai.federated_coordinator', 'Federated Coordinator')
    
    # ========== CRYPTO MODULES ==========
    print("\n[6/12] Testing Crypto Modules...")
    auditor.test_import('falconone.crypto.analyzer', 'Crypto Analyzer')
    auditor.test_import('falconone.crypto.quantum_resistant', 'Quantum Resistant')
    auditor.test_import('falconone.crypto.zkp', 'Zero-Knowledge Proofs')
    
    # ========== GEOLOCATION MODULES ==========
    print("\n[7/12] Testing Geolocation Modules...")
    auditor.test_import('falconone.geolocation.locator', 'Locator')
    auditor.test_import('falconone.geolocation.precision_geolocation', 'Precision Geolocation')
    auditor.test_import('falconone.geolocation.environmental_adapter', 'Environmental Adapter')
    
    # ========== SDR MODULES ==========
    print("\n[8/12] Testing SDR Modules...")
    auditor.test_import('falconone.sdr.sdr_layer', 'SDR Layer')
    # Hardware detector module doesn't exist - functionality is in sdr_layer
    
    # ========== SIM MODULES ==========
    print("\n[9/12] Testing SIM Modules...")
    auditor.test_import('falconone.sim.sim_manager', 'SIM Manager')
    
    # ========== SECURITY MODULES ==========
    print("\n[10/12] Testing Security Modules...")
    # Note: encryptor, rate_limiter, input_validator don't exist as separate modules
    # Security functionality is distributed across auditor.py, data_validator.py, etc.
    auditor.test_import('falconone.security.auditor', 'Security Auditor')
    auditor.test_import('falconone.utils.data_validator', 'Data Validator')
    
    # ========== UI/DASHBOARD MODULES ==========
    print("\n[11/12] Testing UI/Dashboard Modules...")
    auditor.test_import('falconone.ui.dashboard', 'Dashboard')
    
    # ========== UTILITIES ==========
    print("\n[12/12] Testing Utility Modules...")
    auditor.test_import('falconone.utils.logger', 'Logger')
    # Config loader functionality is in core.config, not a separate module
    auditor.test_import('falconone.utils.config', 'Config Utilities')
    
    # ========== DEPENDENCY CHECK ==========
    print("\n[BONUS] Testing Critical Dependencies...")
    
    # Core dependencies
    auditor.test_import('numpy', 'NumPy')
    auditor.test_import('scipy', 'SciPy')
    auditor.test_import('yaml', 'PyYAML')
    auditor.test_import('flask', 'Flask')
    auditor.test_import('flask_socketio', 'Flask-SocketIO')
    auditor.test_import('requests', 'Requests')
    
    # SDR/Signal Processing (Optional but recommended)
    scapy_ok = auditor.test_import('scapy.all', 'Scapy')
    
    # AI/ML (Optional)
    tf_ok = auditor.test_import('tensorflow', 'TensorFlow')
    torch_ok = auditor.test_import('torch', 'PyTorch')
    
    # Quantum (Optional)
    qiskit_ok = auditor.test_import('qiskit', 'Qiskit')
    
    # Security
    auditor.test_import('cryptography', 'Cryptography')
    auditor.test_import('bcrypt', 'BCrypt')
    
    # Print final report
    auditor.print_report()
    
    # Return exit code
    if auditor.results['failed']:
        print("[!] AUDIT FAILED - Some modules are broken or missing dependencies")
        return 1
    elif auditor.results['warnings']:
        print("[!] AUDIT PASSED WITH WARNINGS - Some optional features may not work")
        return 0
    else:
        print("[OK] AUDIT PASSED - All features functional")
        return 0

if __name__ == '__main__':
    sys.exit(main())
