"""
FalconOne IMSI/TMSI and SMS Catcher System
Version 1.9.6 - Multi-Generation Cellular Monitoring Platform
Real-World Resilience: Error Recovery, Data Validation, Security Auditing
LE Mode: Exploit-Enhanced Interception with Warrant Compliance

v1.9.6 Fixes:
- DataValidator: Fixed undefined rejected_count, removed duplicate get_statistics
- SignalClassifier: Fixed incomplete get_anomaly_report
- RICOptimizer: Fixed gym import ordering bug in SIGINTMultiAgentEnv

v1.9.5 Additions:
- Voice Interceptor Opus Support with Speaker Diarization
- Post-Quantum Crypto Hybrid Schemes (X25519+Kyber, Ed25519+Dilithium)
- OQS (Open Quantum Safe) Library Integration
- Quantum Attack Simulation with Qiskit

TOP CONFIDENTIAL - Research and Development Use Only
"""

__version__ = "1.9.6"
__author__ = "FalconOne Research Team"

# Package-level imports for easy access
from .core.orchestrator import FalconOneOrchestrator
from .core.signal_bus import SignalBus
from .utils.config import Config
from .utils.logger import setup_logger, ModuleLogger
from .utils.error_recoverer import ErrorRecoverer, ErrorType as RecoveryErrorType
from .utils.data_validator import DataValidator, ValidationLevel
from .utils.evidence_chain import EvidenceChain, InterceptType
from .security.auditor import SecurityAuditor, ComplianceStatus
from .le.intercept_enhancer import InterceptEnhancer, ChainType

__all__ = [
    'FalconOneOrchestrator',
    'SignalBus',
    'Config',
    'setup_logger',
    'ModuleLogger',
    'ErrorRecoverer',
    'RecoveryErrorType',
    'DataValidator',
    'ValidationLevel',
    'SecurityAuditor',
    'ComplianceStatus',
    'EvidenceChain',
    'InterceptType',
    'InterceptEnhancer',
    'ChainType',
]

