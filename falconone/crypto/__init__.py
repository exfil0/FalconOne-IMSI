"""FalconOne Cryptanalysis Package (v1.9.4)

Post-Quantum Cryptography enhancements:
- OQS (Open Quantum Safe) library integration
- Hybrid KEM: X25519+Kyber, ECDH+Kyber
- Hybrid Signatures: Ed25519+Dilithium, ECDSA+Dilithium
- Quantum attack simulation with Qiskit
"""
from .analyzer import CryptoAnalyzer
from .post_quantum import (
    # Data structures
    PQKeyPair,
    PQCiphertext,
    PQSignature,
    QuantumThreatAnalysis,
    HybridKeyPair,
    HybridCiphertext,
    HybridSignature,
    # OQS Integration
    OQSWrapper,
    OQSKEMInstance,
    OQSSigInstance,
    # Hybrid Schemes (v1.9.4)
    HybridKEMScheme,
    HybridSignatureScheme,
    # Quantum Simulation (v1.9.4)
    QuantumAttackSimulator,
    # Simulators
    KyberSimulator,
    DilithiumSimulator,
    SPHINCSPlusSimulator,
    # Manager
    PostQuantumCryptoManager,
    QuantumThreatAnalyzer,
)

__all__ = [
    'CryptoAnalyzer',
    # Data structures
    'PQKeyPair',
    'PQCiphertext',
    'PQSignature',
    'QuantumThreatAnalysis',
    'HybridKeyPair',
    'HybridCiphertext',
    'HybridSignature',
    # OQS Integration
    'OQSWrapper',
    'OQSKEMInstance',
    'OQSSigInstance',
    # Hybrid Schemes
    'HybridKEMScheme',
    'HybridSignatureScheme',
    # Quantum Simulation
    'QuantumAttackSimulator',
    # Simulators
    'KyberSimulator',
    'DilithiumSimulator',
    'SPHINCSPlusSimulator',
    # Manager
    'PostQuantumCryptoManager',
    'QuantumThreatAnalyzer',
]
