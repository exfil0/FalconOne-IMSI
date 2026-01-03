"""
FalconOne Post-Quantum Cryptography Module (v1.9.2)
Simulates quantum-resistant defenses against Shor/Grover attacks

Features:
- Lattice-based encryption (CRYSTALS-Kyber/Dilithium simulation)
- Hash-based signatures (SPHINCS+ simulation)
- Code-based encryption (McEliece simulation)
- Hybrid classical/PQ schemes for transition
- Quantum attack simulation and defense analysis

References:
- NIST PQC standardization (FIPS 203, 204, 205)
- CRYSTALS-Kyber/Dilithium specifications
- SPHINCS+ specification
- 3GPP TR 33.848 (Quantum-Safe Security)
"""

import os
import hashlib
import hmac
import secrets
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import struct

# NumPy for lattice operations
import numpy as np

try:
    from ..utils.logger import ModuleLogger
except ImportError:
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PQKeyPair:
    """Post-quantum key pair"""
    algorithm: str
    security_level: int  # NIST security level 1-5
    public_key: bytes
    private_key: bytes
    public_key_size: int
    private_key_size: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PQCiphertext:
    """Post-quantum ciphertext"""
    algorithm: str
    ciphertext: bytes
    encapsulated_key: Optional[bytes] = None  # For KEM
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None  # For AEAD


@dataclass
class PQSignature:
    """Post-quantum digital signature"""
    algorithm: str
    signature: bytes
    message_hash: bytes
    signature_size: int


@dataclass
class QuantumThreatAnalysis:
    """Analysis of quantum threat to a cryptographic operation"""
    algorithm: str
    classical_security_bits: int
    quantum_security_bits: int
    vulnerable_to_shor: bool
    vulnerable_to_grover: bool
    estimated_qubits_required: int
    estimated_quantum_time: str  # "hours", "days", "years", "infeasible"
    recommendation: str


# =============================================================================
# Abstract Base Classes
# =============================================================================

class PQCAlgorithm(ABC):
    """Abstract base class for post-quantum algorithms"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def security_level(self) -> int:
        pass
    
    @abstractmethod
    def keygen(self) -> PQKeyPair:
        pass


class PQKEM(PQCAlgorithm):
    """Key Encapsulation Mechanism base class"""
    
    @abstractmethod
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Returns (ciphertext, shared_secret)"""
        pass
    
    @abstractmethod
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Returns shared_secret"""
        pass


class PQSignatureScheme(PQCAlgorithm):
    """Digital signature scheme base class"""
    
    @abstractmethod
    def sign(self, private_key: bytes, message: bytes) -> bytes:
        pass
    
    @abstractmethod
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        pass


# =============================================================================
# CRYSTALS-Kyber Simulation (Lattice-based KEM)
# =============================================================================

class KyberSimulator(PQKEM):
    """
    Simulates CRYSTALS-Kyber Key Encapsulation Mechanism
    
    Based on Module-LWE (Learning With Errors) problem.
    NIST FIPS 203 standard for key encapsulation.
    
    Security levels:
    - Kyber512: Level 1 (AES-128 equivalent)
    - Kyber768: Level 3 (AES-192 equivalent)
    - Kyber1024: Level 5 (AES-256 equivalent)
    """
    
    # Kyber parameters by security level
    PARAMS = {
        512: {'n': 256, 'k': 2, 'q': 3329, 'eta1': 3, 'eta2': 2, 'du': 10, 'dv': 4,
              'pk_size': 800, 'sk_size': 1632, 'ct_size': 768, 'ss_size': 32},
        768: {'n': 256, 'k': 3, 'q': 3329, 'eta1': 2, 'eta2': 2, 'du': 10, 'dv': 4,
              'pk_size': 1184, 'sk_size': 2400, 'ct_size': 1088, 'ss_size': 32},
        1024: {'n': 256, 'k': 4, 'q': 3329, 'eta1': 2, 'eta2': 2, 'du': 11, 'dv': 5,
               'pk_size': 1568, 'sk_size': 3168, 'ct_size': 1568, 'ss_size': 32}
    }
    
    def __init__(self, variant: int = 768, logger: logging.Logger = None):
        """
        Initialize Kyber simulator
        
        Args:
            variant: 512, 768, or 1024
            logger: Logger instance
        """
        if variant not in self.PARAMS:
            raise ValueError(f"Invalid Kyber variant: {variant}")
        
        self.variant = variant
        self.params = self.PARAMS[variant]
        self.logger = ModuleLogger('KyberSimulator', logger)
        
        self.logger.info(f"Kyber-{variant} simulator initialized",
                        security_level=self.security_level)
    
    @property
    def name(self) -> str:
        return f"Kyber-{self.variant}"
    
    @property
    def security_level(self) -> int:
        return {512: 1, 768: 3, 1024: 5}[self.variant]
    
    def keygen(self) -> PQKeyPair:
        """
        Generate Kyber key pair (simulated)
        
        In real implementation, this would:
        1. Sample random seed
        2. Expand seed to matrix A via XOF
        3. Sample secret vector s and error vector e from centered binomial
        4. Compute public key t = As + e
        5. Return (pk, sk) where pk = (ρ, t) and sk = (pk, s, H(pk), z)
        """
        n = self.params['n']
        k = self.params['k']
        q = self.params['q']
        
        # Simulate lattice-based key generation
        seed = secrets.token_bytes(32)
        
        # Generate polynomial matrix A (simulated hash expansion)
        np.random.seed(int.from_bytes(seed[:4], 'big'))
        A = np.random.randint(0, q, size=(k, k, n), dtype=np.int32)
        
        # Sample secret and error polynomials
        s = self._sample_cbd(k, n, self.params['eta1'])
        e = self._sample_cbd(k, n, self.params['eta1'])
        
        # Compute public key t = A*s + e (simplified, no NTT)
        t = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            for j in range(k):
                t[i] = (t[i] + self._poly_mul(A[i, j], s[j], q)) % q
            t[i] = (t[i] + e[i]) % q
        
        # Serialize keys
        public_key = self._serialize_pk(seed, t)
        private_key = self._serialize_sk(seed, s, t)
        
        self.logger.debug("Kyber key pair generated")
        
        return PQKeyPair(
            algorithm=self.name,
            security_level=self.security_level,
            public_key=public_key,
            private_key=private_key,
            public_key_size=len(public_key),
            private_key_size=len(private_key)
        )
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulate shared secret (simulated)
        
        Returns:
            (ciphertext, shared_secret)
        """
        n = self.params['n']
        k = self.params['k']
        q = self.params['q']
        
        # Parse public key
        seed, t = self._deserialize_pk(public_key)
        
        # Generate random message
        m = secrets.token_bytes(32)
        
        # Expand seed to matrix A
        np.random.seed(int.from_bytes(seed[:4], 'big'))
        A = np.random.randint(0, q, size=(k, k, n), dtype=np.int32)
        
        # Sample ephemeral secret and errors
        r = self._sample_cbd(k, n, self.params['eta1'])
        e1 = self._sample_cbd(k, n, self.params['eta2'])
        e2 = self._sample_cbd(1, n, self.params['eta2'])[0]
        
        # Compute u = A^T * r + e1
        u = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            for j in range(k):
                u[i] = (u[i] + self._poly_mul(A[j, i], r[j], q)) % q
            u[i] = (u[i] + e1[i]) % q
        
        # Compute v = t^T * r + e2 + encode(m)
        v = e2.copy()
        for i in range(k):
            v = (v + self._poly_mul(t[i], r[i], q)) % q
        
        # Encode message into polynomial
        m_poly = self._encode_message(m, n, q)
        v = (v + m_poly) % q
        
        # Compress and serialize ciphertext
        ciphertext = self._serialize_ct(u, v)
        
        # Derive shared secret
        shared_secret = hashlib.sha3_256(m + ciphertext).digest()
        
        self.logger.debug("Kyber encapsulation completed")
        
        return ciphertext, shared_secret
    
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """
        Decapsulate shared secret (simulated)
        
        Returns:
            shared_secret
        """
        n = self.params['n']
        k = self.params['k']
        q = self.params['q']
        
        # Parse private key and ciphertext
        seed, s, t = self._deserialize_sk(private_key)
        u, v = self._deserialize_ct(ciphertext)
        
        # Compute m' = v - s^T * u
        m_prime = v.copy()
        for i in range(k):
            m_prime = (m_prime - self._poly_mul(s[i], u[i], q)) % q
        
        # Decode message
        m = self._decode_message(m_prime, n, q)
        
        # Derive shared secret (simplified - no re-encryption check)
        shared_secret = hashlib.sha3_256(m + ciphertext).digest()
        
        self.logger.debug("Kyber decapsulation completed")
        
        return shared_secret
    
    def _sample_cbd(self, k: int, n: int, eta: int) -> np.ndarray:
        """Sample from centered binomial distribution"""
        result = np.zeros((k, n), dtype=np.int32)
        for i in range(k):
            for j in range(n):
                a = sum(secrets.randbelow(2) for _ in range(eta))
                b = sum(secrets.randbelow(2) for _ in range(eta))
                result[i, j] = a - b
        return result
    
    def _poly_mul(self, a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
        """Polynomial multiplication mod x^n + 1 (simplified)"""
        n = len(a)
        c = np.zeros(n, dtype=np.int32)
        for i in range(n):
            for j in range(n):
                idx = (i + j) % n
                sign = 1 if (i + j) < n else -1
                c[idx] = (c[idx] + sign * int(a[i]) * int(b[j])) % q
        return c
    
    def _encode_message(self, m: bytes, n: int, q: int) -> np.ndarray:
        """Encode message bits into polynomial"""
        poly = np.zeros(n, dtype=np.int32)
        bits = ''.join(format(byte, '08b') for byte in m)
        for i in range(min(len(bits), n)):
            poly[i] = (q // 2) if bits[i] == '1' else 0
        return poly
    
    def _decode_message(self, poly: np.ndarray, n: int, q: int) -> bytes:
        """Decode polynomial to message"""
        threshold = q // 4
        bits = ''
        for i in range(min(256, n)):  # 32 bytes = 256 bits
            val = poly[i]
            if val > q // 2:
                val = val - q
            bits += '1' if abs(val) > threshold else '0'
        
        # Convert bits to bytes
        m = bytes(int(bits[i:i+8], 2) for i in range(0, 256, 8))
        return m
    
    def _serialize_pk(self, seed: bytes, t: np.ndarray) -> bytes:
        """Serialize public key"""
        return seed + t.tobytes()
    
    def _deserialize_pk(self, pk: bytes) -> Tuple[bytes, np.ndarray]:
        """Deserialize public key"""
        seed = pk[:32]
        t = np.frombuffer(pk[32:], dtype=np.int32).reshape((self.params['k'], self.params['n']))
        return seed, t
    
    def _serialize_sk(self, seed: bytes, s: np.ndarray, t: np.ndarray) -> bytes:
        """Serialize private key"""
        return seed + s.tobytes() + t.tobytes()
    
    def _deserialize_sk(self, sk: bytes) -> Tuple[bytes, np.ndarray, np.ndarray]:
        """Deserialize private key"""
        k, n = self.params['k'], self.params['n']
        seed = sk[:32]
        s_size = k * n * 4
        s = np.frombuffer(sk[32:32+s_size], dtype=np.int32).reshape((k, n))
        t = np.frombuffer(sk[32+s_size:], dtype=np.int32).reshape((k, n))
        return seed, s, t
    
    def _serialize_ct(self, u: np.ndarray, v: np.ndarray) -> bytes:
        """Serialize ciphertext"""
        return u.tobytes() + v.tobytes()
    
    def _deserialize_ct(self, ct: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """Deserialize ciphertext"""
        k, n = self.params['k'], self.params['n']
        u_size = k * n * 4
        u = np.frombuffer(ct[:u_size], dtype=np.int32).reshape((k, n))
        v = np.frombuffer(ct[u_size:], dtype=np.int32).reshape((n,))
        return u, v


# =============================================================================
# CRYSTALS-Dilithium Simulation (Lattice-based Signatures)
# =============================================================================

class DilithiumSimulator(PQSignatureScheme):
    """
    Simulates CRYSTALS-Dilithium Digital Signature Algorithm
    
    Based on Module-LWE and Module-SIS problems.
    NIST FIPS 204 standard for digital signatures.
    
    Security levels:
    - Dilithium2: Level 2 (~128-bit)
    - Dilithium3: Level 3 (~192-bit)
    - Dilithium5: Level 5 (~256-bit)
    """
    
    PARAMS = {
        2: {'n': 256, 'k': 4, 'l': 4, 'q': 8380417, 'eta': 2, 'tau': 39,
            'pk_size': 1312, 'sk_size': 2528, 'sig_size': 2420},
        3: {'n': 256, 'k': 6, 'l': 5, 'q': 8380417, 'eta': 4, 'tau': 49,
            'pk_size': 1952, 'sk_size': 4000, 'sig_size': 3293},
        5: {'n': 256, 'k': 8, 'l': 7, 'q': 8380417, 'eta': 2, 'tau': 60,
            'pk_size': 2592, 'sk_size': 4864, 'sig_size': 4595}
    }
    
    def __init__(self, variant: int = 3, logger: logging.Logger = None):
        if variant not in self.PARAMS:
            raise ValueError(f"Invalid Dilithium variant: {variant}")
        
        self.variant = variant
        self.params = self.PARAMS[variant]
        self.logger = ModuleLogger('DilithiumSimulator', logger)
        
        self.logger.info(f"Dilithium-{variant} simulator initialized")
    
    @property
    def name(self) -> str:
        return f"Dilithium-{self.variant}"
    
    @property
    def security_level(self) -> int:
        return self.variant
    
    def keygen(self) -> PQKeyPair:
        """Generate Dilithium key pair (simplified simulation)"""
        seed = secrets.token_bytes(32)
        
        # Simulate key sizes
        public_key = secrets.token_bytes(self.params['pk_size'])
        private_key = seed + secrets.token_bytes(self.params['sk_size'] - 32)
        
        return PQKeyPair(
            algorithm=self.name,
            security_level=self.security_level,
            public_key=public_key,
            private_key=private_key,
            public_key_size=len(public_key),
            private_key_size=len(private_key)
        )
    
    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign message (simplified simulation)"""
        # Hash message
        msg_hash = hashlib.sha3_512(message).digest()
        
        # Simulate signature generation
        signature = hashlib.sha3_256(private_key + msg_hash).digest()
        signature += secrets.token_bytes(self.params['sig_size'] - 32)
        
        return signature
    
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify signature (simplified simulation)"""
        # In real implementation, this would verify the lattice-based proof
        msg_hash = hashlib.sha3_512(message).digest()
        
        # Simulate verification (always succeeds for valid format)
        return len(signature) == self.params['sig_size']


# =============================================================================
# SPHINCS+ Simulation (Hash-based Signatures)
# =============================================================================

class SPHINCSPlusSimulator(PQSignatureScheme):
    """
    Simulates SPHINCS+ Stateless Hash-Based Signature Scheme
    
    Based on hash function security (not lattice assumptions).
    NIST FIPS 205 standard for digital signatures.
    
    Variants:
    - SPHINCS+-128s/f: Level 1 (128-bit security, s=small, f=fast)
    - SPHINCS+-192s/f: Level 3
    - SPHINCS+-256s/f: Level 5
    """
    
    PARAMS = {
        '128s': {'n': 16, 'h': 63, 'pk_size': 32, 'sk_size': 64, 'sig_size': 7856},
        '128f': {'n': 16, 'h': 66, 'pk_size': 32, 'sk_size': 64, 'sig_size': 17088},
        '192s': {'n': 24, 'h': 63, 'pk_size': 48, 'sk_size': 96, 'sig_size': 16224},
        '192f': {'n': 24, 'h': 66, 'pk_size': 48, 'sk_size': 96, 'sig_size': 35664},
        '256s': {'n': 32, 'h': 64, 'pk_size': 64, 'sk_size': 128, 'sig_size': 29792},
        '256f': {'n': 32, 'h': 68, 'pk_size': 64, 'sk_size': 128, 'sig_size': 49856}
    }
    
    def __init__(self, variant: str = '256s', logger: logging.Logger = None):
        if variant not in self.PARAMS:
            raise ValueError(f"Invalid SPHINCS+ variant: {variant}")
        
        self.variant = variant
        self.params = self.PARAMS[variant]
        self.logger = ModuleLogger('SPHINCSPlusSimulator', logger)
        
        self.logger.info(f"SPHINCS+-{variant} simulator initialized")
    
    @property
    def name(self) -> str:
        return f"SPHINCS+-{self.variant}"
    
    @property
    def security_level(self) -> int:
        return {'128s': 1, '128f': 1, '192s': 3, '192f': 3, '256s': 5, '256f': 5}[self.variant]
    
    def keygen(self) -> PQKeyPair:
        """Generate SPHINCS+ key pair"""
        seed = secrets.token_bytes(self.params['n'] * 3)
        
        public_key = hashlib.sha3_256(seed).digest()[:self.params['pk_size']]
        private_key = seed[:self.params['sk_size']]
        
        return PQKeyPair(
            algorithm=self.name,
            security_level=self.security_level,
            public_key=public_key,
            private_key=private_key,
            public_key_size=len(public_key),
            private_key_size=len(private_key)
        )
    
    def sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign message with hash-based signature"""
        randomizer = secrets.token_bytes(self.params['n'])
        msg_hash = hashlib.sha3_512(randomizer + message).digest()
        
        # Simulate FORS + hypertree signature
        signature = randomizer + secrets.token_bytes(self.params['sig_size'] - self.params['n'])
        
        return signature
    
    def verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify hash-based signature"""
        return len(signature) == self.params['sig_size']


# =============================================================================
# Quantum Threat Analyzer
# =============================================================================

class QuantumThreatAnalyzer:
    """
    Analyze cryptographic operations for quantum vulnerability
    
    Provides:
    - Shor's algorithm impact assessment (RSA, ECC, DH)
    - Grover's algorithm impact (symmetric ciphers, hashes)
    - Migration recommendations to PQC
    """
    
    # Classical algorithms vulnerable to quantum attacks
    SHOR_VULNERABLE = {
        'RSA': {'classical_bits': 2048, 'quantum_bits': 0, 'qubits': 4096},
        'RSA-4096': {'classical_bits': 4096, 'quantum_bits': 0, 'qubits': 8192},
        'ECDSA-P256': {'classical_bits': 128, 'quantum_bits': 0, 'qubits': 2330},
        'ECDSA-P384': {'classical_bits': 192, 'quantum_bits': 0, 'qubits': 3484},
        'ECDH': {'classical_bits': 128, 'quantum_bits': 0, 'qubits': 2330},
        'DH-2048': {'classical_bits': 112, 'quantum_bits': 0, 'qubits': 4096},
    }
    
    # Symmetric algorithms (Grover halves effective key length)
    GROVER_AFFECTED = {
        'AES-128': {'classical_bits': 128, 'quantum_bits': 64},
        'AES-192': {'classical_bits': 192, 'quantum_bits': 96},
        'AES-256': {'classical_bits': 256, 'quantum_bits': 128},
        'ChaCha20': {'classical_bits': 256, 'quantum_bits': 128},
        'SHA-256': {'classical_bits': 256, 'quantum_bits': 128},
        'SHA-384': {'classical_bits': 384, 'quantum_bits': 192},
        'SHA-512': {'classical_bits': 512, 'quantum_bits': 256},
    }
    
    # Post-quantum algorithms (resistant)
    PQ_RESISTANT = {
        'Kyber-512': {'classical_bits': 128, 'quantum_bits': 128},
        'Kyber-768': {'classical_bits': 192, 'quantum_bits': 192},
        'Kyber-1024': {'classical_bits': 256, 'quantum_bits': 256},
        'Dilithium-2': {'classical_bits': 128, 'quantum_bits': 128},
        'Dilithium-3': {'classical_bits': 192, 'quantum_bits': 192},
        'Dilithium-5': {'classical_bits': 256, 'quantum_bits': 256},
        'SPHINCS+-256s': {'classical_bits': 256, 'quantum_bits': 256},
    }
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = ModuleLogger('QuantumThreatAnalyzer', logger)
        self.logger.info("Quantum threat analyzer initialized")
    
    def analyze_algorithm(self, algorithm: str) -> QuantumThreatAnalysis:
        """
        Analyze quantum threat for a specific algorithm
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            QuantumThreatAnalysis with vulnerability assessment
        """
        if algorithm in self.SHOR_VULNERABLE:
            info = self.SHOR_VULNERABLE[algorithm]
            return QuantumThreatAnalysis(
                algorithm=algorithm,
                classical_security_bits=info['classical_bits'],
                quantum_security_bits=0,
                vulnerable_to_shor=True,
                vulnerable_to_grover=False,
                estimated_qubits_required=info['qubits'],
                estimated_quantum_time="hours",
                recommendation=f"Migrate to {self._get_pq_recommendation(algorithm)}"
            )
        
        elif algorithm in self.GROVER_AFFECTED:
            info = self.GROVER_AFFECTED[algorithm]
            return QuantumThreatAnalysis(
                algorithm=algorithm,
                classical_security_bits=info['classical_bits'],
                quantum_security_bits=info['quantum_bits'],
                vulnerable_to_shor=False,
                vulnerable_to_grover=True,
                estimated_qubits_required=info['quantum_bits'],
                estimated_quantum_time="years" if info['quantum_bits'] >= 128 else "days",
                recommendation="Double key size for equivalent post-quantum security"
            )
        
        elif algorithm in self.PQ_RESISTANT:
            info = self.PQ_RESISTANT[algorithm]
            return QuantumThreatAnalysis(
                algorithm=algorithm,
                classical_security_bits=info['classical_bits'],
                quantum_security_bits=info['quantum_bits'],
                vulnerable_to_shor=False,
                vulnerable_to_grover=False,
                estimated_qubits_required=0,
                estimated_quantum_time="infeasible",
                recommendation="Already quantum-resistant"
            )
        
        else:
            return QuantumThreatAnalysis(
                algorithm=algorithm,
                classical_security_bits=0,
                quantum_security_bits=0,
                vulnerable_to_shor=False,
                vulnerable_to_grover=False,
                estimated_qubits_required=0,
                estimated_quantum_time="unknown",
                recommendation="Unknown algorithm - manual analysis required"
            )
    
    def _get_pq_recommendation(self, algorithm: str) -> str:
        """Get PQ replacement recommendation"""
        if 'RSA' in algorithm or 'DH' in algorithm:
            return "Kyber-768 (KEM) + Dilithium-3 (signatures)"
        elif 'ECDSA' in algorithm or 'ECDH' in algorithm:
            return "Kyber-768 (key exchange) + Dilithium-3 (signatures)"
        return "Kyber-768 or SPHINCS+-256s"
    
    def create_hybrid_scheme(self, 
                            classical_kem: str,
                            pq_kem: PQKEM) -> Dict[str, Any]:
        """
        Create hybrid classical/PQ key encapsulation
        
        Combines:
        - Classical ECDH (X25519) for backwards compatibility
        - Post-quantum KEM (Kyber) for quantum resistance
        
        Shared secret = KDF(classical_ss || pq_ss)
        """
        # Generate PQ key pair
        pq_keypair = pq_kem.keygen()
        
        return {
            'scheme': f"Hybrid-{classical_kem}-{pq_kem.name}",
            'classical_algorithm': classical_kem,
            'pq_algorithm': pq_kem.name,
            'pq_public_key': pq_keypair.public_key,
            'pq_private_key': pq_keypair.private_key,
            'security_level': f"Classical: {classical_kem}, PQ: {pq_kem.security_level}",
            'total_pk_size': pq_keypair.public_key_size + 32,  # + X25519
            'total_ct_size': pq_kem.params['ct_size'] + 32
        }
    
    def simulate_shor_attack(self, algorithm: str, key_bits: int) -> Dict[str, Any]:
        """
        Simulate Shor's algorithm attack on public key crypto
        
        Args:
            algorithm: Target algorithm (RSA, ECDSA, etc.)
            key_bits: Key size in bits
            
        Returns:
            Attack simulation results
        """
        if algorithm.startswith('RSA'):
            # Shor requires ~2n qubits for n-bit RSA
            qubits_needed = 2 * key_bits
            gate_count = key_bits ** 3  # O(n^3) gates
            
        elif algorithm.startswith('ECC') or algorithm.startswith('ECDSA'):
            # Shor for ECC: ~6n qubits for n-bit curve
            qubits_needed = 6 * key_bits // 2
            gate_count = (key_bits // 2) ** 3
            
        else:
            return {'error': f'Unknown algorithm: {algorithm}'}
        
        # Current quantum computer limitations
        current_qubits = 1000  # Approximate 2024 state
        
        return {
            'algorithm': algorithm,
            'key_bits': key_bits,
            'qubits_required': qubits_needed,
            'gate_count': gate_count,
            'current_capability': current_qubits,
            'cryptographically_relevant': qubits_needed <= current_qubits,
            'estimated_timeline': self._estimate_quantum_timeline(qubits_needed),
            'recommendation': 'Migrate to PQC now' if qubits_needed < 10000 else 'Plan migration'
        }
    
    def _estimate_quantum_timeline(self, qubits_needed: int) -> str:
        """Estimate when quantum computers could break the algorithm"""
        if qubits_needed <= 100:
            return "Already possible (with error correction challenges)"
        elif qubits_needed <= 1000:
            return "2025-2030"
        elif qubits_needed <= 5000:
            return "2030-2035"
        elif qubits_needed <= 10000:
            return "2035-2040"
        else:
            return "2040+"


# =============================================================================
# Main PQC Manager
# =============================================================================

class PostQuantumCryptoManager:
    """
    Central manager for post-quantum cryptography operations
    
    Provides:
    - Algorithm selection based on security requirements
    - Key management for PQ algorithms
    - Hybrid scheme support
    - Threat analysis and recommendations
    """
    
    def __init__(self, default_security_level: int = 3, 
                 logger: logging.Logger = None):
        """
        Initialize PQC manager
        
        Args:
            default_security_level: NIST security level (1, 2, 3, or 5)
            logger: Logger instance
        """
        self.security_level = default_security_level
        self.logger = ModuleLogger('PostQuantumCryptoManager', logger)
        
        # Initialize algorithm simulators
        self.kyber = KyberSimulator(
            {1: 512, 2: 512, 3: 768, 5: 1024}[default_security_level],
            logger
        )
        self.dilithium = DilithiumSimulator(
            {1: 2, 2: 2, 3: 3, 5: 5}[default_security_level],
            logger
        )
        self.sphincs = SPHINCSPlusSimulator(
            {1: '128s', 2: '128s', 3: '192s', 5: '256s'}[default_security_level],
            logger
        )
        
        self.threat_analyzer = QuantumThreatAnalyzer(logger)
        
        # Key storage
        self.key_pairs: Dict[str, PQKeyPair] = {}
        
        self.logger.info("Post-quantum crypto manager initialized",
                        security_level=default_security_level)
    
    def generate_kem_keypair(self, key_id: str = None) -> PQKeyPair:
        """Generate and store Kyber KEM key pair"""
        keypair = self.kyber.keygen()
        
        if key_id:
            self.key_pairs[key_id] = keypair
        
        return keypair
    
    def generate_signature_keypair(self, algorithm: str = 'dilithium',
                                   key_id: str = None) -> PQKeyPair:
        """Generate and store signature key pair"""
        if algorithm == 'dilithium':
            keypair = self.dilithium.keygen()
        elif algorithm == 'sphincs':
            keypair = self.sphincs.keygen()
        else:
            raise ValueError(f"Unknown signature algorithm: {algorithm}")
        
        if key_id:
            self.key_pairs[key_id] = keypair
        
        return keypair
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate shared secret using Kyber"""
        return self.kyber.encapsulate(public_key)
    
    def decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate shared secret"""
        return self.kyber.decapsulate(private_key, ciphertext)
    
    def sign(self, private_key: bytes, message: bytes, 
             algorithm: str = 'dilithium') -> bytes:
        """Sign message with PQ signature"""
        if algorithm == 'dilithium':
            return self.dilithium.sign(private_key, message)
        elif algorithm == 'sphincs':
            return self.sphincs.sign(private_key, message)
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def verify(self, public_key: bytes, message: bytes, signature: bytes,
               algorithm: str = 'dilithium') -> bool:
        """Verify PQ signature"""
        if algorithm == 'dilithium':
            return self.dilithium.verify(public_key, message, signature)
        elif algorithm == 'sphincs':
            return self.sphincs.verify(public_key, message, signature)
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def analyze_threat(self, algorithm: str) -> QuantumThreatAnalysis:
        """Analyze quantum threat for algorithm"""
        return self.threat_analyzer.analyze_algorithm(algorithm)
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about configured algorithms"""
        return {
            'kem': {
                'algorithm': self.kyber.name,
                'security_level': self.kyber.security_level,
                'public_key_size': self.kyber.params['pk_size'],
                'ciphertext_size': self.kyber.params['ct_size'],
                'shared_secret_size': self.kyber.params['ss_size']
            },
            'signature_dilithium': {
                'algorithm': self.dilithium.name,
                'security_level': self.dilithium.security_level,
                'public_key_size': self.dilithium.params['pk_size'],
                'signature_size': self.dilithium.params['sig_size']
            },
            'signature_sphincs': {
                'algorithm': self.sphincs.name,
                'security_level': self.sphincs.security_level,
                'public_key_size': self.sphincs.params['pk_size'],
                'signature_size': self.sphincs.params['sig_size']
            }
        }
