"""
FalconOne Post-Quantum Cryptography Module (v1.9.4)
Simulates quantum-resistant defenses against Shor/Grover attacks

Features:
- Lattice-based encryption (CRYSTALS-Kyber/Dilithium simulation)
- Hash-based signatures (SPHINCS+ simulation)
- Code-based encryption (McEliece simulation)
- Hybrid classical/PQ schemes for transition
- Quantum attack simulation and defense analysis
- OQS (Open Quantum Safe) library integration (v1.9.4)
- Hybrid Key Exchange: X25519+Kyber, ECDH+Kyber (v1.9.4)
- Hybrid Signatures: ECDSA+Dilithium, Ed25519+Dilithium (v1.9.4)

References:
- NIST PQC standardization (FIPS 203, 204, 205)
- CRYSTALS-Kyber/Dilithium specifications
- SPHINCS+ specification
- 3GPP TR 33.848 (Quantum-Safe Security)
- IETF draft-ietf-tls-hybrid-design (Hybrid Key Exchange)
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

# OQS (Open Quantum Safe) library integration
try:
    import oqs
    OQS_AVAILABLE = True
except ImportError:
    OQS_AVAILABLE = False

# Classical cryptography for hybrids
try:
    from cryptography.hazmat.primitives.asymmetric import x25519, ec
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False

# Qiskit for quantum simulation
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import Aer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

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


@dataclass
class HybridKeyPair:
    """Hybrid classical + post-quantum key pair (v1.9.4)"""
    classical_algorithm: str
    pq_algorithm: str
    classical_public_key: bytes
    classical_private_key: bytes
    pq_public_key: bytes
    pq_private_key: bytes
    combined_public_key: bytes  # Concatenated for transport
    security_level: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HybridCiphertext:
    """Hybrid KEM ciphertext (v1.9.4)"""
    classical_ciphertext: bytes
    pq_ciphertext: bytes
    combined_ciphertext: bytes  # For transport
    kdf_info: bytes  # Context for key derivation


@dataclass
class HybridSignature:
    """Hybrid classical + PQ signature (v1.9.4)"""
    classical_algorithm: str
    pq_algorithm: str
    classical_signature: bytes
    pq_signature: bytes
    combined_signature: bytes  # Concatenated
    message_hash: bytes


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
# OQS Library Integration (v1.9.4)
# =============================================================================

class OQSWrapper:
    """
    Open Quantum Safe (liboqs) library wrapper
    
    Provides native implementations when available, falls back to simulators.
    Supports: Kyber, Dilithium, SPHINCS+, Falcon, NTRU
    
    Reference: https://openquantumsafe.org/
    """
    
    # Supported algorithms with OQS naming
    SUPPORTED_KEMS = {
        'kyber512': 'Kyber512',
        'kyber768': 'Kyber768',
        'kyber1024': 'Kyber1024',
        'ntru_hps2048509': 'NTRU-HPS-2048-509',
        'ntru_hps2048677': 'NTRU-HPS-2048-677',
        'ntru_hps4096821': 'NTRU-HPS-4096-821',
        'bikel1': 'BIKE-L1',
        'bikel3': 'BIKE-L3',
        'hqc128': 'HQC-128',
        'hqc192': 'HQC-192',
        'hqc256': 'HQC-256',
    }
    
    SUPPORTED_SIGS = {
        'dilithium2': 'Dilithium2',
        'dilithium3': 'Dilithium3',
        'dilithium5': 'Dilithium5',
        'falcon512': 'Falcon-512',
        'falcon1024': 'Falcon-1024',
        'sphincs_sha2_128f': 'SPHINCS+-SHA2-128f-simple',
        'sphincs_sha2_192f': 'SPHINCS+-SHA2-192f-simple',
        'sphincs_sha2_256f': 'SPHINCS+-SHA2-256f-simple',
        'sphincs_shake_128f': 'SPHINCS+-SHAKE-128f-simple',
        'sphincs_shake_192f': 'SPHINCS+-SHAKE-192f-simple',
        'sphincs_shake_256f': 'SPHINCS+-SHAKE-256f-simple',
    }
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = ModuleLogger('OQSWrapper', logger)
        self._available_kems: List[str] = []
        self._available_sigs: List[str] = []
        self._check_availability()
    
    def _check_availability(self):
        """Check which OQS algorithms are available"""
        if not OQS_AVAILABLE:
            self.logger.warning("OQS library not available - using simulators")
            return
        
        try:
            self._available_kems = oqs.get_enabled_kem_mechanisms()
            self._available_sigs = oqs.get_enabled_sig_mechanisms()
            self.logger.info(f"OQS available: {len(self._available_kems)} KEMs, "
                           f"{len(self._available_sigs)} signatures")
        except Exception as e:
            self.logger.error(f"Failed to query OQS: {e}")
    
    @property
    def is_available(self) -> bool:
        return OQS_AVAILABLE and (self._available_kems or self._available_sigs)
    
    def create_kem(self, algorithm: str) -> Optional['OQSKEMInstance']:
        """Create a KEM instance"""
        if not OQS_AVAILABLE:
            return None
        
        oqs_name = self.SUPPORTED_KEMS.get(algorithm.lower())
        if not oqs_name or oqs_name not in self._available_kems:
            self.logger.warning(f"KEM {algorithm} not available in OQS")
            return None
        
        try:
            return OQSKEMInstance(oqs_name, self.logger.logger)
        except Exception as e:
            self.logger.error(f"Failed to create KEM {algorithm}: {e}")
            return None
    
    def create_sig(self, algorithm: str) -> Optional['OQSSigInstance']:
        """Create a signature instance"""
        if not OQS_AVAILABLE:
            return None
        
        oqs_name = self.SUPPORTED_SIGS.get(algorithm.lower())
        if not oqs_name or oqs_name not in self._available_sigs:
            self.logger.warning(f"Signature {algorithm} not available in OQS")
            return None
        
        try:
            return OQSSigInstance(oqs_name, self.logger.logger)
        except Exception as e:
            self.logger.error(f"Failed to create sig {algorithm}: {e}")
            return None
    
    def list_available_kems(self) -> List[str]:
        """List available KEM algorithms"""
        return list(self._available_kems)
    
    def list_available_sigs(self) -> List[str]:
        """List available signature algorithms"""
        return list(self._available_sigs)


class OQSKEMInstance:
    """OQS KEM instance wrapper"""
    
    def __init__(self, algorithm: str, logger: logging.Logger = None):
        self.algorithm = algorithm
        self.logger = ModuleLogger('OQSKEM', logger)
        self._kem = oqs.KeyEncapsulation(algorithm) if OQS_AVAILABLE else None
        self._public_key: Optional[bytes] = None
        self._secret_key: Optional[bytes] = None
    
    def keygen(self) -> Tuple[bytes, bytes]:
        """Generate key pair - returns (public_key, secret_key)"""
        if not self._kem:
            raise RuntimeError("OQS not available")
        
        self._public_key = self._kem.generate_keypair()
        self._secret_key = self._kem.export_secret_key()
        return self._public_key, self._secret_key
    
    def encapsulate(self, public_key: bytes = None) -> Tuple[bytes, bytes]:
        """Encapsulate - returns (ciphertext, shared_secret)"""
        if not self._kem:
            raise RuntimeError("OQS not available")
        
        pk = public_key or self._public_key
        if not pk:
            raise ValueError("No public key provided")
        
        ciphertext, shared_secret = self._kem.encap_secret(pk)
        return ciphertext, shared_secret
    
    def decapsulate(self, ciphertext: bytes, secret_key: bytes = None) -> bytes:
        """Decapsulate - returns shared_secret"""
        if not self._kem:
            raise RuntimeError("OQS not available")
        
        # Need to reconstruct with secret key
        if secret_key:
            self._kem = oqs.KeyEncapsulation(self.algorithm, secret_key)
        
        return self._kem.decap_secret(ciphertext)
    
    @property
    def details(self) -> Dict[str, Any]:
        """Get algorithm details"""
        if not self._kem:
            return {}
        return {
            'name': self.algorithm,
            'claimed_security': self._kem.details.get('claimed_nist_level', 0),
            'public_key_length': self._kem.details.get('length_public_key', 0),
            'secret_key_length': self._kem.details.get('length_secret_key', 0),
            'ciphertext_length': self._kem.details.get('length_ciphertext', 0),
            'shared_secret_length': self._kem.details.get('length_shared_secret', 0),
        }


class OQSSigInstance:
    """OQS Signature instance wrapper"""
    
    def __init__(self, algorithm: str, logger: logging.Logger = None):
        self.algorithm = algorithm
        self.logger = ModuleLogger('OQSSig', logger)
        self._sig = oqs.Signature(algorithm) if OQS_AVAILABLE else None
        self._public_key: Optional[bytes] = None
        self._secret_key: Optional[bytes] = None
    
    def keygen(self) -> Tuple[bytes, bytes]:
        """Generate key pair - returns (public_key, secret_key)"""
        if not self._sig:
            raise RuntimeError("OQS not available")
        
        self._public_key = self._sig.generate_keypair()
        self._secret_key = self._sig.export_secret_key()
        return self._public_key, self._secret_key
    
    def sign(self, message: bytes, secret_key: bytes = None) -> bytes:
        """Sign message"""
        if not self._sig:
            raise RuntimeError("OQS not available")
        
        if secret_key:
            self._sig = oqs.Signature(self.algorithm, secret_key)
        
        return self._sig.sign(message)
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes = None) -> bool:
        """Verify signature"""
        if not self._sig:
            raise RuntimeError("OQS not available")
        
        pk = public_key or self._public_key
        if not pk:
            raise ValueError("No public key provided")
        
        return self._sig.verify(message, signature, pk)
    
    @property
    def details(self) -> Dict[str, Any]:
        """Get algorithm details"""
        if not self._sig:
            return {}
        return {
            'name': self.algorithm,
            'claimed_security': self._sig.details.get('claimed_nist_level', 0),
            'public_key_length': self._sig.details.get('length_public_key', 0),
            'secret_key_length': self._sig.details.get('length_secret_key', 0),
            'signature_length': self._sig.details.get('length_signature', 0),
        }


# =============================================================================
# Hybrid Classical + Post-Quantum Schemes (v1.9.4)
# IETF draft-ietf-tls-hybrid-design compliant
# =============================================================================

class HybridKEMScheme:
    """
    Hybrid Key Encapsulation combining classical and post-quantum KEMs
    
    Implements IETF draft-ietf-tls-hybrid-design for TLS 1.3 hybrid key exchange.
    Uses HKDF to derive the final shared secret from both components.
    
    Supported combinations:
    - X25519 + Kyber768 (default, recommended)
    - X25519 + Kyber1024 (higher security)
    - ECDH-P256 + Kyber768
    - ECDH-P384 + Kyber1024
    
    Security: The combined scheme is secure if EITHER component is secure
    (defense in depth against quantum and classical attacks).
    """
    
    CLASSICAL_ALGORITHMS = {
        'x25519': ('X25519', 128),
        'ecdh_p256': ('ECDH-P256', 128),
        'ecdh_p384': ('ECDH-P384', 192),
        'ecdh_p521': ('ECDH-P521', 256),
    }
    
    PQ_ALGORITHMS = {
        'kyber512': (512, 128),
        'kyber768': (768, 192),
        'kyber1024': (1024, 256),
    }
    
    def __init__(self, 
                 classical: str = 'x25519',
                 pq: str = 'kyber768',
                 oqs_wrapper: OQSWrapper = None,
                 logger: logging.Logger = None):
        """
        Initialize hybrid KEM
        
        Args:
            classical: Classical algorithm ('x25519', 'ecdh_p256', 'ecdh_p384', 'ecdh_p521')
            pq: Post-quantum algorithm ('kyber512', 'kyber768', 'kyber1024')
            oqs_wrapper: OQS wrapper instance for native crypto
            logger: Logger instance
        """
        if classical not in self.CLASSICAL_ALGORITHMS:
            raise ValueError(f"Unsupported classical algorithm: {classical}")
        if pq not in self.PQ_ALGORITHMS:
            raise ValueError(f"Unsupported PQ algorithm: {pq}")
        
        self.classical = classical
        self.pq = pq
        self.oqs = oqs_wrapper
        self.logger = ModuleLogger('HybridKEM', logger)
        
        self._classical_private: Optional[Any] = None
        self._classical_public: Optional[bytes] = None
        self._pq_public: Optional[bytes] = None
        self._pq_private: Optional[bytes] = None
        self._kyber_sim: Optional['KyberSimulator'] = None
        
        # Calculate combined security level
        _, classical_bits = self.CLASSICAL_ALGORITHMS[classical]
        _, pq_bits = self.PQ_ALGORITHMS[pq]
        self.security_bits = max(classical_bits, pq_bits)  # Hybrid: max security
        
        self.logger.info(f"Initialized {classical}+{pq} hybrid KEM",
                        security_bits=self.security_bits)
    
    def keygen(self) -> HybridKeyPair:
        """Generate hybrid key pair"""
        # Generate classical key pair
        if self.classical == 'x25519':
            if not X25519_AVAILABLE:
                raise RuntimeError("X25519 not available")
            self._classical_private = X25519PrivateKey.generate()
            self._classical_public = self._classical_private.public_key().public_bytes_raw()
            classical_private_bytes = self._classical_private.private_bytes_raw()
        else:
            # ECDH with NIST curves
            if not ECDH_AVAILABLE:
                raise RuntimeError("ECDH not available")
            curve_map = {
                'ecdh_p256': ec.SECP256R1(),
                'ecdh_p384': ec.SECP384R1(),
                'ecdh_p521': ec.SECP521R1(),
            }
            curve = curve_map[self.classical]
            self._classical_private = ec.generate_private_key(curve)
            self._classical_public = self._classical_private.public_key().public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint
            )
            classical_private_bytes = self._classical_private.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            )
        
        # Generate PQ key pair (try OQS first, fall back to simulator)
        pq_variant = self.PQ_ALGORITHMS[self.pq][0]
        
        if self.oqs and self.oqs.is_available:
            oqs_kem = self.oqs.create_kem(self.pq)
            if oqs_kem:
                self._pq_public, self._pq_private = oqs_kem.keygen()
            else:
                self._use_kyber_simulator(pq_variant)
        else:
            self._use_kyber_simulator(pq_variant)
        
        # Combine public keys for transport
        combined_public = self._combine_keys(self._classical_public, self._pq_public)
        
        return HybridKeyPair(
            classical_algorithm=self.classical,
            pq_algorithm=self.pq,
            classical_public_key=self._classical_public,
            classical_private_key=classical_private_bytes,
            pq_public_key=self._pq_public,
            pq_private_key=self._pq_private,
            combined_public_key=combined_public,
            security_level=f"{self.security_bits}-bit hybrid"
        )
    
    def _use_kyber_simulator(self, variant: int):
        """Fall back to Kyber simulator"""
        self._kyber_sim = KyberSimulator(variant, self.logger.logger)
        keypair = self._kyber_sim.keygen()
        self._pq_public = keypair.public_key
        self._pq_private = keypair.private_key
    
    def _combine_keys(self, classical: bytes, pq: bytes) -> bytes:
        """Combine keys with length prefix for unambiguous parsing"""
        # Format: [2-byte classical len][classical key][2-byte pq len][pq key]
        return (
            len(classical).to_bytes(2, 'big') + classical +
            len(pq).to_bytes(2, 'big') + pq
        )
    
    def _split_keys(self, combined: bytes) -> Tuple[bytes, bytes]:
        """Split combined key back into components"""
        offset = 0
        classical_len = int.from_bytes(combined[offset:offset+2], 'big')
        offset += 2
        classical = combined[offset:offset+classical_len]
        offset += classical_len
        pq_len = int.from_bytes(combined[offset:offset+2], 'big')
        offset += 2
        pq = combined[offset:offset+pq_len]
        return classical, pq
    
    def encapsulate(self, peer_public_key: bytes) -> Tuple[HybridCiphertext, bytes]:
        """
        Encapsulate to produce ciphertext and shared secret
        
        Args:
            peer_public_key: Combined public key from keygen()
        
        Returns:
            (HybridCiphertext, shared_secret)
        """
        # Split peer's combined public key
        classical_pk, pq_pk = self._split_keys(peer_public_key)
        
        # Classical key exchange
        if self.classical == 'x25519':
            if not X25519_AVAILABLE:
                raise RuntimeError("X25519 not available")
            ephemeral_private = X25519PrivateKey.generate()
            ephemeral_public = ephemeral_private.public_key().public_bytes_raw()
            peer_x25519_public = X25519PublicKey.from_public_bytes(classical_pk)
            classical_shared = ephemeral_private.exchange(peer_x25519_public)
            classical_ct = ephemeral_public  # X25519 "ciphertext" is ephemeral public key
        else:
            # ECDH
            if not ECDH_AVAILABLE:
                raise RuntimeError("ECDH not available")
            curve_map = {
                'ecdh_p256': ec.SECP256R1(),
                'ecdh_p384': ec.SECP384R1(),
                'ecdh_p521': ec.SECP521R1(),
            }
            curve = curve_map[self.classical]
            ephemeral_private = ec.generate_private_key(curve)
            ephemeral_public = ephemeral_private.public_key().public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint
            )
            peer_ecdh_public = ec.EllipticCurvePublicKey.from_encoded_point(curve, classical_pk)
            classical_shared = ephemeral_private.exchange(ec.ECDH(), peer_ecdh_public)
            classical_ct = ephemeral_public
        
        # PQ encapsulation
        pq_variant = self.PQ_ALGORITHMS[self.pq][0]
        if self.oqs and self.oqs.is_available:
            oqs_kem = self.oqs.create_kem(self.pq)
            if oqs_kem:
                pq_ct, pq_shared = oqs_kem.encapsulate(pq_pk)
            else:
                pq_ct, pq_shared = self._encap_simulator(pq_pk, pq_variant)
        else:
            pq_ct, pq_shared = self._encap_simulator(pq_pk, pq_variant)
        
        # Combine shared secrets using HKDF
        combined_ct = self._combine_keys(classical_ct, pq_ct)
        kdf_info = f"hybrid-{self.classical}-{self.pq}".encode()
        
        if HKDF_AVAILABLE:
            shared_secret = HKDF(
                algorithm=hashes.SHA384(),
                length=48,  # 384 bits
                salt=None,
                info=kdf_info,
            ).derive(classical_shared + pq_shared)
        else:
            # Fallback: simple concatenation and hash
            import hashlib
            shared_secret = hashlib.sha384(classical_shared + pq_shared + kdf_info).digest()
        
        ciphertext = HybridCiphertext(
            classical_ciphertext=classical_ct,
            pq_ciphertext=pq_ct,
            combined_ciphertext=combined_ct,
            kdf_info=kdf_info
        )
        
        self.logger.debug("Hybrid encapsulation complete",
                         classical_ct_size=len(classical_ct),
                         pq_ct_size=len(pq_ct))
        
        return ciphertext, shared_secret
    
    def _encap_simulator(self, pq_pk: bytes, variant: int) -> Tuple[bytes, bytes]:
        """Encapsulate using Kyber simulator"""
        sim = KyberSimulator(variant, self.logger.logger)
        return sim.encapsulate(pq_pk)
    
    def decapsulate(self, ciphertext: HybridCiphertext, 
                    keypair: HybridKeyPair) -> bytes:
        """
        Decapsulate to recover shared secret
        
        Args:
            ciphertext: HybridCiphertext from encapsulate()
            keypair: HybridKeyPair from keygen()
        
        Returns:
            shared_secret (bytes)
        """
        # Classical decapsulation
        if self.classical == 'x25519':
            if not X25519_AVAILABLE:
                raise RuntimeError("X25519 not available")
            private_key = X25519PrivateKey.from_private_bytes(keypair.classical_private_key)
            peer_ephemeral = X25519PublicKey.from_public_bytes(ciphertext.classical_ciphertext)
            classical_shared = private_key.exchange(peer_ephemeral)
        else:
            # ECDH
            if not ECDH_AVAILABLE:
                raise RuntimeError("ECDH not available")
            curve_map = {
                'ecdh_p256': ec.SECP256R1(),
                'ecdh_p384': ec.SECP384R1(),
                'ecdh_p521': ec.SECP521R1(),
            }
            curve = curve_map[self.classical]
            private_key = serialization.load_der_private_key(
                keypair.classical_private_key, password=None
            )
            peer_ephemeral = ec.EllipticCurvePublicKey.from_encoded_point(
                curve, ciphertext.classical_ciphertext
            )
            classical_shared = private_key.exchange(ec.ECDH(), peer_ephemeral)
        
        # PQ decapsulation
        pq_variant = self.PQ_ALGORITHMS[self.pq][0]
        if self.oqs and self.oqs.is_available:
            oqs_kem = self.oqs.create_kem(self.pq)
            if oqs_kem:
                pq_shared = oqs_kem.decapsulate(ciphertext.pq_ciphertext, keypair.pq_private_key)
            else:
                pq_shared = self._decap_simulator(ciphertext.pq_ciphertext, 
                                                   keypair.pq_private_key, pq_variant)
        else:
            pq_shared = self._decap_simulator(ciphertext.pq_ciphertext,
                                               keypair.pq_private_key, pq_variant)
        
        # Derive shared secret
        kdf_info = ciphertext.kdf_info
        
        if HKDF_AVAILABLE:
            shared_secret = HKDF(
                algorithm=hashes.SHA384(),
                length=48,
                salt=None,
                info=kdf_info,
            ).derive(classical_shared + pq_shared)
        else:
            import hashlib
            shared_secret = hashlib.sha384(classical_shared + pq_shared + kdf_info).digest()
        
        self.logger.debug("Hybrid decapsulation complete")
        return shared_secret
    
    def _decap_simulator(self, ciphertext: bytes, private_key: bytes, variant: int) -> bytes:
        """Decapsulate using Kyber simulator"""
        sim = KyberSimulator(variant, self.logger.logger)
        return sim.decapsulate(private_key, ciphertext)


class HybridSignatureScheme:
    """
    Hybrid Digital Signature combining classical and post-quantum signatures
    
    Both signatures are computed and concatenated. Verification requires
    BOTH signatures to be valid. This provides defense in depth.
    
    Supported combinations:
    - Ed25519 + Dilithium3 (default, recommended)
    - Ed25519 + Dilithium5 (higher security)
    - ECDSA-P256 + Dilithium3
    - ECDSA-P384 + Dilithium5
    """
    
    CLASSICAL_ALGORITHMS = {
        'ed25519': ('Ed25519', 128),
        'ecdsa_p256': ('ECDSA-P256', 128),
        'ecdsa_p384': ('ECDSA-P384', 192),
    }
    
    PQ_ALGORITHMS = {
        'dilithium2': (2, 128),
        'dilithium3': (3, 192),
        'dilithium5': (5, 256),
    }
    
    def __init__(self,
                 classical: str = 'ed25519',
                 pq: str = 'dilithium3',
                 oqs_wrapper: OQSWrapper = None,
                 logger: logging.Logger = None):
        """
        Initialize hybrid signature scheme
        
        Args:
            classical: Classical algorithm ('ed25519', 'ecdsa_p256', 'ecdsa_p384')
            pq: Post-quantum algorithm ('dilithium2', 'dilithium3', 'dilithium5')
            oqs_wrapper: OQS wrapper instance
            logger: Logger instance
        """
        if classical not in self.CLASSICAL_ALGORITHMS:
            raise ValueError(f"Unsupported classical algorithm: {classical}")
        if pq not in self.PQ_ALGORITHMS:
            raise ValueError(f"Unsupported PQ algorithm: {pq}")
        
        self.classical = classical
        self.pq = pq
        self.oqs = oqs_wrapper
        self.logger = ModuleLogger('HybridSig', logger)
        
        self._classical_private: Optional[Any] = None
        self._classical_public: Optional[bytes] = None
        self._pq_public: Optional[bytes] = None
        self._pq_private: Optional[bytes] = None
        self._dilithium_sim: Optional['DilithiumSimulator'] = None
        
        _, classical_bits = self.CLASSICAL_ALGORITHMS[classical]
        _, pq_bits = self.PQ_ALGORITHMS[pq]
        self.security_bits = max(classical_bits, pq_bits)
        
        self.logger.info(f"Initialized {classical}+{pq} hybrid signature",
                        security_bits=self.security_bits)
    
    def keygen(self) -> HybridKeyPair:
        """Generate hybrid signing key pair"""
        from cryptography.hazmat.primitives.asymmetric import ed25519
        
        # Generate classical key pair
        if self.classical == 'ed25519':
            self._classical_private = ed25519.Ed25519PrivateKey.generate()
            self._classical_public = self._classical_private.public_key().public_bytes_raw()
            classical_private_bytes = self._classical_private.private_bytes_raw()
        else:
            # ECDSA
            if not ECDH_AVAILABLE:
                raise RuntimeError("ECDSA not available")
            curve_map = {
                'ecdsa_p256': ec.SECP256R1(),
                'ecdsa_p384': ec.SECP384R1(),
            }
            curve = curve_map[self.classical]
            self._classical_private = ec.generate_private_key(curve)
            self._classical_public = self._classical_private.public_key().public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint
            )
            classical_private_bytes = self._classical_private.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            )
        
        # Generate PQ key pair
        pq_level = self.PQ_ALGORITHMS[self.pq][0]
        
        if self.oqs and self.oqs.is_available:
            oqs_sig = self.oqs.create_sig(self.pq)
            if oqs_sig:
                self._pq_public, self._pq_private = oqs_sig.keygen()
            else:
                self._use_dilithium_simulator(pq_level)
        else:
            self._use_dilithium_simulator(pq_level)
        
        combined_public = self._combine_keys(self._classical_public, self._pq_public)
        
        return HybridKeyPair(
            classical_algorithm=self.classical,
            pq_algorithm=self.pq,
            classical_public_key=self._classical_public,
            classical_private_key=classical_private_bytes,
            pq_public_key=self._pq_public,
            pq_private_key=self._pq_private,
            combined_public_key=combined_public,
            security_level=f"{self.security_bits}-bit hybrid"
        )
    
    def _use_dilithium_simulator(self, level: int):
        """Fall back to Dilithium simulator"""
        self._dilithium_sim = DilithiumSimulator(level, self.logger.logger)
        keypair = self._dilithium_sim.keygen()
        self._pq_public = keypair.public_key
        self._pq_private = keypair.private_key
    
    def _combine_keys(self, classical: bytes, pq: bytes) -> bytes:
        """Combine keys with length prefix"""
        return (
            len(classical).to_bytes(2, 'big') + classical +
            len(pq).to_bytes(2, 'big') + pq
        )
    
    def _split_keys(self, combined: bytes) -> Tuple[bytes, bytes]:
        """Split combined key"""
        offset = 0
        classical_len = int.from_bytes(combined[offset:offset+2], 'big')
        offset += 2
        classical = combined[offset:offset+classical_len]
        offset += classical_len
        pq_len = int.from_bytes(combined[offset:offset+2], 'big')
        offset += 2
        pq = combined[offset:offset+pq_len]
        return classical, pq
    
    def sign(self, message: bytes, keypair: HybridKeyPair) -> HybridSignature:
        """
        Sign message with both classical and PQ schemes
        
        Args:
            message: Message to sign
            keypair: HybridKeyPair from keygen()
        
        Returns:
            HybridSignature
        """
        from cryptography.hazmat.primitives.asymmetric import ed25519
        import hashlib
        
        # Hash message first for consistency
        message_hash = hashlib.sha384(message).digest()
        
        # Classical signature
        if self.classical == 'ed25519':
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
                keypair.classical_private_key
            )
            classical_sig = private_key.sign(message)
        else:
            # ECDSA
            private_key = serialization.load_der_private_key(
                keypair.classical_private_key, password=None
            )
            classical_sig = private_key.sign(
                message,
                ec.ECDSA(hashes.SHA384())
            )
        
        # PQ signature
        pq_level = self.PQ_ALGORITHMS[self.pq][0]
        if self.oqs and self.oqs.is_available:
            oqs_sig = self.oqs.create_sig(self.pq)
            if oqs_sig:
                pq_sig = oqs_sig.sign(message, keypair.pq_private_key)
            else:
                pq_sig = self._sign_simulator(message, keypair.pq_private_key, pq_level)
        else:
            pq_sig = self._sign_simulator(message, keypair.pq_private_key, pq_level)
        
        # Combine signatures
        combined_sig = self._combine_keys(classical_sig, pq_sig)
        
        self.logger.debug("Hybrid signature created",
                         classical_sig_size=len(classical_sig),
                         pq_sig_size=len(pq_sig))
        
        return HybridSignature(
            classical_algorithm=self.classical,
            pq_algorithm=self.pq,
            classical_signature=classical_sig,
            pq_signature=pq_sig,
            combined_signature=combined_sig,
            message_hash=message_hash
        )
    
    def _sign_simulator(self, message: bytes, private_key: bytes, level: int) -> bytes:
        """Sign using Dilithium simulator"""
        sim = DilithiumSimulator(level, self.logger.logger)
        return sim.sign(private_key, message)
    
    def verify(self, message: bytes, signature: HybridSignature,
               public_key: bytes) -> bool:
        """
        Verify hybrid signature (BOTH must be valid)
        
        Args:
            message: Original message
            signature: HybridSignature to verify
            public_key: Combined public key from keygen()
        
        Returns:
            True if BOTH signatures are valid
        """
        from cryptography.hazmat.primitives.asymmetric import ed25519
        
        classical_pk, pq_pk = self._split_keys(public_key)
        
        # Verify classical signature
        try:
            if self.classical == 'ed25519':
                public = ed25519.Ed25519PublicKey.from_public_bytes(classical_pk)
                public.verify(signature.classical_signature, message)
                classical_valid = True
            else:
                # ECDSA
                curve_map = {
                    'ecdsa_p256': ec.SECP256R1(),
                    'ecdsa_p384': ec.SECP384R1(),
                }
                curve = curve_map[self.classical]
                public = ec.EllipticCurvePublicKey.from_encoded_point(curve, classical_pk)
                public.verify(signature.classical_signature, message, ec.ECDSA(hashes.SHA384()))
                classical_valid = True
        except Exception as e:
            self.logger.warning(f"Classical signature verification failed: {e}")
            classical_valid = False
        
        # Verify PQ signature
        pq_level = self.PQ_ALGORITHMS[self.pq][0]
        if self.oqs and self.oqs.is_available:
            oqs_sig = self.oqs.create_sig(self.pq)
            if oqs_sig:
                pq_valid = oqs_sig.verify(message, signature.pq_signature, pq_pk)
            else:
                pq_valid = self._verify_simulator(message, signature.pq_signature, pq_pk, pq_level)
        else:
            pq_valid = self._verify_simulator(message, signature.pq_signature, pq_pk, pq_level)
        
        result = classical_valid and pq_valid
        self.logger.debug("Hybrid signature verification",
                         classical_valid=classical_valid,
                         pq_valid=pq_valid,
                         result=result)
        
        return result
    
    def _verify_simulator(self, message: bytes, signature: bytes, 
                          public_key: bytes, level: int) -> bool:
        """Verify using Dilithium simulator"""
        sim = DilithiumSimulator(level, self.logger.logger)
        return sim.verify(public_key, message, signature)


# =============================================================================
# Quantum Attack Simulation (v1.9.4)
# =============================================================================

class QuantumAttackSimulator:
    """
    Simulate quantum attacks using Qiskit for validation
    
    Provides educational simulation of:
    - Shor's algorithm (factoring, discrete log)
    - Grover's algorithm (search speedup)
    - Quantum key distribution (BB84)
    
    Note: These are simplified simulations for testing, not actual attacks.
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = ModuleLogger('QuantumSim', logger)
        self._backend = None
        
        if QISKIT_AVAILABLE:
            try:
                from qiskit_aer import AerSimulator
                self._backend = AerSimulator()
                self.logger.info("Qiskit simulator initialized")
            except Exception as e:
                self.logger.warning(f"Qiskit backend unavailable: {e}")
    
    @property
    def is_available(self) -> bool:
        return QISKIT_AVAILABLE and self._backend is not None
    
    def simulate_grovers_speedup(self, search_space_bits: int) -> Dict[str, Any]:
        """
        Simulate Grover's algorithm speedup for brute-force attacks
        
        Classical: O(2^n) queries
        Quantum: O(2^(n/2)) queries (quadratic speedup)
        
        Args:
            search_space_bits: Size of search space in bits (e.g., 128 for AES-128)
        
        Returns:
            Analysis of Grover speedup impact
        """
        classical_queries = 2 ** search_space_bits
        quantum_queries = 2 ** (search_space_bits // 2)
        
        # Estimate time (assuming 1 billion queries/second classical, 1 million quantum)
        classical_time_years = classical_queries / (1e9 * 365.25 * 24 * 3600)
        quantum_time_years = quantum_queries / (1e6 * 365.25 * 24 * 3600)
        
        result = {
            'algorithm': 'grovers',
            'search_space_bits': search_space_bits,
            'classical_queries': classical_queries,
            'quantum_queries': quantum_queries,
            'speedup_factor': classical_queries / quantum_queries,
            'classical_time_years': classical_time_years,
            'quantum_time_years': quantum_time_years,
            'effective_security_bits': search_space_bits // 2,
            'recommendation': self._grover_recommendation(search_space_bits)
        }
        
        self.logger.info("Grover speedup analysis",
                        original_bits=search_space_bits,
                        effective_bits=search_space_bits // 2)
        
        return result
    
    def _grover_recommendation(self, bits: int) -> str:
        """Generate recommendation based on Grover impact"""
        effective = bits // 2
        if effective >= 128:
            return "Secure against Grover's algorithm (128+ effective bits)"
        elif effective >= 80:
            return "Marginally secure - consider doubling key size"
        else:
            return f"VULNERABLE: Only {effective} effective bits - double key size immediately"
    
    def simulate_shors_attack(self, algorithm: str, key_bits: int) -> Dict[str, Any]:
        """
        Simulate Shor's algorithm impact on asymmetric crypto
        
        Shor's provides exponential speedup for:
        - Integer factorization (RSA)
        - Discrete logarithm (DH, ECDH, DSA, ECDSA)
        
        Args:
            algorithm: 'rsa', 'dh', 'ecdh', 'ecdsa', 'dsa'
            key_bits: Key size in bits
        
        Returns:
            Analysis of Shor attack impact
        """
        # Estimate qubits needed (simplified)
        if algorithm in ('rsa', 'dh', 'dsa'):
            qubits_needed = 2 * key_bits + 1  # 2n+1 qubits for Shor on n-bit numbers
            time_complexity = "polynomial"
        elif algorithm in ('ecdh', 'ecdsa'):
            qubits_needed = 6 * key_bits  # Roughly 6n for ECC
            time_complexity = "polynomial"
        else:
            qubits_needed = 0
            time_complexity = "unknown"
        
        # Current quantum computers: ~1000 qubits, noisy
        current_qubits = 1000
        feasibility = "infeasible" if qubits_needed > current_qubits * 10 else "potentially feasible"
        
        result = {
            'algorithm': 'shors',
            'target_algorithm': algorithm,
            'key_bits': key_bits,
            'qubits_required': qubits_needed,
            'time_complexity': time_complexity,
            'current_feasibility': feasibility,
            'quantum_resistant': False,
            'recommendation': f"Migrate to PQC: Kyber (KEM), Dilithium (signatures)"
        }
        
        self.logger.warning(f"Shor analysis: {algorithm}-{key_bits} vulnerable",
                           qubits_needed=qubits_needed)
        
        return result
    
    def run_qkd_simulation(self, num_qubits: int = 8) -> Dict[str, Any]:
        """
        Run simplified BB84 QKD simulation using Qiskit
        
        Demonstrates quantum key distribution principles:
        - Alice prepares qubits in random bases
        - Bob measures in random bases
        - They compare bases and extract key
        
        Args:
            num_qubits: Number of qubits to simulate
        
        Returns:
            QKD simulation results
        """
        if not self.is_available:
            return {
                'success': False,
                'error': 'Qiskit not available',
                'recommendation': 'Install qiskit and qiskit-aer for quantum simulations'
            }
        
        import random
        from qiskit import QuantumCircuit
        
        # Alice's random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(num_qubits)]
        alice_bases = [random.randint(0, 1) for _ in range(num_qubits)]  # 0=Z, 1=X
        
        # Bob's random measurement bases
        bob_bases = [random.randint(0, 1) for _ in range(num_qubits)]
        
        # Build circuit
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Alice encodes
        for i in range(num_qubits):
            if alice_bits[i] == 1:
                qc.x(i)
            if alice_bases[i] == 1:  # X basis
                qc.h(i)
        
        qc.barrier()
        
        # Bob measures
        for i in range(num_qubits):
            if bob_bases[i] == 1:  # X basis
                qc.h(i)
            qc.measure(i, i)
        
        # Run simulation
        from qiskit import transpile
        qc_transpiled = transpile(qc, self._backend)
        job = self._backend.run(qc_transpiled, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Extract Bob's measured bits
        measured_bits_str = list(counts.keys())[0]
        bob_bits = [int(b) for b in reversed(measured_bits_str)]
        
        # Sift key (keep only matching bases)
        sifted_key = []
        matching_indices = []
        for i in range(num_qubits):
            if alice_bases[i] == bob_bases[i]:
                sifted_key.append(alice_bits[i])
                matching_indices.append(i)
        
        # Calculate error rate
        errors = sum(1 for i in matching_indices if alice_bits[i] != bob_bits[i])
        error_rate = errors / len(matching_indices) if matching_indices else 0
        
        return {
            'success': True,
            'num_qubits': num_qubits,
            'alice_bits': alice_bits,
            'alice_bases': alice_bases,
            'bob_bases': bob_bases,
            'bob_bits': bob_bits,
            'matching_bases': len(matching_indices),
            'sifted_key_length': len(sifted_key),
            'sifted_key': sifted_key,
            'error_rate': error_rate,
            'eavesdropper_detected': error_rate > 0.11,  # BB84 threshold
            'key_bits_per_qubit': len(sifted_key) / num_qubits
        }
    
    def validate_hybrid_scheme(self, classical: str, pq: str) -> Dict[str, Any]:
        """
        Validate a hybrid scheme against quantum attacks
        
        Args:
            classical: Classical algorithm (e.g., 'x25519', 'ecdsa_p256')
            pq: Post-quantum algorithm (e.g., 'kyber768', 'dilithium3')
        
        Returns:
            Security analysis of hybrid scheme
        """
        classical_analysis = {
            'x25519': {'type': 'ecdh', 'bits': 255, 'classical_secure': True, 'quantum_secure': False},
            'ecdh_p256': {'type': 'ecdh', 'bits': 256, 'classical_secure': True, 'quantum_secure': False},
            'ecdh_p384': {'type': 'ecdh', 'bits': 384, 'classical_secure': True, 'quantum_secure': False},
            'ed25519': {'type': 'ecdsa', 'bits': 255, 'classical_secure': True, 'quantum_secure': False},
            'ecdsa_p256': {'type': 'ecdsa', 'bits': 256, 'classical_secure': True, 'quantum_secure': False},
            'ecdsa_p384': {'type': 'ecdsa', 'bits': 384, 'classical_secure': True, 'quantum_secure': False},
        }
        
        pq_analysis = {
            'kyber512': {'nist_level': 1, 'classical_secure': True, 'quantum_secure': True},
            'kyber768': {'nist_level': 3, 'classical_secure': True, 'quantum_secure': True},
            'kyber1024': {'nist_level': 5, 'classical_secure': True, 'quantum_secure': True},
            'dilithium2': {'nist_level': 2, 'classical_secure': True, 'quantum_secure': True},
            'dilithium3': {'nist_level': 3, 'classical_secure': True, 'quantum_secure': True},
            'dilithium5': {'nist_level': 5, 'classical_secure': True, 'quantum_secure': True},
        }
        
        classical_info = classical_analysis.get(classical, {})
        pq_info = pq_analysis.get(pq, {})
        
        return {
            'hybrid_scheme': f"{classical}+{pq}",
            'classical_component': {
                'algorithm': classical,
                **classical_info,
                'shor_vulnerable': True,
            },
            'pq_component': {
                'algorithm': pq,
                **pq_info,
                'shor_vulnerable': False,
            },
            'hybrid_security': {
                'classical_secure': classical_info.get('classical_secure', False) or pq_info.get('classical_secure', False),
                'quantum_secure': pq_info.get('quantum_secure', False),
                'defense_in_depth': True,
                'nist_level': pq_info.get('nist_level', 0),
            },
            'recommendation': f"Recommended: Both components provide security - "
                            f"classical component protects against unknown PQC weaknesses, "
                            f"PQ component protects against quantum attacks",
            'compliant_with': [
                'IETF draft-ietf-tls-hybrid-design',
                'NIST SP 800-186 (ECC)',
                'NIST FIPS 203/204 (Kyber/Dilithium)',
            ]
        }


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
